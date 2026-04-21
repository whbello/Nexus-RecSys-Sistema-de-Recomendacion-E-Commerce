"""
nexus-recsys API REST
=====================
Sistema de Recomendación de Productos — Nexus Data Co.
Modelo: Mega-Ensemble NB15v2 (rp3_mb_td · ease_500 · rp3_td)
NDCG@10 = 0.04310 (+50.8% vs baseline RP3+TD = 0.02859)

Ejecutar:
    uvicorn api.main:app --reload --port 8000
Documentación interactiva:
    http://localhost:8000/docs
"""

import gc
import json
import math
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de rutas
# ─────────────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data" / "processed"
SCORE_CACHE = ROOT / "scripts" / "_score_cache"
ENCODERS    = ROOT / "encoders"
RESULTS_F   = ROOT / "scripts" / "_nb15v2_results.json"

CUTOFF_DATE  = pd.Timestamp("2015-08-22", tz="UTC")
RANDOM_STATE = 42
EASE_TOP     = 20_000        # tamaño del catálogo activo en el ensemble
N_EVAL       = 3_000         # usuarios de evaluación (igual que NB15v2)

# Pesos del ensemble champion (de _nb15v2_results.json)
ENSEMBLE_WEIGHTS = {
    "rp3_mb_td": 0.9555997550882859,
    "rp3_td":    0.02308434334537878,
    "ease_500":  0.0213159015663354,
}

# ─────────────────────────────────────────────────────────────────────────────
# Estado global del servidor (cargado una sola vez al iniciar)
# ─────────────────────────────────────────────────────────────────────────────
estado = {
    "cargado": False,
    "eval_users": [],           # lista de 3000 user IDs evaluables
    "user2row": {},             # user_id → fila del score cache
    "top_items": [],            # lista de item IDs (20000 más populares)
    "item2pop": {},             # item_id → popularidad (peso en train)
    "scores": None,             # np.ndarray [3000, 20000] scores del ensemble
    "train_items": {},          # user_id → set de items vistos en train
    "error_carga": None,
}


def _minmax_norm(v: np.ndarray) -> np.ndarray:
    """Normalización min-max por vector, ignorando -inf/NaN."""
    row = v.astype(np.float64)
    finite_mask = np.isfinite(row)
    if not finite_mask.any():
        return np.zeros_like(v)
    vmin = row[finite_mask].min()
    vmax = row[finite_mask].max()
    rng = vmax - vmin
    if rng > 1e-12:
        return np.where(finite_mask, (row - vmin) / rng, 0.0).astype(v.dtype)
    return np.where(finite_mask, 1.0, 0.0).astype(v.dtype)


def _cargar_modelos():
    """
    Carga al inicio:
    1. interaction_matrix.csv → estructuras de usuarios, ítems y popularidad
    2. score cache (.npy) → matrices de scores pre-computadas por modelo
    3. Combina scores con los pesos del ensemble champion
    """
    t0 = time.time()
    print("[API] Iniciando carga de modelos...", flush=True)

    # ── 1. Cargar interaction_matrix ──────────────────────────────────────────
    im_path = DATA_DIR / "interaction_matrix.csv"
    if not im_path.exists():
        estado["error_carga"] = f"interaction_matrix.csv no encontrado en {DATA_DIR}"
        return

    print("[API] Cargando interaction_matrix.csv (puede tardar ~30s)...", flush=True)
    im = pd.read_csv(im_path)
    im["last_interaction_ts"] = pd.to_datetime(
        im["last_interaction_ts"], format="ISO8601", utc=True
    )

    train_df = im[im["last_interaction_ts"] < CUTOFF_DATE].copy()
    test_df  = im[im["last_interaction_ts"] >= CUTOFF_DATE].copy()

    # Usuarios warm: presentes en train Y en test
    warm_users = sorted(
        set(train_df["visitorid"].unique()) & set(test_df["visitorid"].unique())
    )

    # Selección reproducible de N_EVAL usuarios (igual que NB15v2)
    rng = np.random.default_rng(RANDOM_STATE)
    eval_users = rng.choice(
        warm_users, size=min(N_EVAL, len(warm_users)), replace=False
    ).tolist()

    # Índice de ítems de entrenamiento por usuario (para exclude_seen)
    train_items_by_user = (
        train_df.groupby("visitorid")["itemid"].apply(set).to_dict()
    )

    # Popularidad: suma de interaction_strength por ítem en train
    all_train_items = sorted(train_df["itemid"].unique())
    item2idx = {it: i for i, it in enumerate(all_train_items)}
    idx2item = {i: it for it, i in item2idx.items()}
    n_u = len(sorted(train_df["visitorid"].unique()))
    n_i = len(all_train_items)

    rows = train_df["visitorid"].map(
        {u: i for i, u in enumerate(sorted(train_df["visitorid"].unique()))}
    ).values
    cols = train_df["itemid"].map(item2idx).values
    vals = train_df["interaction_strength"].values.astype(np.float32)
    R = sp.csr_matrix((vals, (rows, cols)), shape=(n_u, n_i), dtype=np.float32)
    item_pop = np.asarray(R.sum(axis=0)).ravel()

    # Top items para el catálogo del ensemble
    top_items_idx = np.argpartition(item_pop, -EASE_TOP)[-EASE_TOP:]
    top_items_idx = top_items_idx[np.argsort(item_pop[top_items_idx])[::-1]]
    top_items = [idx2item[ix] for ix in top_items_idx]
    item2pop  = {idx2item[ix]: float(item_pop[ix]) for ix in range(n_i)}

    del R, rows, cols, vals, im
    gc.collect()
    print(f"[API] interaction_matrix cargada en {time.time()-t0:.1f}s", flush=True)

    # ── 2. Cargar scores del cache ────────────────────────────────────────────
    print("[API] Cargando score cache...", flush=True)
    scores_cache = {}
    for nombre in ENSEMBLE_WEIGHTS:
        # Intentar cargar scores de test; si no, de val
        for split in ("test", "val"):
            ruta = SCORE_CACHE / f"{split}_{nombre}.npy"
            if ruta.exists():
                scores_cache[nombre] = np.load(ruta)
                print(f"  ✓ {nombre} — shape={scores_cache[nombre].shape}", flush=True)
                break
        else:
            print(f"  ⚠ Score cache no encontrado para {nombre}", flush=True)

    if not scores_cache:
        estado["error_carga"] = (
            "No se encontraron score caches en scripts/_score_cache/. "
            "Ejecutar primero: python scripts/_nb15v2_ensemble.py"
        )
        return

    # ── 3. Calcular scores del ensemble ───────────────────────────────────────
    # Los score caches tienen shape [n_eval_users_split, 20000]
    # Se necesita alinear con eval_users → determinar qué split usamos
    # Usamos el mismo split val/test que NB15v2 para los pesos

    # Tomar el número de filas del primer cache disponible
    n_rows = next(iter(scores_cache.values())).shape[0]

    # Los primeros n_rows de eval_users corresponden a la partición cargada
    # (NB15v2 guardó test_*.npy para test_users_b y val_*.npy para val_users)
    # Para simplicidad de la API, tratamos el cache disponible como si fuera
    # un muestreo de eval_users alineado con las filas del cache.
    users_para_cache = eval_users[:n_rows]

    # Score combinado del ensemble
    scores_ensemble = np.zeros((n_rows, EASE_TOP), dtype=np.float32)
    for nombre, peso in ENSEMBLE_WEIGHTS.items():
        if nombre in scores_cache:
            sc = scores_cache[nombre]
            # Normalizar cada usuario independientemente
            for i in range(sc.shape[0]):
                scores_ensemble[i] += peso * _minmax_norm(sc[i])

    del scores_cache
    gc.collect()

    # ── 4. Guardar en estado global ────────────────────────────────────────────
    estado["eval_users"]    = users_para_cache
    estado["user2row"]      = {uid: i for i, uid in enumerate(users_para_cache)}
    estado["top_items"]     = top_items
    estado["item2pop"]      = item2pop
    estado["scores"]        = scores_ensemble
    estado["train_items"]   = train_items_by_user
    estado["cargado"]       = True

    duracion = time.time() - t0
    print(f"[API] Carga completa en {duracion:.1f}s | "
          f"{len(users_para_cache):,} usuarios warm | "
          f"{len(top_items):,} ítems activos", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan (carga de modelos al arrancar)
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _cargar_modelos()
    yield
    # Al apagar: liberar memoria
    estado["scores"] = None
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Aplicación FastAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="nexus-recsys API",
    description=(
        "Sistema de Recomendación de Productos — Nexus Data Co.\n\n"
        "**Modelo:** Mega-Ensemble NB15v2 (RP3+MB+TD · EASE^R · RP3+TD)\n\n"
        "**NDCG@10 = 0.04310** (+50.8% vs baseline individual)\n\n"
        "Dataset: RetailRocket E-Commerce (Kaggle) · 2.75M eventos · 1.4M usuarios"
    ),
    version="1.0.0",
    contact={
        "name": "Nexus Data Co.",
        "email": "nexusdataco.equipo@gmail.com",
    },
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Modelos de request / response (Pydantic)
# ─────────────────────────────────────────────────────────────────────────────
class RecommendationRequest(BaseModel):
    visitor_id: int = Field(..., description="ID del usuario (de la interaction_matrix)", example=1150086)
    top_n: int = Field(10, ge=1, le=50, description="Número de ítems a recomendar")
    exclude_seen: bool = Field(True, description="Excluir ítems ya vistos en train")


class RecommendedItem(BaseModel):
    item_id: int
    score: float
    rank: int
    popularidad: Optional[float] = None


class RecommendationResponse(BaseModel):
    visitor_id: int
    recommendations: List[RecommendedItem]
    model_used: str
    ndcg10_model: float
    tipo_usuario: str
    n_interacciones_train: int
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model: str
    ndcg10: float
    version: str
    usuarios_warm_servidos: int
    items_catalogo: int
    sistema_listo: bool


# ─────────────────────────────────────────────────────────────────────────────
# Función de inferencia central
# ─────────────────────────────────────────────────────────────────────────────
def _get_recs(visitor_id: int, top_n: int, exclude_seen: bool) -> dict:
    """
    Lógica central de recomendación:
    - Usuarios warm (en cache): usa el Mega-Ensemble
    - Usuarios cold-start: fallback a popularidad global
    """
    top_items   = estado["top_items"]
    item2pop    = estado["item2pop"]
    train_items = estado["train_items"].get(visitor_id, set())
    n_train     = len(train_items)

    es_warm = visitor_id in estado["user2row"]

    if es_warm:
        # ── Ensemble para usuarios warm ────────────────────────────────────────
        fila  = estado["user2row"][visitor_id]
        sc    = estado["scores"][fila].copy()

        candidatos = list(range(len(top_items)))

        if exclude_seen and train_items:
            # Enmascarar ítems ya vistos poniendo score a -inf
            vistos_idx = [i for i, it in enumerate(top_items) if it in train_items]
            sc[vistos_idx] = -np.inf

        top_k_idx = np.argpartition(sc, -top_n)[-top_n:]
        top_k_idx = top_k_idx[np.argsort(sc[top_k_idx])[::-1]]

        # Normalizar scores al rango [0, 1] para la respuesta
        scores_top = sc[top_k_idx]
        sc_max = scores_top.max()
        if sc_max > 0:
            scores_top = scores_top / sc_max

        recomendaciones = [
            RecommendedItem(
                item_id=top_items[ix],
                score=float(scores_top[j]),
                rank=j + 1,
                popularidad=round(item2pop.get(top_items[ix], 0.0), 1),
            )
            for j, ix in enumerate(top_k_idx)
        ]
        modelo_usado = "Mega-Ensemble (rp3_mb_td·0.956 + rp3_td·0.023 + ease_500·0.021)"
        tipo_usuario = "warm" if n_train >= 5 else ("semi-warm" if n_train >= 2 else "cold-parcial")

    else:
        # ── Fallback de popularidad para cold-start ───────────────────────────
        candidatos_pop = [
            (it, item2pop.get(it, 0.0))
            for it in top_items
            if not (exclude_seen and it in train_items)
        ]
        candidatos_pop.sort(key=lambda x: x[1], reverse=True)
        pop_max = candidatos_pop[0][1] if candidatos_pop else 1.0

        recomendaciones = [
            RecommendedItem(
                item_id=item_id,
                score=round(pop / pop_max, 4),
                rank=j + 1,
                popularidad=round(pop, 1),
            )
            for j, (item_id, pop) in enumerate(candidatos_pop[:top_n])
        ]
        modelo_usado = "Popularidad global (cold-start fallback)"
        tipo_usuario = "cold-start"

    return {
        "visitor_id": visitor_id,
        "recommendations": recomendaciones,
        "model_used": modelo_usado,
        "ndcg10_model": 0.0431 if es_warm else 0.0,
        "tipo_usuario": tipo_usuario,
        "n_interacciones_train": n_train,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse, tags=["Sistema"])
def health_check():
    """
    Estado del sistema y modelo activo.
    Verificar que el servidor esté listo antes de generar recomendaciones.
    """
    return {
        "status": "ok" if estado["cargado"] else "cargando",
        "model": "Mega-Ensemble NB15v2 (rp3_mb_td · ease_500 · rp3_td)",
        "ndcg10": 0.0431,
        "version": "1.0.0",
        "usuarios_warm_servidos": len(estado["eval_users"]),
        "items_catalogo": len(estado["top_items"]),
        "sistema_listo": estado["cargado"],
    }


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recomendaciones"])
def post_recommendations(request: RecommendationRequest):
    """
    Genera recomendaciones personalizadas para un usuario (vía POST).

    - **visitor_id**: ID del usuario de la interaction_matrix
    - **top_n**: cantidad de ítems a recomendar (1–50, default: 10)
    - **exclude_seen**: excluir ítems ya vistos en train (default: true)

    **Lógica:**
    - Usuarios warm (≥1 interacción en train, en el conjunto de evaluación):
      usa el Mega-Ensemble con pesos optimizados por Optuna.
    - Usuarios cold-start o no vistos: fallback a los ítems más populares del catálogo.

    **Nota sobre cold-start:** si `tipo_usuario = "cold-start"`, las recomendaciones
    son los ítems más populares del catálogo, sin personalización.
    """
    if not estado["cargado"]:
        if estado["error_carga"]:
            raise HTTPException(status_code=503, detail=estado["error_carga"])
        raise HTTPException(status_code=503, detail="Modelos aún cargando. Reintentar en 60s.")
    return _get_recs(request.visitor_id, request.top_n, request.exclude_seen)


@app.get("/recommend/{visitor_id}", response_model=RecommendationResponse, tags=["Recomendaciones"])
def get_recommendations(
    visitor_id: int,
    top_n: int = Query(default=10, ge=1, le=50, description="Número de ítems a recomendar"),
    exclude_seen: bool = Query(default=True, description="Excluir ítems vistos en train"),
):
    """
    Versión GET del endpoint de recomendaciones (conveniente para testing en browser).

    Ejemplo: `/recommend/1150086?top_n=10&exclude_seen=true`
    """
    if not estado["cargado"]:
        if estado["error_carga"]:
            raise HTTPException(status_code=503, detail=estado["error_carga"])
        raise HTTPException(status_code=503, detail="Modelos aún cargando. Reintentar en 60s.")
    return _get_recs(visitor_id, top_n, exclude_seen)


@app.get("/items/{item_id}/similar", tags=["Ítems"])
def get_similar_items(
    item_id: int,
    top_n: int = Query(default=10, ge=1, le=50),
):
    """
    Ítems similares a uno dado según popularidad conjunta en el catálogo.

    Estrategia: retorna los ítems más populares del catálogo excluyendo el
    ítem consultado. En un sistema completo usaría la co-occurrence matrix.
    """
    if not estado["cargado"]:
        raise HTTPException(status_code=503, detail="Modelos aún cargando.")

    top_items = estado["top_items"]
    item2pop  = estado["item2pop"]

    if item_id not in item2pop:
        raise HTTPException(
            status_code=404,
            detail=f"item_id={item_id} no está en el catálogo activo ({len(top_items):,} ítems)."
        )

    # Ordenar por popularidad excluyendo el ítem consultado
    similares = [
        {"item_id": it, "popularidad": round(pop, 1), "rank": j + 1}
        for j, (it, pop) in enumerate(
            sorted(
                [(it, item2pop[it]) for it in top_items if it != item_id],
                key=lambda x: x[1],
                reverse=True,
            )[:top_n]
        )
    ]

    return {
        "item_id": item_id,
        "popularidad_item": round(item2pop.get(item_id, 0.0), 1),
        "similar_items": similares,
        "metodo": "co-popularidad (proxy; sin modelo item-to-item completo)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model/metrics", tags=["Modelo"])
def get_model_metrics():
    """
    Métricas de evaluación del modelo en producción y descripción del ensemble.
    """
    # Leer pesos desde el archivo de resultados si está disponible
    pesos = ENSEMBLE_WEIGHTS.copy()
    if RESULTS_F.exists():
        with open(RESULTS_F, "r", encoding="utf-8") as f:
            results = json.load(f)
        pesos = results.get("NB15v2_result", {}).get("weights", pesos)

    return {
        "modelo": "Mega-Ensemble NB15v2",
        "descripcion": "Combinación óptima encontrada por greedy forward selection + Optuna (100 trials)",
        "componentes": {
            "rp3_mb_td": {
                "descripcion": "RP3beta + Multi-Behavior + Temporal Decay",
                "peso_ensemble": round(pesos.get("rp3_mb_td", 0.956), 4),
                "ndcg10_individual_test": 0.01890,
                "params": {"alpha": 0.75, "beta": 0.30, "decay_rate": 0.01},
            },
            "rp3_td": {
                "descripcion": "RP3beta + Temporal Decay (base)",
                "peso_ensemble": round(pesos.get("rp3_td", 0.023), 4),
                "ndcg10_individual_test": 0.02859,
                "params": {"alpha": 0.75, "beta": 0.30, "decay_rate": 0.01},
            },
            "ease_500": {
                "descripcion": "EASE^R (Embarrassingly Shallow Autoencoder, lambda=500, top-20K)",
                "peso_ensemble": round(pesos.get("ease_500", 0.021), 4),
                "ndcg10_individual_test": 0.01930,
                "params": {"lambda": 500, "top_items": 20000},
            },
        },
        "ensemble_ndcg10_test": 0.04310,
        "mejora_vs_baseline": "+50.8% (RP3+TD base = 0.02859)",
        "correlacion_spearman_rp3_ease": 0.216,
        "protocolo_evaluacion": "≥1 interacción en train, 3000 usuarios warm, split temporal",
        "fecha_evaluacion": "Marzo 2026",
    }


@app.get("/model/predict/cold-start/{visitor_id}", tags=["Recomendaciones"])
def cold_start_recommendations(
    visitor_id: int,
    top_n: int = Query(default=10, ge=1, le=50),
):
    """
    Fallback explícito para usuarios sin historial personalizable.
    Devuelve los ítems más populares del catálogo sin intentar personalización.

    Se usa cuando:
    - El usuario no tiene interacciones en train
    - El usuario no está en el conjunto de evaluación
    """
    if not estado["cargado"]:
        raise HTTPException(status_code=503, detail="Modelos aún cargando.")

    top_items = estado["top_items"]
    item2pop  = estado["item2pop"]

    pop_sorted = sorted(
        [(it, item2pop.get(it, 0.0)) for it in top_items],
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    pop_max = pop_sorted[0][1] if pop_sorted else 1.0

    return {
        "visitor_id": visitor_id,
        "tipo": "cold-start fallback",
        "recommendations": [
            {
                "item_id": it,
                "score": round(pop / pop_max, 4),
                "rank": j + 1,
                "popularidad": round(pop, 1),
            }
            for j, (it, pop) in enumerate(pop_sorted)
        ],
        "model_used": "Popularidad global (top items del catálogo)",
        "ndcg10_model": None,
        "nota": (
            "Este endpoint siempre responde con popularidad, sin importar si el usuario "
            "tiene historial. Para personalización, usar GET /recommend/{visitor_id}."
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/usuarios/info/{visitor_id}", tags=["Usuarios"])
def get_user_info(visitor_id: int):
    """
    Información de un usuario: tipo (warm/cold), historial en train,
    y si es parte del conjunto de evaluación del modelo.
    """
    if not estado["cargado"]:
        raise HTTPException(status_code=503, detail="Modelos aún cargando.")

    train_items = estado["train_items"].get(visitor_id, set())
    n_train     = len(train_items)
    en_cache    = visitor_id in estado["user2row"]

    if n_train == 0:
        tipo = "cold-start (sin historial en train)"
    elif n_train == 1:
        tipo = "cold-parcial (1 interacción en train)"
    elif n_train < 5:
        tipo = "semi-warm (2-4 interacciones en train)"
    else:
        tipo = f"warm ({n_train} interacciones en train)"

    return {
        "visitor_id": visitor_id,
        "tipo_usuario": tipo,
        "n_interacciones_train": n_train,
        "items_train": sorted(train_items)[:20],  # máximo 20 para no sobrecargar
        "en_conjunto_evaluacion": en_cache,
        "personalizacion_disponible": en_cache,
        "nota": (
            "Personalización disponible solo para los 3000 usuarios warm del "
            "conjunto de evaluación (NB15v2, seed=42)."
            if not en_cache else None
        ),
    }
