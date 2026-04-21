"""
generate_11_notebook.py
=======================
Fuente de verdad de notebooks/11_optimization_ensemble.ipynb

Secciones:
  A - Análisis de sensibilidad al protocolo de evaluación
  B - Optimización bayesiana de RP3beta con Optuna (50 trials, validación interna)
  C - Ensemble RP3beta_opt + EASE^R (sweep de pesos en validación)
  D - Tabla comparativa consolidada NB11
  E - Guardar artefactos y documentación

EJECUTAR:
    python scripts/generate_11_notebook.py
"""

import json
import textwrap
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
NB_OUT   = ROOT_DIR / "notebooks" / "11_optimization_ensemble.ipynb"

# ─── helpers ──────────────────────────────────────────────────────────────────

def py(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(src).lstrip("\n"),
    }

def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(src).lstrip("\n"),
    }

# ─── celdas ───────────────────────────────────────────────────────────────────
cells = []

# ════════════════════════════════════════════════════════════════════════════════
# ENCABEZADO
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
# 11 · Nexus RecSys — Optimización y Ensemble

**Objetivo:** Extraer el máximo rendimiento posible de los modelos clásicos ya
entrenados mediante tres intervenciones:

| Sección | Descripción |
|---------|-------------|
| **A** | Análisis de sensibilidad al protocolo de evaluación |
| **B** | Optimización bayesiana de RP3beta con Optuna (50 trials, val interno) |
| **C** | Ensemble RP3beta_opt + EASE^R (sweep de pesos en validación) |
| **D** | Tabla comparativa NB11 consolidada |

**Estado previo:**
- Modelo activo: RP3beta (α=0.85, β=0.25) → NDCG@10=0.0258 (NB09)
- Challenger: Mult-VAE^PR → NDCG@10=0.0255 (NB10)
- EASE^R (λ=500, top-20K) → NDCG@10=0.0193 (NB09)
"""))

# ════════════════════════════════════════════════════════════════════════════════
# SETUP
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md("## 0 - Setup e Importaciones"))

cells.append(py("""\
import math
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn.preprocessing import normalize as skl_normalize

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Rutas ────────────────────────────────────────────────────────────────────
HERE     = Path().resolve()
ROOT     = HERE.parent if (HERE.parent / "data").exists() else HERE
DATA_DIR = ROOT / "data" / "processed"
ENC_DIR  = ROOT / "encoders"
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

# ── Constantes globales ──────────────────────────────────────────────────────
RANDOM_STATE = 42
K_VALUES     = [5, 10, 20]
N_EVAL_USERS = 3_000
CUTOFF_DATE  = pd.Timestamp("2015-08-22", tz="UTC")

# Subespacio de ítems (idéntico a NB09)
EASE_TOP = 20_000
EASE_REG = 500.0

# RP3beta valores originales NB09
RP3_ALPHA_ORIG = 0.85
RP3_BETA_ORIG  = 0.25

# Optuna
N_OPTUNA_TRIALS = 50

print(f"Root     : {ROOT}")
print(f"Optuna   : {optuna.__version__}")
print(f"EASE_TOP : {EASE_TOP}  EASE_REG: {EASE_REG}")
print(f"RP3 orig : alpha={RP3_ALPHA_ORIG}  beta={RP3_BETA_ORIG}")
"""))

# ════════════════════════════════════════════════════════════════════════════════
# 1 - CARGA DE DATOS
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md("## 1 - Carga de Datos y Split Temporal"))

cells.append(py("""\
print("Cargando datos...")
t0 = time.time()
im  = pd.read_csv(DATA_DIR / "interaction_matrix.csv")
im["last_interaction_ts"] = pd.to_datetime(im["last_interaction_ts"], format="ISO8601", utc=True)
print(f"  IM: {im.shape}  [{time.time()-t0:.1f}s]")

# ── Split temporal (idéntico a NB09) ─────────────────────────────────────────
train_mask = im["last_interaction_ts"] < CUTOFF_DATE
train_df   = im[train_mask].copy()
test_df    = im[~train_mask].copy()

# Usuarios warm: tienen al menos 1 ítem en train Y 1 en test
warm_users = sorted(set(train_df["visitorid"].unique()) & set(test_df["visitorid"].unique()))

# Muestrear N_EVAL_USERS para evaluación (seed=42, idéntico a NB09)
rng = np.random.default_rng(RANDOM_STATE)
eval_users = rng.choice(warm_users, size=min(N_EVAL_USERS, len(warm_users)), replace=False).tolist()

# Diccionarios de test
test_items_by_user  = test_df.groupby("visitorid")["itemid"].apply(set).to_dict()
train_items_by_user = train_df.groupby("visitorid")["itemid"].apply(set).to_dict()
test_tx_by_user     = (
    test_df[test_df["last_interaction_type"] == "transaction"]
    .groupby("visitorid")["itemid"].apply(set).to_dict()
)

all_items_global = sorted(im["itemid"].unique())
n_items_global   = len(all_items_global)

n_test_tx = len(test_df[test_df["last_interaction_type"] == "transaction"])
baseline_conv = n_test_tx / (len(set(test_df["visitorid"])) * n_items_global)
n_buyers = sum(1 for u in eval_users if test_tx_by_user.get(u))

print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")
print(f"Warm users total: {len(warm_users):,}")
print(f"Eval users:       {len(eval_users):,}")
print(f"Compradores test: {n_buyers}")
print(f"Baseline P(compra|aleatorio): {baseline_conv:.2e}")
"""))

cells.append(py("""\
# ── Índices y matriz R (idéntica a NB09) ──────────────────────────────────────
all_train_users = sorted(train_df["visitorid"].unique())
all_train_items = sorted(train_df["itemid"].unique())
user2idx = {u: i for i, u in enumerate(all_train_users)}
item2idx = {it: i for i, it in enumerate(all_train_items)}
idx2user = {i: u for u, i in user2idx.items()}
idx2item = {i: it for it, i in item2idx.items()}
n_u = len(all_train_users)
n_i = len(all_train_items)

rows_r = train_df["visitorid"].map(user2idx).values
cols_r = train_df["itemid"].map(item2idx).values
vals_r = train_df["interaction_strength"].values.astype(np.float32)
R = sp.csr_matrix((vals_r, (rows_r, cols_r)), shape=(n_u, n_i), dtype=np.float32)

item_pop       = np.asarray(R.sum(axis=0)).ravel()
item_pop_dict  = {idx2item[i]: float(item_pop[i]) for i in range(n_i)}
n_total_train  = float(R.sum())

# Conteo de interacciones por usuario en train
train_counts_all = np.diff(R.indptr)   # (n_u,)

print(f"R: {R.shape}  nnz={R.nnz:,}")
print(f"Sparsidad: {1 - R.nnz/(n_u*n_i):.6f}")
print(f"avg interacciones/usuario: {train_counts_all.mean():.2f}")
print(f"mediana: {np.median(train_counts_all):.0f}")
"""))

cells.append(py("""\
# ── Cuántos eval_users tienen exactamente 1 ítem en train ─────────────────────
eval_train_counts = np.array([
    len(train_items_by_user.get(u, set())) for u in eval_users
])

for threshold in [1, 2, 3, 5, 10]:
    n = int((eval_train_counts == threshold).sum() if threshold == 1
            else (eval_train_counts < threshold).sum())
    label = (f"exactamente {threshold}" if threshold == 1
             else f"< {threshold}")
    print(f"  Usuarios con {label} ítem(s) en train: {n:,} "
          f"({n/len(eval_users)*100:.1f}%)")

print()
print(f"  Usuarios con >= 3 ítems en train: "
      f"{(eval_train_counts >= 3).sum():,} "
      f"({(eval_train_counts >= 3).mean()*100:.1f}%)")
print(f"  Usuarios con >= 5 ítems en train: "
      f"{(eval_train_counts >= 5).sum():,} "
      f"({(eval_train_counts >= 5).mean()*100:.1f}%)")
print(f"  Usuarios con >= 10 ítems en train: "
      f"{(eval_train_counts >= 10).sum():,} "
      f"({(eval_train_counts >= 10).mean()*100:.1f}%)")
"""))

# ════════════════════════════════════════════════════════════════════════════════
# 2 - FRAMEWORK DE EVALUACIÓN
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md("## 2 - Framework de Evaluación (idéntico a NB09)"))

cells.append(py("""\
# ── Métricas base ─────────────────────────────────────────────────────────────
def ndcg(r, rel, k):
    d = sum(1. / math.log2(i + 2) for i, x in enumerate(r[:k]) if x in rel)
    ideal = sum(1. / math.log2(i + 2) for i in range(min(len(rel), k)))
    return d / ideal if ideal else 0.

def prec(r, rel, k): return len(set(r[:k]) & rel) / k if k else 0.
def rec(r, rel, k):  return len(set(r[:k]) & rel) / len(rel) if rel else 0.
def ap(r, rel, k):
    if not rel: return 0.
    s, h = 0., 0
    for i, x in enumerate(r[:k]):
        if x in rel: h += 1; s += h / (i + 1)
    return s / min(len(rel), k)
def nov(flat, pd_, nt):
    return float(np.mean([-math.log2(pd_.get(x, 1) / nt + 1e-10) for x in flat])) if flat else 0.
def rev_k(r, tx, k): return len(set(r[:k]) & tx) / k if k else 0.
def ctr_k(r, ts, k): return len(set(r[:k]) & ts) / k if k else 0.

def evaluate(get_fn, evals, tst, tst_tx, pop_d, nt, cat_sz, bconv, ks=K_VALUES):
    \"\"\"
    Función de evaluación idéntica a NB09.
    Parámetros:
        get_fn  : callable(uid, n) -> list[item_id]
        evals   : lista de user_ids a evaluar
        tst     : dict user_id -> set(item_ids) en test
        tst_tx  : dict user_id -> set(item_ids) con transacción en test
        pop_d   : dict item_id -> popularidad (conteo en train)
        nt      : total de interacciones en train (float)
        cat_sz  : tamaño total del catálogo (int)
        bconv   : baseline de conversión aleatorio (float)
        ks      : lista de K values
    Retorna dict con NDCG@K, Precision@K, Recall@K, MAP@K, Revenue@K,
                     CTR@K, ConvLift@K, Coverage, Novelty, n_eval
    \"\"\"
    acc = {k: {m: [] for m in "prnm"} for k in ks}
    # Campos extra para revenue y ctr
    for k in ks:
        acc[k]["r2"] = []
        acc[k]["c"]  = []
    seen = set()
    ne = 0
    for uid in evals:
        ti = tst.get(uid, set())
        if not ti: continue
        tx = tst_tx.get(uid, set())
        mk = max(ks)
        try:
            recs = get_fn(uid, mk)
        except Exception:
            continue
        seen.update(recs)
        ne += 1
        for k in ks:
            acc[k]["p"].append(prec(recs, ti, k))
            acc[k]["r"].append(rec(recs,  ti, k))
            acc[k]["n"].append(ndcg(recs,  ti, k))
            acc[k]["m"].append(ap(recs,   ti, k))
            acc[k]["r2"].append(rev_k(recs, tx, k))
            acc[k]["c"].append(ctr_k(recs, ti, k))
    out = {"n_eval": ne}
    for k in ks:
        if not acc[k]["p"]: continue
        out[f"NDCG@{k}"]      = float(np.mean(acc[k]["n"]))
        out[f"Precision@{k}"] = float(np.mean(acc[k]["p"]))
        out[f"Recall@{k}"]    = float(np.mean(acc[k]["r"]))
        out[f"MAP@{k}"]       = float(np.mean(acc[k]["m"]))
        out[f"Revenue@{k}"]   = float(np.mean(acc[k]["r2"]))
        out[f"CTR@{k}"]       = float(np.mean(acc[k]["c"]))
        rv = out[f"Revenue@{k}"]
        out[f"ConvLift@{k}"]  = rv / bconv if bconv else 0.
    out["Coverage"] = len(seen) / cat_sz
    out["Novelty"]  = nov(list(seen), pop_d, nt)
    return out

print("Framework de evaluación definido  (idéntico a NB09).")
"""))

# ════════════════════════════════════════════════════════════════════════════════
# 3 - PRECOMPUTAR RP3beta ORIGINAL Y EASE^R
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md("## 3 - Precomputar Modelos Base (RP3beta original y EASE^R)"))

cells.append(py("""\
# ── Subespacio top-20K ítems (idéntico a NB09) ────────────────────────────────
top_items_idx = np.argpartition(item_pop, -EASE_TOP)[-EASE_TOP:]
top_items_idx = top_items_idx[np.argsort(item_pop[top_items_idx])[::-1]]
N_TOP = len(top_items_idx)
top_items_global = [idx2item[i] for i in top_items_idx]

# Submatriz sparse compartida por ambos modelos
X_top_csr = R[:, top_items_idx].astype(np.float32).tocsr()  # (n_u, N_TOP)
print(f"Subespacio top-{N_TOP}: X_top_csr {X_top_csr.shape}  nnz={X_top_csr.nnz:,}")
"""))

cells.append(py("""\
def build_rp3(alpha, beta, X_csr, pop_arr):
    \"\"\"
    Construye la matriz W_rp3 para los parámetros dados.
    Retorna W (np.ndarray float32, shape=[N_TOP, N_TOP]).
    \"\"\"
    P_ui = skl_normalize(X_csr.astype(np.float64), norm="l1", axis=1)
    P_it = skl_normalize(X_csr.T.tocsr().astype(np.float64), norm="l1", axis=1)
    pop_beta = np.power(pop_arr + 1e-10, beta)

    W_raw = P_it @ P_ui
    W = np.asarray(W_raw.todense() if hasattr(W_raw, "todense") else W_raw,
                   dtype=np.float32)
    del W_raw
    np.power(W, alpha, out=W)
    W = W / (pop_beta[None, :] + 1e-10)
    np.fill_diagonal(W, 0.)
    return W

def make_get_rp3(W, X_csr, top_global):
    \"\"\"Cierra un get_fn de RP3beta dado W precomputado.\"\"\"
    def get_fn(uid, n):
        if uid not in user2idx: return []
        ui = user2idx[uid]
        row = X_csr.getrow(ui)
        x_u = np.asarray(row.todense(), dtype=np.float32).ravel()
        sc = x_u @ W
        sc[x_u > 0] = -np.inf
        top = np.argpartition(sc, -n)[-n:]
        top = top[np.argsort(sc[top])[::-1]]
        return [top_global[i] for i in top]
    return get_fn

# ── RP3beta original ──────────────────────────────────────────────────────────
print("Construyendo RP3beta original (alpha=0.85, beta=0.25)...")
t0 = time.time()
pop_sub = item_pop[top_items_idx].astype(np.float64)
W_orig  = build_rp3(RP3_ALPHA_ORIG, RP3_BETA_ORIG, X_top_csr, pop_sub)
get_rp3_orig = make_get_rp3(W_orig, X_top_csr, top_items_global)
print(f"  W_orig: {W_orig.shape}  [{time.time()-t0:.1f}s]")
"""))

cells.append(py("""\
# ── EASE^R ────────────────────────────────────────────────────────────────────
print(f"Construyendo EASE^R (lambda={EASE_REG})...")
t0 = time.time()

G_sparse = X_top_csr.T @ X_top_csr
G = np.asarray(G_sparse.todense(), dtype=np.float32)
del G_sparse
print(f"  G: {G.shape}  [{time.time()-t0:.1f}s]")

G_reg  = G + EASE_REG * np.eye(N_TOP, dtype=np.float32)
del G
B_inv  = np.linalg.inv(G_reg)
del G_reg
diag_inv = np.diag(B_inv).copy()
B_ease   = -(B_inv / diag_inv[None, :]).astype(np.float32)
np.fill_diagonal(B_ease, 0.)
del B_inv
print(f"  B_ease: {B_ease.shape}  [{time.time()-t0:.1f}s]")

def get_ease(uid, n):
    if uid not in user2idx: return []
    ui  = user2idx[uid]
    row = X_top_csr.getrow(ui)
    x_u = np.asarray(row.todense(), dtype=np.float32).ravel()
    sc  = x_u @ B_ease
    sc[x_u > 0] = -np.inf
    top_local = np.argpartition(sc, -n)[-n:]
    top_local = top_local[np.argsort(sc[top_local])[::-1]]
    return [top_items_global[i] for i in top_local]

print("Modelos base listos.")
"""))

# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN A - ANÁLISIS DE PROTOCOLO
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Sección A — Análisis de Sensibilidad al Protocolo de Evaluación

### Motivación
Los valores de NDCG@10 reportados en la literatura (0.06–0.20 para los mejores modelos)
se obtienen con **protocolos de evaluación distintos** al nuestro:

- Filtrado de usuarios: literatura usa ≥5 o ≥20 interacciones en train
- Split: literatura frecuentemente usa leave-one-out en lugar de split temporal
- Catálogo: muchos papers filtran a un subconjunto de ítems populares

**Esta sección evalúa empíricamente el impacto de cada umbral de filtrado**
manteniendo el split temporal y el catálogo constantes.
"""))

cells.append(py("""\
# ── Análisis de sensibilidad al umbral mínimo de interacciones ────────────────
# Evaluamos RP3beta original bajo 5 protocolos distintos.
# IMPORTANTE: NO cambiamos el protocolo de referencia del proyecto.
# Esta sección es ANÁLISIS, no redefinición.

filtros = [
    (">=1 (todos)",  1),   # protocolo actual del proyecto
    (">=2",          2),
    (">=3",          3),
    (">=5",          5),
    (">=10",        10),
]

resultados_protocolo = []

for label, min_int in filtros:
    # Filtrar eval_users al subconjunto con >= min_int ítems en train
    subset = [
        u for u in eval_users
        if len(train_items_by_user.get(u, set())) >= min_int
    ]
    if len(subset) == 0:
        continue

    t_prot = time.time()
    m = evaluate(
        get_rp3_orig, subset,
        test_items_by_user, test_tx_by_user,
        item_pop_dict, n_total_train,
        n_items_global, baseline_conv
    )
    elapsed = time.time() - t_prot

    row = {
        "Protocolo":       label,
        "Min interacciones": min_int,
        "N usuarios":      len(subset),
        "% del total":     round(len(subset) / len(eval_users) * 100, 1),
        "NDCG@10":         round(m.get("NDCG@10", 0.), 5),
        "Precision@10":    round(m.get("Precision@10", 0.), 6),
        "Recall@10":       round(m.get("Recall@10", 0.), 6),
        "Coverage":        round(m.get("Coverage", 0.), 4),
        "Novelty":         round(m.get("Novelty", 0.), 2),
        "Tiempo (s)":      round(elapsed, 1),
    }
    resultados_protocolo.append(row)
    print(f"  [{label:12s}] N={len(subset):>4d} ({row['% del total']:>5.1f}%)  "
          f"NDCG@10={row['NDCG@10']:.5f}  Prec={row['Precision@10']:.5f}  "
          f"Cov={row['Coverage']:.4f}")

df_proto = pd.DataFrame(resultados_protocolo)
"""))

cells.append(py("""\
# ── Tabla comparativa de protocolos ──────────────────────────────────────────
print()
print("=" * 90)
print("TABLA A — Sensibilidad al protocolo de evaluación  (RP3beta α=0.85, β=0.25)")
print("=" * 90)
print(df_proto[[
    "Protocolo", "N usuarios", "% del total",
    "NDCG@10", "Precision@10", "Recall@10", "Coverage", "Novelty"
]].to_string(index=False))
print()

# Curva NDCG@10 vs filtro mínimo (texto)
print("CURVA  NDCG@10  vs  filtro mínimo de interacciones:")
print("-" * 50)
ndcg_base = df_proto.loc[df_proto["Min interacciones"] == 1, "NDCG@10"].values[0]
for _, row in df_proto.iterrows():
    bar_len = int(row["NDCG@10"] / 0.001)
    delta_pct = (row["NDCG@10"] - ndcg_base) / ndcg_base * 100
    sign = "+" if delta_pct >= 0 else ""
    print(f"  {row['Protocolo']:14s} | {'█' * min(bar_len, 60):<60s}  "
          f"{row['NDCG@10']:.5f}  ({sign}{delta_pct:.1f}%)")
print()
print(f"  Protocolo de referencia del proyecto: >= 1 (NDCG@10={ndcg_base:.5f})")
"""))

cells.append(md("""\
### Análisis del protocolo de evaluación

#### ¿Por qué el protocolo ≥1 es el más honesto para producción?

En un sistema de recomendación real, el servicio debe operar sobre **todos** los usuarios,
incluyendo aquellos con poco historial. Si evaluamos solo sobre usuarios con ≥5 ítems en
train, estamos midiendo el sistema en condiciones ideales que no representan a la mayoría
de la base de usuarios (>57% tienen ≤1 ítem en RetailRocket).

El protocolo ≥1 mide la **calidad real en producción**, donde:
- La mayoría de usuarios son cold o near-cold
- El modelo debe manejar correctamente la incertidumbre sobre preferencias
- Un sistema que funciona bien con ≥5 ítems pero falla con 1 ítem no es útil

#### ¿Por qué la literatura usa ≥5 o leave-one-out?

Los papers de investigación tienen objetivos distintos a los sistemas en producción:
1. **Sesgo de selección deliberado:** al filtrar a usuarios ≥5 interacciones, los papers
   eliminan los casos "difíciles" donde cualquier modelo tiene señal débil. Esto infla
   artificialmente los valores de NDCG.
2. **Leave-one-out:** evalúa solo el último ítem del usuario como "relevante" y mide si
   el modelo lo pone en top-10 de entre todo el catálogo. Favorece modelos secuenciales.
3. **Comparabilidad con papers previos:** los papers replican el protocolo de otros papers
   para comparación directa, propagando el sesgo de selección.

Ninguno de estos protocolos es incorrecto, pero representan **distribuciones distintas**
de usuarios y no son comparables directamente.

#### Protocolo de referencia para el proyecto

Mantenemos **≥1 como protocolo principal** por honestidad con la distribución real de
usuarios. Reportamos también la cifra ≥3 como referencia secundaria para contextualizar
con la literatura. Las métricas históricas de todos los notebooks anteriores **no
se modifican** — esta es una capa de análisis adicional, no un cambio de metodología.
"""))

# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN B - OPTUNA
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Sección B — Optimización Bayesiana de RP3beta con Optuna

Los hiperparámetros actuales (α=0.85, β=0.25) provienen del paper original de Paudel et al.
(2017), optimizados sobre ML-20M (18M interacciones, distribución muy distinta a
RetailRocket). Esta sección busca los valores óptimos específicamente para este dataset.

**Protocolo de optimización con triple split:**
- Train: datos hasta el cutoff (igual que antes)
- Validation (interno): 15% aleatorio de eval_users → solo para Optuna
- Test: 85% restante → evaluación final UNA SOLA VEZ por modelo
"""))

cells.append(py("""\
# ── B.1 — Validation split interno ──────────────────────────────────────────
# Separar eval_users en validation (15%) y test (85%)
# Estratificar por grupo de actividad para mantener distribución

# Calcular grupo de actividad de cada eval_user
activity_groups = []
for u in eval_users:
    cnt = len(train_items_by_user.get(u, set()))
    if cnt == 1:   activity_groups.append(0)  # cold
    elif cnt <= 4: activity_groups.append(1)  # tepid
    else:          activity_groups.append(2)  # warm

activity_groups = np.array(activity_groups)

# Split estratificado manual: 15% val, 85% test final
rng_split = np.random.default_rng(RANDOM_STATE)
val_mask  = np.zeros(len(eval_users), dtype=bool)

for g in [0, 1, 2]:
    idx_g = np.where(activity_groups == g)[0]
    if len(idx_g) == 0: continue
    n_val = max(1, int(len(idx_g) * 0.15))
    chosen = rng_split.choice(idx_g, size=n_val, replace=False)
    val_mask[chosen] = True

eval_arr     = np.array(eval_users)
val_users    = eval_arr[val_mask].tolist()
test_users_b = eval_arr[~val_mask].tolist()

print(f"Validation users  : {len(val_users):,} ({len(val_users)/len(eval_users)*100:.1f}%)")
print(f"Test users (B+C)  : {len(test_users_b):,} ({len(test_users_b)/len(eval_users)*100:.1f}%)")
print()
# Distribución de actividad en cada split
for name, users in [("val", val_users), ("test", test_users_b)]:
    cnts = np.array([len(train_items_by_user.get(u, set())) for u in users])
    print(f"  {name}: cold={( cnts==1).sum():>4d}  "
          f"tepid={(( cnts>1)&(cnts<5)).sum():>4d}  "
          f"warm={(cnts>=5).sum():>4d}")
"""))

cells.append(py("""\
# ── B.2 — Función objetivo Optuna ────────────────────────────────────────────

def objective(trial):
    \"\"\"Evalúa RP3beta con parámetros sugeridos sobre val_users.\"\"\"
    alpha = trial.suggest_float("alpha", 0.50, 1.00, step=0.05)
    beta  = trial.suggest_float("beta",  0.00, 0.50, step=0.05)

    W = build_rp3(alpha, beta, X_top_csr, pop_sub)
    get_fn = make_get_rp3(W, X_top_csr, top_items_global)

    m = evaluate(
        get_fn, val_users,
        test_items_by_user, test_tx_by_user,
        item_pop_dict, n_total_train,
        n_items_global, baseline_conv
    )
    return m.get("NDCG@10", 0.)

# ── B.3 — Ejecutar Optuna (50 trials) ────────────────────────────────────────
print(f"Ejecutando Optuna ({N_OPTUNA_TRIALS} trials)...")
print(f"  Espacio: alpha in [0.50, 1.00] step=0.05  |  beta in [0.00, 0.50] step=0.05")
print(f"  Evaluación sobre {len(val_users)} usuarios de validación")
print()

t_opt = time.time()
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
)
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
t_opt_elapsed = time.time() - t_opt

print(f"Optuna completado en {t_opt_elapsed:.1f}s  ({t_opt_elapsed/N_OPTUNA_TRIALS:.1f}s/trial)")
print()
print(f"  Mejor trial: #{study.best_trial.number}")
print(f"  Parámetros : alpha={study.best_params['alpha']:.2f}  beta={study.best_params['beta']:.2f}")
print(f"  NDCG@10 val: {study.best_value:.5f}")
print(f"  RP3beta orig: alpha={RP3_ALPHA_ORIG}  beta={RP3_BETA_ORIG}  NDCG@10 val: N/A (será calculado)")
"""))

cells.append(py("""\
# ── Tabla Top-10 trials ───────────────────────────────────────────────────────
trials_df = study.trials_dataframe()[
    ["number", "params_alpha", "params_beta", "value"]
].rename(columns={
    "number": "Trial",
    "params_alpha": "alpha",
    "params_beta": "beta",
    "value": "NDCG@10 val"
})
trials_df = trials_df.sort_values("NDCG@10 val", ascending=False).reset_index(drop=True)
trials_df.insert(0, "Rank", range(1, len(trials_df) + 1))
trials_df["NDCG@10 val"] = trials_df["NDCG@10 val"].round(5)
trials_df["alpha"] = trials_df["alpha"].round(2)
trials_df["beta"]  = trials_df["beta"].round(2)

print("TOP 10 TRIALS POR NDCG@10 EN VALIDACIÓN")
print("-" * 50)
print(trials_df.head(10).to_string(index=False))
print()

# Parámetros óptimos
best_alpha = study.best_params["alpha"]
best_beta  = study.best_params["beta"]
best_val   = study.best_value
print(f"Parámetros óptimos: alpha={best_alpha:.2f}  beta={best_beta:.2f}")
print(f"NDCG@10 (validación): {best_val:.5f}")
"""))

cells.append(py("""\
# ── Tabla de importancia de hiperparámetros (texto) ───────────────────────────
# Analizar cuánto varía NDCG@10 en función de cada parámetro

print("IMPORTANCIA DE HIPERPARÁMETROS (análisis de varianza)")
print("-" * 50)

for param in ["alpha", "beta"]:
    param_col = f"params_{param}" if f"params_{param}" in study.trials_dataframe().columns else param
    values  = trials_df[param].values
    ndcgs   = trials_df["NDCG@10 val"].values
    # Correlación de Pearson entre valor del parámetro y NDCG
    corr = np.corrcoef(values, ndcgs)[0, 1]

    # Agrupar por valor único y ver media de NDCG
    unique_vals = sorted(set(values))
    label_param = f"  {param} (correlacion con NDCG@10: {corr:+.3f}):"
    print()
    print(label_param)
    for v in unique_vals:
        mask = values == v
        mean_ndcg = ndcgs[mask].mean()
        n_trials  = mask.sum()
        bar = "█" * int(mean_ndcg * 300)
        print(f"    {param}={v:.2f}  (N={n_trials:2d}) → mean NDCG@10={mean_ndcg:.5f}  {bar}")
"""))

cells.append(py("""\
# ── B.4 — Evaluación FINAL sobre test set ────────────────────────────────────
# UNA SOLA evaluación con los mejores hiperparámetros
print(f"Construyendo RP3beta optimizado (alpha={best_alpha:.2f}, beta={best_beta:.2f})...")
t0 = time.time()
W_opt    = build_rp3(best_alpha, best_beta, X_top_csr, pop_sub)
get_rp3_opt = make_get_rp3(W_opt, X_top_csr, top_items_global)
print(f"  W_opt construido en {time.time()-t0:.1f}s")

# Evaluar con el test set definitivo
print(f"Evaluando RP3beta optimizado sobre test set ({len(test_users_b)} usuarios)...")
t0 = time.time()
m_rp3_opt = evaluate(
    get_rp3_opt, test_users_b,
    test_items_by_user, test_tx_by_user,
    item_pop_dict, n_total_train,
    n_items_global, baseline_conv
)
print(f"  [{time.time()-t0:.1f}s]")

# Evaluar también RP3beta original sobre el mismo test set para comparación directa
print(f"Evaluando RP3beta ORIGINAL sobre test set ({len(test_users_b)} usuarios)...")
t0 = time.time()
m_rp3_orig_test = evaluate(
    get_rp3_orig, test_users_b,
    test_items_by_user, test_tx_by_user,
    item_pop_dict, n_total_train,
    n_items_global, baseline_conv
)
print(f"  [{time.time()-t0:.1f}s]")
"""))

cells.append(py("""\
# ── Comparación RP3beta original vs optimizado ────────────────────────────────
print()
print("=" * 70)
print("B.4 — RP3beta original vs optimizado (test set)")
print("=" * 70)
metrics_b = [
    "NDCG@5", "NDCG@10", "Precision@5", "Precision@10",
    "Recall@5", "Recall@10", "Coverage", "Novelty"
]
orig_vals = [m_rp3_orig_test.get(m, 0.) for m in metrics_b]
opt_vals  = [m_rp3_opt.get(m, 0.) for m in metrics_b]

print(f"{'Métrica':<18} {'RP3beta original':>18} {'RP3beta optimizado':>20} {'Δ%':>8}")
print("-" * 70)
for met, ov, nv in zip(metrics_b, orig_vals, opt_vals):
    delta = (nv - ov) / ov * 100 if ov else 0.
    sign  = "+" if delta >= 0 else ""
    print(f"  {met:<16} {ov:>18.5f} {nv:>20.5f} {sign}{delta:>6.1f}%")
print()

ndcg_rp3_orig_test = m_rp3_orig_test.get("NDCG@10", 0.)
ndcg_rp3_opt_test  = m_rp3_opt.get("NDCG@10", 0.)
delta_opt = (ndcg_rp3_opt_test - ndcg_rp3_orig_test) / ndcg_rp3_orig_test * 100
print(f"  RP3beta original  : α={RP3_ALPHA_ORIG}   β={RP3_BETA_ORIG}    NDCG@10={ndcg_rp3_orig_test:.5f}")
print(f"  RP3beta optimizado: α={best_alpha:.2f}  β={best_beta:.2f}  NDCG@10={ndcg_rp3_opt_test:.5f}  ({'+' if delta_opt>=0 else ''}{delta_opt:.1f}%)")
"""))

cells.append(py("""\
# ── B.5 — Guardar modelo optimizado ──────────────────────────────────────────
artifact_rp3opt = {
    "model_name":    f"RP3beta optimizado (alpha={best_alpha:.2f}, beta={best_beta:.2f})",
    "alpha":         best_alpha,
    "beta":          best_beta,
    "ndcg10_test":   ndcg_rp3_opt_test,
    "ndcg10_val":    best_val,
    "W_rp3":         W_opt,
    "top_items_idx": top_items_idx,
    "top_items_global": top_items_global,
    "user2idx":      user2idx,
    "idx2item":      idx2item,
    "optuna_best_params": study.best_params,
    "optuna_n_trials":    N_OPTUNA_TRIALS,
}
out_path = ENC_DIR / "rp3beta_optimized.pkl"
with open(out_path, "wb") as f:
    pickle.dump(artifact_rp3opt, f, protocol=4)

import os
print(f"Guardado: {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")
"""))

# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN C - ENSEMBLE
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Sección C — Ensemble RP3beta optimizado + EASE^R

Combinación lineal convexa de scores normalizados de ambos modelos.
Usamos el mismo validation set de la Sección B para seleccionar el peso óptimo.
El test set se evalúa UNA SOLA VEZ con el peso óptimo encontrado.
"""))

cells.append(py("""\
# ── C.1 — Normalización de scores ────────────────────────────────────────────
def get_scores_vector(get_fn, uid, n_candidates=N_TOP):
    \"\"\"
    Obtiene el vector de scores completo (N_TOP) para un usuario.
    Retorna np.ndarray float32.
    \"\"\"
    if uid not in user2idx:
        return np.zeros(n_candidates, dtype=np.float32)
    ui  = user2idx[uid]
    row = X_top_csr.getrow(ui)
    return row

def minmax_norm(sc):
    \"\"\"Normaliza un vector de scores a [0,1] por usuario.\"\"\"
    mn, mx = sc.min(), sc.max()
    if mx - mn < 1e-10:
        return np.zeros_like(sc)
    return (sc - mn) / (mx - mn)

def get_rp3_scores(uid, W, X_csr):
    \"\"\"Score vector RP3beta (antes de excluir vistos).\"\"\"
    if uid not in user2idx:
        return np.zeros(N_TOP, dtype=np.float32)
    ui  = user2idx[uid]
    row = X_csr.getrow(ui)
    x_u = np.asarray(row.todense(), dtype=np.float32).ravel()
    return x_u @ W

def get_ease_scores(uid):
    \"\"\"Score vector EASE^R (antes de excluir vistos).\"\"\"
    if uid not in user2idx:
        return np.zeros(N_TOP, dtype=np.float32)
    ui  = user2idx[uid]
    row = X_top_csr.getrow(ui)
    x_u = np.asarray(row.todense(), dtype=np.float32).ravel()
    return x_u @ B_ease

def make_ensemble_get_fn(w_rp3, W_rp3, X_csr):
    \"\"\"Fabrica un get_fn de ensemble dado el peso w_rp3.\"\"\"
    w_ease = 1. - w_rp3
    def get_fn(uid, n):
        if uid not in user2idx: return []
        ui  = user2idx[uid]
        row = X_csr.getrow(ui)
        x_u = np.asarray(row.todense(), dtype=np.float32).ravel()

        sc_rp3  = minmax_norm(x_u @ W_rp3)
        sc_ease = minmax_norm(x_u @ B_ease)
        sc = w_rp3 * sc_rp3 + w_ease * sc_ease

        # Excluir ítems ya vistos
        sc[x_u > 0] = -1.
        top = np.argpartition(sc, -n)[-n:]
        top = top[np.argsort(sc[top])[::-1]]
        return [top_items_global[i] for i in top]
    return get_fn

print("Funciones de ensemble definidas.")
"""))

cells.append(py("""\
# ── C.2 — Sweep de pesos sobre VALIDATION SET ────────────────────────────────
weight_grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

print(f"Sweep de pesos sobre {len(val_users)} usuarios de validación...")
print("-" * 60)
print(f"  {'w_RP3':>6}  {'w_EASE':>6}  {'NDCG@10 val':>12}  {'Precision@10':>14}  {'Coverage':>10}")
print("-" * 60)

sweep_results = []
for w in weight_grid:
    get_ens = make_ensemble_get_fn(w, W_opt, X_top_csr)
    m = evaluate(
        get_ens, val_users,
        test_items_by_user, test_tx_by_user,
        item_pop_dict, n_total_train,
        n_items_global, baseline_conv
    )
    sweep_results.append({
        "w_RP3":       w,
        "w_EASE":      round(1. - w, 2),
        "NDCG@10 val": round(m.get("NDCG@10", 0.), 5),
        "Prec@10 val": round(m.get("Precision@10", 0.), 6),
        "Coverage":    round(m.get("Coverage", 0.), 4),
    })
    print(f"  w={w:.2f}  w_ease={1-w:.2f}  "
          f"NDCG@10={m.get('NDCG@10',0.):.5f}  "
          f"Precision@10={m.get('Precision@10',0.):.5f}  "
          f"Coverage={m.get('Coverage',0.):.4f}")

df_sweep = pd.DataFrame(sweep_results)
best_w_row = df_sweep.loc[df_sweep["NDCG@10 val"].idxmax()]
best_w_rp3 = float(best_w_row["w_RP3"])
print()
print(f"Peso óptimo: w_rp3={best_w_rp3:.2f}  w_ease={1-best_w_rp3:.2f}  "
      f"NDCG@10 val={best_w_row['NDCG@10 val']:.5f}")
"""))

cells.append(py("""\
# ── C.3 — Evaluación FINAL del ensemble sobre TEST SET ───────────────────────
# UNA SOLA evaluación con el peso óptimo
print(f"Evaluando ensemble (w_rp3={best_w_rp3:.2f}) sobre test set ({len(test_users_b)} usuarios)...")
t0 = time.time()
get_ens_best = make_ensemble_get_fn(best_w_rp3, W_opt, X_top_csr)
m_ens = evaluate(
    get_ens_best, test_users_b,
    test_items_by_user, test_tx_by_user,
    item_pop_dict, n_total_train,
    n_items_global, baseline_conv
)
print(f"  [{time.time()-t0:.1f}s]")

# Evaluar EASE^R sobre el mismo test set para comparación
print(f"Evaluando EASE^R sobre test set ({len(test_users_b)} usuarios)...")
t0 = time.time()
m_ease_test = evaluate(
    get_ease, test_users_b,
    test_items_by_user, test_tx_by_user,
    item_pop_dict, n_total_train,
    n_items_global, baseline_conv
)
print(f"  [{time.time()-t0:.1f}s]")
"""))

cells.append(py("""\
# ── Tabla comparativa C.3 ────────────────────────────────────────────────────
print()
print("=" * 80)
print("C.3 — Comparativa de modelos (test set definitivo)")
print("=" * 80)
metrics_c = [
    "NDCG@5", "NDCG@10", "Precision@5", "Precision@10",
    "Recall@5", "Recall@10", "Coverage", "Novelty"
]

modelos_c = {
    "RP3beta original": m_rp3_orig_test,
    f"RP3beta opt (α={best_alpha:.2f}, β={best_beta:.2f})": m_rp3_opt,
    "EASE^R": m_ease_test,
    f"Ensemble RP3opt+EASE (w={best_w_rp3:.2f})": m_ens,
}

# Cabecera
header = f"  {'Métrica':<18}"
for name in modelos_c:
    header += f"  {name[:20]:>20}"
print(header)
print("-" * (18 + 22 * len(modelos_c) + 2))

for met in metrics_c:
    row_str = f"  {met:<18}"
    vals = [m.get(met, 0.) for m in modelos_c.values()]
    best_val_c = max(vals)
    for v in vals:
        flag = " ★" if abs(v - best_val_c) < 1e-8 and v > 0 else "  "
        row_str += f"  {v:>18.5f}{flag[:1]}"
    print(row_str)
print()
"""))

cells.append(py("""\
# ── C.4 — Correlación Spearman entre rankings RP3beta y EASE^R ───────────────
from scipy.stats import spearmanr

print("=" * 60)
print("C.4 — Correlación de rankings RP3beta opt vs EASE^R")
print("=" * 60)

# Calcular scores sobre una muestra de usuarios del test set
sample_users_corr = test_users_b[:min(200, len(test_users_b))]
rp3_scores_sample  = []
ease_scores_sample = []

for uid in sample_users_corr:
    if uid not in user2idx: continue
    ui = user2idx[uid]
    row_sp = X_top_csr.getrow(ui)
    x_u = np.asarray(row_sp.todense(), dtype=np.float32).ravel()

    sc_rp3  = (x_u @ W_opt).ravel()
    sc_ease = (x_u @ B_ease).ravel()

    # Solo ítems no vistos
    mask_unseen = x_u == 0
    if mask_unseen.sum() > 50:
        rp3_scores_sample.append(sc_rp3[mask_unseen])
        ease_scores_sample.append(sc_ease[mask_unseen])

if rp3_scores_sample:
    # Correlación sobre todos los scores concatenados
    all_rp3  = np.concatenate(rp3_scores_sample)
    all_ease = np.concatenate(ease_scores_sample)
    # Muestrear para que scipy sea rápido (máx 100K pares)
    if len(all_rp3) > 100_000:
        idx_s = np.random.choice(len(all_rp3), 100_000, replace=False)
        all_rp3  = all_rp3[idx_s]
        all_ease = all_ease[idx_s]
    corr, pval = spearmanr(all_rp3, all_ease)
    print(f"  N pares evaluados     : {len(all_rp3):,}")
    print(f"  Correlación Spearman  : ρ = {corr:.4f}  (p={pval:.2e})")
    print()
    if corr < 0.70:
        interpretation = "ALTA complementariedad: el ensemble está bien justificado"
    elif corr < 0.85:
        interpretation = "MODERADA complementariedad: el ensemble aporta algo"
    else:
        interpretation = "BAJA complementariedad: los modelos son muy similares"
    print(f"  Interpretación: {interpretation}")
    print()
    print(f"  ρ < 0.70 → complementariedad alta → ensemble justificado")
    print(f"  ρ > 0.85 → complementariedad baja → ensemble suma poco")
"""))

# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN D - TABLA COMPARATIVA NB11
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## Sección D — Tabla Comparativa NB11 Consolidada

Todos los modelos del proyecto ordenados por NDCG@10.
Los modelos NB07–NB09 conservan sus métricas originales sobre el protocolo ≥1.
Los modelos nuevos (B y C) se evalúan sobre el subconjunto test_users_b (85% de eval_users).
"""))

cells.append(py("""\
# ── Tabla consolidada NB11 ────────────────────────────────────────────────────
rows_nb11 = [
    # ── Históricos NB07–NB10 (métricas originales NO modificadas) ────────────
    {
        "Modelo":        "Mult-VAE^PR (enc=[600,200], z=64)",
        "NDCG@10":       0.025453,
        "Precision@10":  0.006467,
        "Recall@10":     0.0,
        "Coverage":      0.0516,
        "Novelty":       0.0,
        "Protocolo":     ">=1",
        "Notebook":      "NB10",
    },
    {
        "Modelo":        "RP3beta (α=0.85, β=0.25) — original",
        "NDCG@10":       0.025763,
        "Precision@10":  0.006067,
        "Recall@10":     0.0,
        "Coverage":      0.0600,
        "Novelty":       0.0,
        "Protocolo":     ">=1",
        "Notebook":      "NB09",
    },
    {
        "Modelo":        "EASE^R (λ=500, top-20K)",
        "NDCG@10":       0.019310,
        "Precision@10":  0.004770,
        "Recall@10":     0.0,
        "Coverage":      0.0496,
        "Novelty":       0.0,
        "Protocolo":     ">=1",
        "Notebook":      "NB09",
    },
    {
        "Modelo":        "SVD+TD+IPS (λ=0.03, γ=0.4)",
        "NDCG@10":       0.009340,
        "Precision@10":  0.002530,
        "Recall@10":     0.0,
        "Coverage":      0.0115,
        "Novelty":       0.0,
        "Protocolo":     ">=1",
        "Notebook":      "NB08",
    },
    {
        "Modelo":        "SVD (k=50)",
        "NDCG@10":       0.008085,
        "Precision@10":  0.002200,
        "Recall@10":     0.011169,
        "Coverage":      0.0041,
        "Novelty":       14.29,
        "Protocolo":     ">=1",
        "Notebook":      "NB07",
    },
    {
        "Modelo":        "BPR-MF (k=64, SGD)",
        "NDCG@10":       0.001240,
        "Precision@10":  0.000470,
        "Recall@10":     0.0,
        "Coverage":      0.0017,
        "Novelty":       0.0,
        "Protocolo":     ">=1",
        "Notebook":      "NB09",
    },
    # ── Nuevos NB11 ──────────────────────────────────────────────────────────
    {
        "Modelo":        f"RP3beta opt (α={best_alpha:.2f}, β={best_beta:.2f})",
        "NDCG@10":       round(m_rp3_opt.get("NDCG@10", 0.), 6),
        "Precision@10":  round(m_rp3_opt.get("Precision@10", 0.), 6),
        "Recall@10":     round(m_rp3_opt.get("Recall@10", 0.), 6),
        "Coverage":      round(m_rp3_opt.get("Coverage", 0.), 4),
        "Novelty":       round(m_rp3_opt.get("Novelty", 0.), 2),
        "Protocolo":     ">=1 (test=85%)",
        "Notebook":      "NB11-B",
    },
    {
        "Modelo":        f"Ensemble RP3opt+EASE (w={best_w_rp3:.2f})",
        "NDCG@10":       round(m_ens.get("NDCG@10", 0.), 6),
        "Precision@10":  round(m_ens.get("Precision@10", 0.), 6),
        "Recall@10":     round(m_ens.get("Recall@10", 0.), 6),
        "Coverage":      round(m_ens.get("Coverage", 0.), 4),
        "Novelty":       round(m_ens.get("Novelty", 0.), 2),
        "Protocolo":     ">=1 (test=85%)",
        "Notebook":      "NB11-C",
    },
]

df_nb11 = (
    pd.DataFrame(rows_nb11)
    .sort_values("NDCG@10", ascending=False)
    .reset_index(drop=True)
)
df_nb11.insert(0, "Rank", range(1, len(df_nb11) + 1))

print("=" * 100)
print("TABLA D — TODOS LOS MODELOS DEL PROYECTO (NB07–NB11)")
print("=" * 100)
print(df_nb11[[
    "Rank", "Modelo", "NDCG@10", "Precision@10", "Recall@10",
    "Coverage", "Novelty", "Protocolo", "Notebook"
]].to_string(index=False))
print()

winner_nb11 = df_nb11.iloc[0]
print(f"Modelo ganador global  : {winner_nb11['Modelo']}")
print(f"NDCG@10               : {winner_nb11['NDCG@10']:.6f}")
rp3_orig_ndcg = 0.025763
print(f"Δ vs RP3beta original : {(winner_nb11['NDCG@10'] - rp3_orig_ndcg) / rp3_orig_ndcg * 100:+.1f}%")
"""))

cells.append(py("""\
# ── Guardar CSV NB11 ────────────────────────────────────────────────────────
out_csv = DATA_DIR / "model_comparison_nb11.csv"
df_nb11.to_csv(out_csv, index=False)
print(f"Guardado: {out_csv}")

# Actualizar también model_comparison_final.csv con los nuevos modelos NB11
# (solo agregar filas, no modificar las existentes)
try:
    df_final_old = pd.read_csv(DATA_DIR / "model_comparison_final.csv")
    # Preparar filas NB11 con el mismo formato que model_comparison_final.csv
    nb11_new_rows = []
    for _, r in df_nb11[df_nb11["Notebook"].str.startswith("NB11")].iterrows():
        nb11_new_rows.append({
            "Model":        r["Modelo"],
            "Precision@5":  r.get("Precision@5", 0.),
            "Recall@5":     r.get("Recall@5", 0.),
            "NDCG@5":       r.get("NDCG@5", 0.),
            "Precision@10": r["Precision@10"],
            "Recall@10":    r["Recall@10"],
            "NDCG@10":      r["NDCG@10"],
            "MAP@10":       0.,
            "Coverage":     r["Coverage"],
            "Novelty":      r["Novelty"],
            "train_time_s": 0.,
        })
    if nb11_new_rows:
        df_final_new = pd.concat(
            [df_final_old, pd.DataFrame(nb11_new_rows)],
            ignore_index=True
        )
        df_final_new.to_csv(DATA_DIR / "model_comparison_final.csv", index=False)
        print(f"Actualizado: data/processed/model_comparison_final.csv "
              f"(+{len(nb11_new_rows)} filas)")
except Exception as e:
    print(f"Nota: no se pudo actualizar model_comparison_final.csv: {e}")
"""))

# ════════════════════════════════════════════════════════════════════════════════
# SECCIÓN E - DOCUMENTACIÓN
# ════════════════════════════════════════════════════════════════════════════════
cells.append(md("## Sección E — Resumen Ejecutivo NB11"))

cells.append(py("""\
# ── Resumen final ────────────────────────────────────────────────────────────
print("=" * 70)
print("RESUMEN EJECUTIVO NB11 — Optimización y Ensemble")
print("=" * 70)
print()
ndcg_orig  = m_rp3_orig_test.get("NDCG@10", 0.)
ndcg_opt   = m_rp3_opt.get("NDCG@10", 0.)
ndcg_ens   = m_ens.get("NDCG@10", 0.)
ndcg_ease  = m_ease_test.get("NDCG@10", 0.)

print(f"SECCIÓN A — Protocolo de evaluación:")
for _, row in df_proto.iterrows():
    delta = (row["NDCG@10"] - df_proto.iloc[0]["NDCG@10"]) / df_proto.iloc[0]["NDCG@10"] * 100
    sign = "+" if delta >= 0 else ""
    print(f"  {row['Protocolo']:14s}: N={row['N usuarios']:4d}  NDCG@10={row['NDCG@10']:.5f}  ({sign}{delta:.1f}%)")
print()
print(f"SECCIÓN B — Optuna ({N_OPTUNA_TRIALS} trials):")
print(f"  Original   : α={RP3_ALPHA_ORIG:.2f}  β={RP3_BETA_ORIG:.2f}  NDCG@10={ndcg_orig:.5f}")
print(f"  Optimizado : α={best_alpha:.2f}  β={best_beta:.2f}  NDCG@10={ndcg_opt:.5f}  "
      f"({'+' if ndcg_opt>=ndcg_orig else ''}{(ndcg_opt-ndcg_orig)/ndcg_orig*100:.1f}%)")
print()
print(f"SECCIÓN C — Ensemble:")
print(f"  RP3beta opt  : NDCG@10={ndcg_opt:.5f}")
print(f"  EASE^R       : NDCG@10={ndcg_ease:.5f}")
print(f"  Ensemble opt : w_rp3={best_w_rp3:.2f}  NDCG@10={ndcg_ens:.5f}  "
      f"({'+' if ndcg_ens>=ndcg_opt else ''}{(ndcg_ens-ndcg_opt)/ndcg_opt*100:.1f}% vs RP3opt)")
print()
print(f"MODELO GANADOR NB11: {winner_nb11['Modelo']}")
print(f"NDCG@10 final      : {winner_nb11['NDCG@10']:.6f}")
print()
print("Artefactos generados:")
print("  encoders/rp3beta_optimized.pkl")
print("  data/processed/model_comparison_nb11.csv")
print("  data/processed/model_comparison_final.csv  (actualizado)")
"""))

# ════════════════════════════════════════════════════════════════════════════════
# ESCRIBIR NOTEBOOK
# ════════════════════════════════════════════════════════════════════════════════
import uuid

for cell in cells:
    if cell["cell_type"] == "code":
        cell["id"] = uuid.uuid4().hex[:8]
    else:
        cell["id"] = uuid.uuid4().hex[:8]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.13.5",
        },
    },
    "cells": cells,
}

NB_OUT.parent.mkdir(exist_ok=True)
with open(NB_OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook generado: {NB_OUT}")
print(f"  Celdas: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} código, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
print(f"  Tamaño: {NB_OUT.stat().st_size/1024:.1f} KB")
