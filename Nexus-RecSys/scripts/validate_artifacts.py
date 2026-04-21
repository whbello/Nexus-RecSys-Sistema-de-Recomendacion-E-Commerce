"""
validate_artifacts.py — Verificación de artefactos del modelo ganador
======================================================================
Verifica que todos los artefactos del Mega-Ensemble (NB15v2) estén
presentes y sean cargables. Aborta con mensaje claro si algo falta.

Modelo ganador: Mega-Ensemble (rp3_mb_td + ease_500 + rp3_td)
NDCG@10 = 0.04310 (+50.8% vs baseline RP3+TD = 0.02859)

Uso: python scripts/validate_artifacts.py
"""
import sys
import json
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
ENCODERS   = ROOT / "encoders"
DATA_PROC  = ROOT / "data" / "processed"
SCORE_CACHE = ROOT / "scripts" / "_score_cache"

ERRORES      = []
ADVERTENCIAS = []


def verificar_json_encoder(ruta: Path, campos_esperados: list):
    """Verifica que un encoder JSON exista y contenga los campos esperados."""
    if not ruta.exists():
        ERRORES.append(f"FALTA encoder: {ruta.relative_to(ROOT)}")
        return
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)
        faltantes = [c for c in campos_esperados if c not in data]
        if faltantes:
            ERRORES.append(f"Campos incompletos en {ruta.name}: {faltantes}")
        else:
            print(f"  ✓ {ruta.relative_to(ROOT)}")
            for c in campos_esperados[:3]:
                print(f"     {c} = {data[c]}")
    except Exception as e:
        ERRORES.append(f"Error leyendo {ruta.name}: {e}")


def verificar_pkl(ruta: Path):
    """Verifica que un archivo pickle exista y sea deserializable."""
    if not ruta.exists():
        ADVERTENCIAS.append(f"Artefacto opcional no encontrado: {ruta.relative_to(ROOT)}")
        return
    try:
        import joblib
        obj = joblib.load(ruta)
        print(f"  ✓ {ruta.relative_to(ROOT)} — tipo: {type(obj).__name__}")
    except ImportError:
        # Fallback con pickle estándar
        import pickle
        with open(ruta, "rb") as f:
            obj = pickle.load(f)
        print(f"  ✓ {ruta.relative_to(ROOT)} — tipo: {type(obj).__name__}")
    except Exception as e:
        ERRORES.append(f"Error cargando {ruta.name}: {e}")


def verificar_npy(ruta: Path, shape_min: tuple = None):
    """Verifica que un archivo .npy exista y tenga el shape mínimo esperado."""
    try:
        import numpy as np
    except ImportError:
        ERRORES.append("numpy no instalado")
        return

    if not ruta.exists():
        ERRORES.append(f"FALTA score cache: {ruta.relative_to(ROOT)}")
        return
    try:
        arr = np.load(ruta, mmap_mode="r")
        shape_ok = True
        if shape_min:
            shape_ok = all(arr.shape[i] >= shape_min[i]
                           for i in range(min(len(shape_min), arr.ndim)))
        estado = "✓" if shape_ok else "⚠"
        print(f"  {estado} {ruta.relative_to(ROOT)} — shape={arr.shape}")
        if not shape_ok:
            ADVERTENCIAS.append(
                f"Shape inesperado en {ruta.name}: {arr.shape}, mínimo esperado {shape_min}"
            )
    except Exception as e:
        ERRORES.append(f"Error cargando {ruta.name}: {e}")


def verificar_csv_existe(ruta: Path):
    """Verifica mínimamente que un CSV exista y sea parseable."""
    if not ruta.exists():
        ERRORES.append(f"FALTA: {ruta.relative_to(ROOT)}")
        return
    try:
        import pandas as pd
        df = pd.read_csv(ruta, nrows=5)
        print(f"  ✓ {ruta.relative_to(ROOT)} — columnas: {list(df.columns)}")
    except Exception as e:
        ERRORES.append(f"Error leyendo {ruta.name}: {e}")


def verificar_champion_ndcg():
    """Verifica que el NDCG@10 reportado sea el del modelo ganador."""
    ruta = DATA_PROC / "model_comparison_final.csv"
    if not ruta.exists():
        ADVERTENCIAS.append("model_comparison_final.csv no encontrado para verificar NDCG")
        return
    try:
        import pandas as pd
        df = pd.read_csv(ruta)
        ndcg_max = df["ndcg10_test"].max()
        NDCG_ESPERADO = 0.04310
        if abs(ndcg_max - NDCG_ESPERADO) < 1e-4:
            print(f"  ✓ NDCG@10 champion = {ndcg_max:.5f} (esperado ≈ {NDCG_ESPERADO})")
        else:
            ADVERTENCIAS.append(
                f"NDCG@10 champion = {ndcg_max:.5f}, diferente del esperado {NDCG_ESPERADO}. "
                "Verificar si hubo actualización del modelo."
            )
    except Exception as e:
        ADVERTENCIAS.append(f"No se pudo verificar NDCG champion: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  nexus-recsys — Validación de artefactos del modelo ganador")
    print("  Mega-Ensemble NB15v2: NDCG@10 = 0.04310")
    print("=" * 65)

    # 1. Verificar encoders (metadatos de modelos)
    print("\n[1/5] Verificando encoders/ (metadatos de modelos)...")
    verificar_json_encoder(
        ENCODERS / "rp3beta_mb_td_meta.json",
        campos_esperados=["model_name", "alpha", "beta", "decay_rate",
                          "w_view", "w_cart", "w_trans"]
    )
    verificar_json_encoder(
        ENCODERS / "rp3beta_td_ips_meta.json",
        campos_esperados=["model_name", "alpha", "beta"]
    )

    # 2. Verificar modelos opcionales serializados
    print("\n[2/5] Verificando modelos opcionales en encoders/...")
    for nombre in ["final_model_v4.pkl", "hybrid_model.pkl",
                   "label_encoders.pkl", "scaler_item.pkl", "scaler_user.pkl"]:
        verificar_pkl(ENCODERS / nombre)

    # 3. Verificar score cache del ensemble ganador
    print("\n[3/5] Verificando score cache del ensemble (scripts/_score_cache/)...")
    modelos_champion = ["rp3_mb_td", "ease_500", "rp3_td"]
    for nm in modelos_champion:
        verificar_npy(SCORE_CACHE / f"test_{nm}.npy", shape_min=(100, 100))
        verificar_npy(SCORE_CACHE / f"val_{nm}.npy",  shape_min=(10, 100))

    # 4. Verificar datos procesados necesarios
    print("\n[4/5] Verificando data/processed/...")
    verificar_csv_existe(DATA_PROC / "model_comparison_final.csv")
    verificar_csv_existe(DATA_PROC / "item_features.csv")
    if (DATA_PROC / "interaction_matrix.csv").exists():
        print(f"  ✓ interaction_matrix.csv (existe — no se carga por tamaño)")
    else:
        ERRORES.append("FALTA: data/processed/interaction_matrix.csv")

    # 5. Verificar NDCG del champion
    print("\n[5/5] Verificando métricas del modelo ganador...")
    verificar_champion_ndcg()

    # ─── Resumen ─────────────────────────────────────────────────────────────
    nb15_results = ROOT / "scripts" / "_nb15v2_results.json"
    if nb15_results.exists():
        with open(nb15_results, "r") as f:
            r = json.load(f)
        w = r.get("NB15v2_result", {}).get("weights", {})
        print(f"\n  Pesos del ensemble guardados:")
        for nm, wv in w.items():
            print(f"    {nm}: {wv:.4f}")

    print("\n" + "=" * 65)
    if ADVERTENCIAS:
        print(f"  ⚠  {len(ADVERTENCIAS)} advertencia(s):")
        for warn in ADVERTENCIAS:
            print(f"     → {warn}")

    if ERRORES:
        print(f"\n  ✗  {len(ERRORES)} ERROR(ES) CRÍTICO(S):")
        for err in ERRORES:
            print(f"     → {err}")
        print("\n  Artefactos incompletos. Ejecutar notebooks faltantes.")
        print("=" * 65)
        sys.exit(1)
    else:
        print("  ✅ Todos los artefactos del modelo ganador están presentes.")
        print("  El sistema está listo para inferencia.")
    print("=" * 65)


if __name__ == "__main__":
    main()
