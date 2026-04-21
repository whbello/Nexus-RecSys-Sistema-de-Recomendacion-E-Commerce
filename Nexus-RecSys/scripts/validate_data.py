"""
validate_data.py — Verificación de datos de entrada de nexus-recsys
=====================================================================
Verifica que los archivos raw necesarios existan y tengan el shape esperado.
Aborta con mensaje claro si falta algo o si hay inconsistencias.

Uso: python scripts/validate_data.py
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW  = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"

# ─────────────────────────────────────────────────────────────────────────────
# Configuración esperada de archivos y tamaños mínimos
# ─────────────────────────────────────────────────────────────────────────────
ARCHIVOS_PROCESADOS_REQUERIDOS = {
    "interaction_matrix.csv": {
        "descripcion": "Matriz de interacciones usuario-ítem",
        "min_filas": 1_000_000,
        "columnas_requeridas": ["visitorid", "itemid", "interaction_strength",
                                "last_interaction_ts", "last_interaction_type"],
    },
    "item_features.csv": {
        "descripcion": "Features de ítems procesados",
        "min_filas": 100_000,
        "columnas_requeridas": ["itemid"],
    },
    "model_comparison_final.csv": {
        "descripcion": "Tabla comparativa de modelos (resultados finales)",
        "min_filas": 3,
        "columnas_requeridas": ["notebook", "model", "ndcg10_test"],
    },
    "train_test_split_info.json": {
        "descripcion": "Metadatos del split temporal",
        "min_filas": None,
        "columnas_requeridas": None,
    },
}

CARPETA_SCORE_CACHE = ROOT / "scripts" / "_score_cache"
SCORE_CACHE_REQUERIDOS = [
    "test_rp3_mb_td.npy",
    "test_ease_500.npy",
    "test_rp3_td.npy",
    "val_rp3_mb_td.npy",
    "val_ease_500.npy",
    "val_rp3_td.npy",
]

ERRORES   = []
ADVERTENCIAS = []


def verificar_csv(ruta: Path, config: dict):
    """Verifica existencia, tamaño mínimo y columnas de un CSV."""
    try:
        import pandas as pd
    except ImportError:
        ERRORES.append("pandas no está instalado. Ejecutá: pip install pandas")
        return

    if not ruta.exists():
        ERRORES.append(f"FALTA: {ruta.relative_to(ROOT)}")
        return

    try:
        # Leer solo el encabezado y el número de filas de forma eficiente
        df_head = pd.read_csv(ruta, nrows=5)
        columnas = set(df_head.columns)

        # Verificar columnas requeridas
        if config.get("columnas_requeridas"):
            faltantes = set(config["columnas_requeridas"]) - columnas
            if faltantes:
                ERRORES.append(
                    f"Columnas faltantes en {ruta.name}: {faltantes}\n"
                    f"  Encontradas: {sorted(columnas)}"
                )

        # Verificar tamaño mínimo
        if config.get("min_filas"):
            # Contar filas sin cargar todo el CSV
            with open(ruta, "r", encoding="utf-8") as f:
                n_filas = sum(1 for _ in f) - 1  # descontar encabezado
            if n_filas < config["min_filas"]:
                ERRORES.append(
                    f"Muy pocas filas en {ruta.name}: "
                    f"{n_filas:,} < mínimo esperado {config['min_filas']:,}"
                )
            else:
                print(f"  ✓ {ruta.name}: {n_filas:,} filas")
        else:
            print(f"  ✓ {ruta.name}: existe")

    except Exception as e:
        ERRORES.append(f"Error leyendo {ruta.name}: {e}")


def verificar_json(ruta: Path):
    """Verifica existencia y parseo de un JSON."""
    import json
    if not ruta.exists():
        ERRORES.append(f"FALTA: {ruta.relative_to(ROOT)}")
        return
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)
        campos_esperados = ["cutoff_date", "train_interactions", "test_interactions"]
        faltantes = [c for c in campos_esperados if c not in data]
        if faltantes:
            ADVERTENCIAS.append(f"Campos faltantes en {ruta.name}: {faltantes}")
        else:
            print(f"  ✓ {ruta.name}: cutoff={data.get('cutoff_date')}")
    except Exception as e:
        ERRORES.append(f"Error parseando {ruta.name}: {e}")


def verificar_npy(ruta: Path):
    """Verifica existencia y forma de un array numpy."""
    try:
        import numpy as np
    except ImportError:
        ERRORES.append("numpy no está instalado.")
        return

    if not ruta.exists():
        ERRORES.append(f"FALTA cache: {ruta.relative_to(ROOT)}")
        return
    try:
        arr = np.load(ruta, mmap_mode="r")
        if arr.ndim != 2:
            ERRORES.append(f"Shape incorrecto en {ruta.name}: esperado 2D, encontrado {arr.ndim}D")
        else:
            print(f"  ✓ {ruta.name}: shape={arr.shape}")
    except Exception as e:
        ERRORES.append(f"Error cargando {ruta.name}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  nexus-recsys — Validación de datos de entrada")
    print("=" * 60)

    # 1. Verificar datos procesados
    print("\n[1/3] Verificando data/processed/...")
    for nombre, config in ARCHIVOS_PROCESADOS_REQUERIDOS.items():
        ruta = DATA_PROC / nombre
        if nombre.endswith(".json"):
            verificar_json(ruta)
        else:
            verificar_csv(ruta, config)

    # 2. Verificar score cache
    print(f"\n[2/3] Verificando {CARPETA_SCORE_CACHE.relative_to(ROOT)}/...")
    if not CARPETA_SCORE_CACHE.exists():
        ERRORES.append(
            f"Carpeta _score_cache NO encontrada en {CARPETA_SCORE_CACHE}\n"
            "  Ejecutá el pipeline de entrenamiento (NB13–NB15) para generarla."
        )
    else:
        for nombre in SCORE_CACHE_REQUERIDOS:
            verificar_npy(CARPETA_SCORE_CACHE / nombre)

    # 3. Verificar encoders
    print("\n[3/3] Verificando encoders/...")
    encoders_dir = ROOT / "encoders"
    encoders_requeridos = ["rp3beta_mb_td_meta.json", "rp3beta_td_ips_meta.json"]
    for enc in encoders_requeridos:
        ruta = encoders_dir / enc
        if ruta.exists():
            print(f"  ✓ {enc}")
        else:
            ADVERTENCIAS.append(f"Encoder opcional no encontrado: {enc}")

    # ─── Resumen ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if ADVERTENCIAS:
        print(f"  ⚠  {len(ADVERTENCIAS)} advertencia(s):")
        for w in ADVERTENCIAS:
            print(f"     → {w}")

    if ERRORES:
        print(f"\n  ✗  {len(ERRORES)} ERROR(ES) CRÍTICO(S):")
        for e in ERRORES:
            print(f"     → {e}")
        print("\n  Pipeline abortado. Corregir errores antes de continuar.")
        print("=" * 60)
        sys.exit(1)
    else:
        print("  ✅ Todos los datos de entrada son válidos.")
        print("  Continuar con el pipeline.")
    print("=" * 60)


if __name__ == "__main__":
    main()
