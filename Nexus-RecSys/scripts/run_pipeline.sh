#!/usr/bin/env bash
# ============================================================
# nexus-recsys — Pipeline completo end-to-end
# ============================================================
# Ejecutar desde la raíz del proyecto:
#   bash scripts/run_pipeline.sh
#
# El script detiene la ejecución ante cualquier error (set -e).
# Logs detallados de cada paso guardados en: pipeline_run.log
# ============================================================
set -e

LOG_FILE="pipeline_run.log"
INICIO=$(date +%s)

log() {
    echo "$1" | tee -a "$LOG_FILE"
}

log "======================================================"
log "  NEXUS-RECSYS — Pipeline Completo End-to-End"
log "  Sistema de Recomendación de Productos"
log "  Fecha: $(date '+%Y-%m-%d %H:%M:%S')"
log "======================================================"

# ──────────────────────────────────────────────────────────
# [1/7] Verificar e instalar dependencias
# ──────────────────────────────────────────────────────────
log ""
log "[1/7] Verificando e instalando dependencias..."
pip install -r requirements.txt --quiet 2>&1 | tee -a "$LOG_FILE"
log "  ✓ Dependencias verificadas"

# ──────────────────────────────────────────────────────────
# [2/7] Validar datos de entrada
# ──────────────────────────────────────────────────────────
log ""
log "[2/7] Validando datos de entrada..."
python scripts/validate_data.py 2>&1 | tee -a "$LOG_FILE"
log "  ✓ Datos de entrada validados"

# ──────────────────────────────────────────────────────────
# [3/7] EDA y Pipeline ETL (NB01-NB04)
# ──────────────────────────────────────────────────────────
log ""
log "[3/7] Ejecutando EDA y pipeline ETL (NB01-NB04)..."

for NB in 01_eda_events 02_eda_items_categories 03_funnel_analysis 04_merge_pipeline; do
    log "  → Ejecutando notebooks/${NB}.ipynb..."
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=1800 \
        --ExecutePreprocessor.kernel_name=python3 \
        "notebooks/${NB}.ipynb" --output "notebooks/${NB}.ipynb" \
        2>&1 | tee -a "$LOG_FILE"
    log "  ✓ ${NB}.ipynb completado"
done

# ──────────────────────────────────────────────────────────
# [4/7] Feature Engineering (NB05-NB06)
# ──────────────────────────────────────────────────────────
log ""
log "[4/7] Ejecutando feature engineering (NB05-NB06)..."

for NB in 05_synthetic_demographics 06_feature_engineering; do
    log "  → Ejecutando notebooks/${NB}.ipynb..."
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=1800 \
        --ExecutePreprocessor.kernel_name=python3 \
        "notebooks/${NB}.ipynb" --output "notebooks/${NB}.ipynb" \
        2>&1 | tee -a "$LOG_FILE"
    log "  ✓ ${NB}.ipynb completado"
done

# ──────────────────────────────────────────────────────────
# [5/7] Modelado base (NB07-NB09)
# ──────────────────────────────────────────────────────────
log ""
log "[5/7] Ejecutando modelado (NB07-NB09)..."
log "  NOTA: Esta etapa puede tardar 15-30 minutos dependiendo del hardware."

# Primero generar el notebook de modelado base
python scripts/generate_modeling_notebook.py 2>&1 | tee -a "$LOG_FILE"

for NB in 07_modeling 08_business_metrics_roi 09_advanced_models 10_multivae; do
    log "  → Ejecutando notebooks/${NB}.ipynb..."
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=3600 \
        --ExecutePreprocessor.kernel_name=python3 \
        "notebooks/${NB}.ipynb" --output "notebooks/${NB}.ipynb" \
        2>&1 | tee -a "$LOG_FILE"
    log "  ✓ ${NB}.ipynb completado"
done

# ──────────────────────────────────────────────────────────
# [6/7] Ensemble final (NB11 + NB15v2 champion)
# ──────────────────────────────────────────────────────────
log ""
log "[6/7] Ejecutando optimización y ensemble final..."
log "  NOTA: NB11 ejecuta 50 trials Optuna. NB13-NB15 construyen el ensemble."

for NB in 11_optimization_ensemble 13_ensemble_advanced 14_strategies_advanced; do
    log "  → Ejecutando notebooks/${NB}.ipynb..."
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=7200 \
        --ExecutePreprocessor.kernel_name=python3 \
        "notebooks/${NB}.ipynb" --output "notebooks/${NB}.ipynb" \
        2>&1 | tee -a "$LOG_FILE"
    log "  ✓ ${NB}.ipynb completado"
done

# Script del ensemble champion NB15v2
log "  → Ejecutando ensemble champion (NB15v2)..."
python scripts/_nb15v2_ensemble.py 2>&1 | tee -a "$LOG_FILE"
log "  ✓ Ensemble champion completado"

# ──────────────────────────────────────────────────────────
# [7/7] Verificación final de artefactos
# ──────────────────────────────────────────────────────────
log ""
log "[7/7] Verificando artefactos generados..."
python scripts/validate_artifacts.py 2>&1 | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────────────────
# Resumen final
# ──────────────────────────────────────────────────────────
FIN=$(date +%s)
DURACION=$((FIN - INICIO))
MINUTOS=$((DURACION / 60))
SEGUNDOS=$((DURACION % 60))

log ""
log "======================================================"
log "  ✅ Pipeline completado exitosamente"
log "  Duración total: ${MINUTOS}m ${SEGUNDOS}s"
log "  Modelo ganador: Mega-Ensemble (rp3_mb_td + ease_500 + rp3_td)"
log "  NDCG@10 = 0.04310 (+50.8% vs baseline RP3+TD)"
log ""
log "  Próximos pasos:"
log "    → API:       uvicorn api.main:app --port 8000"
log "    → Dashboard: streamlit run dashboard/app.py"
log "    → Métricas:  cat data/processed/model_comparison_final.csv"
log "    → Logs:      cat ${LOG_FILE}"
log "======================================================"
