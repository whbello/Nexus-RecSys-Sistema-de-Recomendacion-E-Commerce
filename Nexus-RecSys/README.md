# Nexus RecSys

**Sistema de Recomendación de E-Commerce sobre el dataset público RetailRocket**

> **Pipeline completo: NB01–NB15 · API REST (FastAPI) · Dashboard interactivo (Streamlit)**  
> Champion: Mega-Ensemble NB15v2 · **NDCG@10 = 0.04310** (+50.8% vs baseline RP3+TD)

---

## Descripción

Nexus RecSys es un pipeline *end-to-end* de ciencia de datos que construye un **sistema de recomendación de productos** sobre el dataset público de comportamiento de usuarios de [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset). El proyecto cubre desde la exploración inicial del dataset hasta la evaluación de modelos avanzados de recomendación (filtrado colaborativo, VAE generativo, modelos secuenciales), siguiendo buenas prácticas de reproducibilidad y trazabilidad.

### Objetivo general

Producir un sistema capaz de recomendar productos a usuarios a partir de señales de comportamiento implícito (vistas, carritos, compras), maximizando la métrica NDCG@10 en un protocolo de evaluación con split temporal estricto y sin data leakage.

### Dataset fuente

| Archivo | Descripción | Registros |
|---|---|---|
| `events.csv` | Log de interacciones usuario-ítem (`view`, `addtocart`, `transaction`) | ~2.75 M |
| `item_properties_part1/2.csv` | Snapshot-log de atributos de ítems (precio, categoría, disponibilidad) | ~20 M |
| `category_tree.csv` | Jerarquía padre-hijo de categorías del catálogo | ~1.6 K |

---

## Estructura del repositorio

```
nexus-recsys/
├── data/
│   ├── raw/                    ← CSVs originales de Kaggle — INMUTABLES
│   ├── interim/                ← Checkpoints .parquet entre notebooks (NB01–NB05)
│   └── processed/              ← Outputs finales listos para modelado (NB06)
├── notebooks/                  ← Pipeline secuencial NB01–NB14
│   ├── 01_eda_events.ipynb                  ← EDA de eventos
│   ├── 02_eda_items_categories.ipynb        ← EDA ítems y categorías
│   ├── 03_funnel_analysis.ipynb             ← Análisis embudo de conversión
│   ├── 04_merge_pipeline.ipynb              ← Merge e integración de fuentes
│   ├── 05_synthetic_demographics.ipynb      ← Demografía sintética
│   ├── 06_feature_engineering.ipynb         ← Feature engineering final
│   ├── 07_modeling.ipynb                    ← 9 modelos base (CF, CBF, Híbrido)
│   ├── 08_business_metrics_roi.ipynb        ← SVD+TD+IPS + métricas de negocio
│   ├── 09_advanced_models.ipynb             ← EASE^R · RP3beta · BPR · NCF · SASRec
│   ├── 10_multivae.ipynb                    ← Mult-VAE^PR challenger
│   ├── 11_optimization_ensemble.ipynb       ← Optimización ensemble básico
│   ├── 12_sasrec_warm.ipynb                 ← SASRec completo (warm users)
│   ├── 13_ensemble_advanced.ipynb           ← Ensemble avanzado: Spearman + RP3+MB+TD
│   └── 14_strategies_advanced.ipynb         ← IPS · MB · LightGCN · Ensemble → 0.04069
├── scripts/                    ← Generadores, modelos y pipeline champion
│   ├── generate_modeling_notebook.py        ← Fuente de verdad de NB07
│   ├── generate_08_notebook.py              ← Fuente de verdad de NB08
│   ├── generate_09_notebook.py              ← Fuente de verdad de NB09
│   ├── generate_10_notebook.py              ← Fuente de verdad de NB10
│   ├── generate_11_notebook.py              ← Fuente de verdad de NB11
│   ├── generate_12_notebook.py              ← Fuente de verdad de NB12
│   ├── build_nb13.py                        ← Builder de NB13
│   ├── multivae_model.py                    ← Implementación PyTorch Mult-VAE^PR
│   ├── sasrec_model.py                      ← Implementación SASRec (transformer)
│   ├── _nb14v3_run.py                       ← Pipeline NB14 avanzado (IPS/MB/LightGCN)
│   ├── _nb14v3_results.json                 ← Resultados NB14: 0.04069
│   ├── _nb15v2_ensemble.py                  ← Champion script NB15v2 (greedy+Optuna)
│   └── _nb15v2_results.json                 ← Champion: NDCG@10=0.04310
├── docs/
│   ├── model_justification.md              ← Justificación técnica del modelo champion
│   ├── TRADE_OFFS.md                       ← Decisiones de diseño y trade-offs
│   ├── REPRODUCIBILITY.md                 ← Guía de reproducibilidad end-to-end
│   ├── validation_plan.md                 ← Plan de validación y protocolo de evaluación
│   ├── API_DOCS.md                        ← Documentación de la API REST
│   ├── ARTIFACTS.md                       ← Manifiesto de artefactos del repositorio
│   ├── model_comparison_final.csv         ← Tabla champion: NB13→NB14→NB15
│   └── fig_*.png / model_comparison_*.csv ← Generados por notebooks (no versionados)
├── api/
│   ├── main.py                            ← API REST FastAPI (7 endpoints)
│   └── README.md                          ← Documentación de uso de la API
├── dashboard/
│   ├── app.py                             ← Dashboard Streamlit (5 páginas)
│   ├── catalog.py                         ← Gestión del catálogo de productos
│   ├── llm_engine.py                      ← Motor LLM (Groq / NEXUS AI)
│   ├── plot_config.py                     ← Configuración de gráficos
│   ├── styles.py                          ← Estilos y CSS del dashboard
│   └── README.md                          ← Documentación del dashboard
├── encoders/                   ← Modelos serializados (.pkl) — no versionados
├── requirements.txt
└── README.md
```

---

## Pipeline de notebooks

Los notebooks están numerados y deben ejecutarse en orden. **NB07–NB10 se generan automáticamente** desde `scripts/generate_*_notebook.py` — si necesitas modificar el pipeline de modelado, edita el generador y regenera.

### 01 · EDA de Eventos

**Entrada:** `data/raw/events.csv`  
**Salida:** `data/interim/cp01_events_clean.parquet`

Exploración y limpieza del log central de comportamiento: validación de schema, distribución de tipos de evento, segmentación de usuarios por actividad, análisis de popularidad de ítems y patrones temporales.

**Hallazgos clave:**
- Tasa de conversión view→compra: ~0.7% (embudo muy estrecho, típico e-commerce)
- Más del 50% de los visitantes tienen ≤ 2 eventos → problema estructural de cold-start
- Distribución de actividad sigue ley de potencias → riesgo de sesgo de popularidad

---

### 02 · EDA de Ítems y Categorías

**Entrada:** `data/raw/item_properties_part1/2.csv`, `data/raw/category_tree.csv`  
**Salida:** `data/interim/cp02_items_flat.parquet`, `data/interim/cp02_category_enriched.parquet`

Las propiedades vienen en formato **snapshot-log vertical**. Este notebook las transforma a tabla *wide* (una fila por ítem, valor más reciente). Paralelamente resuelve la jerarquía del árbol de categorías para calcular profundidad y ancestro raíz.

---

### 03 · Análisis del Funnel de Conversión

**Entrada:** `data/interim/cp01_events_clean.parquet`  
**Salida:** `data/interim/cp03_funnel_metrics.parquet`

Análisis del embudo `view → addtocart → transaction` a nivel macro (usuarios únicos) y micro (pares user×item). Cuantifica anomalías estructurales, calcula tiempos entre etapas y construye métricas de comportamiento por visitante.

---

### 04 · Merge Pipeline — Dataset Integrado

**Entrada:** `cp01`, `cp02_items_flat`, `cp02_category_enriched`, `cp03_funnel_metrics`  
**Salida:** `data/interim/cp04_merged.parquet`

Integración progresiva de todas las fuentes con *left joins validados* (assert de integridad tras cada join).

---

### 05 · Datos Demográficos Sintéticos

**Entrada:** `cp04_merged.parquet`  
**Salida:** `data/interim/cp05_with_demographics.parquet`

El dataset RetailRocket no incluye información personal. Se generan perfiles demográficos sintéticos con distribuciones basadas en benchmarks de e-commerce LATAM: `age`, `gender`, `country`, `region`, `customer_segment` (derivado del comportamiento real), `registration_days_ago`.

> Los datos demográficos son **ficticios** y sirven únicamente como features de contexto.

---

### 06 · Feature Engineering

**Entrada:** `data/interim/cp05_with_demographics.parquet`  
**Salida:** `data/processed/` + `encoders/`

Construye el feature set final en tres granularidades:

| Artefacto | Granularidad | Descripción |
|---|---|---|
| `user_features.csv` | 1 fila / usuario | Comportamiento + demografía + encoded + scaled |
| `item_features.csv` | 1 fila / ítem | Popularidad + conversión + scaled |
| `interaction_matrix.csv` | 1 fila / par user×item | `interaction_strength`, timestamps, tipo de interacción |
| `train_test_split_info.json` | — | Fecha de corte, conteos y porcentajes train/test |

**Estadísticas del dataset procesado:**

| Métrica | Valor |
|---------|-------|
| Usuarios únicos | 1.407.580 |
| Ítems únicos | 235.061 |
| Interacciones en train | 1.763.782 |
| Interacciones en test | 381.397 |
| Sparsidad | 99.9993% |
| Split temporal (cutoff) | 2015-08-22 |
| Ítems/usuario (media) | 2.11 · mediana 1 · p95 5 |

---

### 07 · Modelado Base

**Entrada:** `data/processed/` (5 artefactos)  
**Salida:** `encoders/final_model.pkl`, `encoders/lgb_model_opt.txt`, `docs/model_comparison_final.csv`

> Generado por `scripts/generate_modeling_notebook.py`

Implementa y compara **9 modelos** en 4 familias:

| Modelo | Tipo | NDCG@10 |
|--------|------|--------:|
| Popularity Baseline | Regla heurística | — |
| SVD (k=50) | Factorización de Matrices | 0.0081 |
| NMF (k=50) | Factorización No-Negativa | — |
| LightGBM LTR | Learning-to-Rank | — |
| Item-CF (similitud coseno) | Item-Based CF | — |
| Content-Based Filtering | Perfil por features de ítem | — |
| SVD Optimizado (Optuna) | SVD + confidence weighting | **0.0081** |
| LightGBM Optimizado (Optuna) | LTR + hiperparámetros | — |
| **Híbrido SVD Opt + CBF** | α·SVD + (1-α)·CB | — |

**Modelo ganador NB07:** SVD Optimizado (NDCG@10 = 0.0081)

---

### 08 · Métricas de Negocio y ROI ★

**Entrada:** `data/processed/`  
**Salida:** `encoders/final_model_v2.pkl`, `docs/model_comparison_08_roi.csv`, figuras de negocio

> Generado por `scripts/generate_08_notebook.py`

Introduce **Temporal Decay** (λ=0.03) e **IPS Debiasing** (γ=0.4) sobre SVD, e implementa métricas de impacto de negocio: `Revenue@K`, `CTR@K`, `ConvLift@K`.

| Mejora | NDCG@10 | Δ vs SVD |
|--------|--------:|:--------:|
| SVD base | 0.0081 | — |
| SVD + TD | 0.0082 | +1.3% |
| **SVD + TD + IPS** | **0.0093** | **+14.8%** |

**Modelo ganador NB08:** SVD+TD+IPS (NDCG@10 = 0.0093)

---

### 09 · Modelos Avanzados CF ★★

**Entrada:** `data/processed/`  
**Salida:** `encoders/final_model_v3.pkl`, `docs/model_comparison_09_advanced.csv`

> Generado por `scripts/generate_09_notebook.py`

Implementa 5 familias de modelos de la literatura de recomendación:

| Modelo | Familia | NDCG@10 | Δ vs NB08 |
|--------|---------|--------:|:---------:|
| **RP3beta (α=0.85, β=0.25)** | **Random Walk CF** | **0.0258** | **+176%** |
| EASE^R (λ=500, top-20K) | CF lineal (forma cerrada) | 0.0193 | +107% |
| Ensemble (TD+IPS+EASE+BPR) | Ensemble ponderado | 0.0004 | — |
| BPR-MF (k=64) | MF bayesiano (SGD vectorizado) | 0.0012 | — |
| NCF / NeuMF (PyTorch, 80K users) | Deep Learning CF | 0.0002 | — |
| SASRec-lite (Transformer, 50K users) | DL secuencial | 0.0005 | — |

**Modelo ganador NB09 y del proyecto:** RP3beta — NDCG@10 = **0.0258** (+176% sobre NB08)

> RP3beta implementa *Random Walk with Restarts* de longitud 3 sobre el grafo bipartito usuario-ítem con penalización de popularidad (Paudel et al. 2017). Es computacionalmente eficiente (~6s) y supera a todos los modelos de Deep Learning en este dataset de sparsidad extrema (99.9993%).

---

### 10 · Mult-VAE^PR Challenger

**Entrada:** `data/processed/`  
**Salida:** `encoders/final_model_v4.pkl`, `docs/model_comparison_10_multivae.csv`

> Generado por `scripts/generate_10_notebook.py` · Implementación: `scripts/multivae_model.py`

Implementa **Mult-VAE^PR** (Variational Autoencoders for CF, Liang et al. WWW 2018) como challenger del ganador RP3beta. Implementación PyTorch pura (sin RecBole ni frameworks externos).

**Arquitectura:**
```
Input x ∈ {0,1}^20K (historial binarizado, top-20K ítems)
  → L2-norm + Dropout(0.5)
  → Linear(20K→600) → Tanh → Linear(600→200) → Tanh
  → fc_mu(200→64) | fc_logvar(200→64)    ← espacio latente z=64
  → Reparametrización: z = μ + ε·σ
  → Linear(64→200) → Tanh → Linear(200→600) → Tanh → Linear(600→20K)
  → logits ∈ ℝ^20K
```

**Configuración:** enc=[600,200], z=64, β_max=0.3 (annealing KL), 50 épocas, batch=512, 150K usuarios de entrenamiento

| Modelo | NDCG@10 | Precision@10 | Coverage | Tiempo |
|--------|--------:|-------------:|---------:|-------:|
| RP3beta (NB09) | **0.0258** | 0.00607 | 0.0600 | ~6s |
| **Mult-VAE^PR (NB10)** | **0.0255** | **0.00647** | 0.0516 | ~7.100s |
| EASE^R (NB09) | 0.0193 | 0.00477 | 0.0496 | ~130s |

**Conclusión NB10:** Mult-VAE^PR alcanza NDCG@10=0.0255, a solo −1.2% de RP3beta. Es el único modelo que compite con el ganador, pero la diferencia de 1.000× en tiempo de cómputo no justifica el reemplazo. **RP3beta mantiene la posición de modelo activo.**

---

## Resultados consolidados (NB07–NB15)

### Progresión del champion

| Notebook | Mejor modelo | NDCG@10 | Δ vs baseline RP3+TD |
|---|---|---:|---:|
| NB07 | SVD Optimizado | 0.0081 | — |
| NB08 | SVD+TD+IPS | 0.0093 | — |
| NB09 | RP3beta + EASE^R | 0.0258 | — |
| NB10 | Mult-VAE^PR | 0.0255 | — |
| NB11–NB13 | RP3+TD optimizado | ~0.028 | — |
| NB12 | SASRec warm | < RP3beta | regresión |
| **NB13-C** | **RP3+TD (baseline final)** | **0.02859** | **— (baseline)** |
| NB14-E4 | Ensemble Spearman (40 trials) | 0.04069 | +42.3% |
| **NB15v2** | **Ensemble Optimizado (100 trials)** | **0.04310** | **+50.8%** |

**Champion definitivo:** Ensemble (RP3+TD + EASE^R_500 + RP3+MB+TD)  
Pesos: {rp3_td: 0.023, ease_500: 0.021, rp3_mb_td: **0.956**} — optimizados con Optuna 100 trials.  
Exploración NB15: EASE multi-lambda [50,200,500,1000,3000] + iALS scipy + category fallback → trío original confirmado como óptimo.

> Evaluación: 2.551 usuarios test · split temporal 2015-08-22 · espacio top-20K ítems · NDCG@10

---

## Mapa de checkpoints

```
data/raw/
├── events.csv · item_properties_part1/2.csv · category_tree.csv
        │
        ▼ NB01
data/interim/cp01_events_clean.parquet        ~2.75M filas
        │
        ▼ NB02
data/interim/cp02_items_flat.parquet          ~417K ítems
data/interim/cp02_category_enriched.parquet   ~1.6K categorías
        │
        ▼ NB03
data/interim/cp03_funnel_metrics.parquet      1 fila/visitante
        │
        ▼ NB04
data/interim/cp04_merged.parquet              ~2.75M filas × N cols
        │
        ▼ NB05
data/interim/cp05_with_demographics.parquet   + perfil demográfico sintético
        │
        ▼ NB06
data/processed/user_features.csv
data/processed/item_features.csv
data/processed/interaction_matrix.csv         1.763.782 pares user×item
data/processed/train_test_split_info.json
encoders/scaler_user.pkl · scaler_item.pkl · label_encoders.pkl
        │
        ▼ NB07  (generado por scripts/generate_modeling_notebook.py)
encoders/final_model.pkl                      SVD Opt   NDCG@10=0.0081
encoders/hybrid_model.pkl
encoders/lgb_model_opt.txt
docs/model_comparison_final.csv
        │
        ▼ NB08  (generado por scripts/generate_08_notebook.py)
encoders/final_model_v2.pkl                   SVD+TD+IPS  NDCG@10=0.0093
docs/model_comparison_08_roi.csv
docs/fig_08_*.png
        │
        ▼ NB09  (generado por scripts/generate_09_notebook.py)
encoders/final_model_v3.pkl                   RP3beta  NDCG@10=0.0258  ← ACTIVO
docs/model_comparison_09_advanced.csv
docs/fig_09_*.png
        │
        ▼ NB10  (generado por scripts/generate_10_notebook.py)
encoders/final_model_v4.pkl                   Mult-VAE^PR  NDCG@10=0.0255
docs/model_comparison_10_multivae.csv
docs/fig_10_*.png
        │
        ▼ NB11–NB13  (generadores + build scripts)
data/processed/model_comparison_nb11.csv
data/processed/model_comparison_nb13.csv
data/processed/warm_*.pkl                     Secuencias SASRec (NB12)
encoders/sasrec_warm_best.pt                  Pesos SASRec  NDCG@10 < RP3beta
encoders/rp3beta_td_ips_meta.json
encoders/rp3beta_mb_td_meta.json
        │
        ▼ NB14  (scripts/_nb14v3_run.py)
data/processed/model_comparison_nb14.csv
scripts/_nb14v3_results.json                  NDCG@10=0.04069 (Ensemble Spearman)
        │
        ▼ NB15  (scripts/_nb15v2_ensemble.py)  ← CHAMPION
scripts/_nb15v2_results.json                  NDCG@10=0.04310 (+50.8% vs baseline)
docs/model_comparison_final.csv               Tabla curada NB13→NB14→NB15
```

---

## Instalación y ejecución

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd nexus-recsys

# 2. Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\Activate.ps1        # Windows PowerShell

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar datos de Kaggle
kaggle datasets download retailrocket/ecommerce-dataset -p data/raw/ --unzip
# (o descargar manualmente y colocar los 4 CSV en data/raw/)

# 5. Ejecutar EDA y feature engineering (NB01–NB06)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda_events.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda_items_categories.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_funnel_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_merge_pipeline.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_synthetic_demographics.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/06_feature_engineering.ipynb

# 6. Generar y ejecutar notebooks de modelado (NB07–NB10)
python scripts/generate_modeling_notebook.py
python scripts/generate_08_notebook.py
python scripts/generate_09_notebook.py
python scripts/generate_10_notebook.py

jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600  notebooks/07_modeling.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 notebooks/08_business_metrics_roi.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/09_advanced_models.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks/10_multivae.ipynb
```

> **Nota NB10:** El entrenamiento de Mult-VAE^PR requiere ~2 horas en CPU. Ajustar `MVAE_MAX_TRAIN_USERS` y `MVAE_EPOCHS` para pruebas rápidas.

---

## Convenciones del proyecto

| Convención | Aplicación |
|---|---|
| `snake_case` | Variables, funciones, nombres de columnas |
| Idioma | Comentarios y markdown en **español** |
| `random_state=42` | Toda operación estocástica |
| `pathlib.Path` | Todas las rutas de archivo |
| Split temporal | Cutoff fijo `2015-08-22` (percentil ~80 de fechas) |
| `.parquet` | Formato de checkpoints (preserva dtypes, ~5× más compacto que CSV) |
| Protocolo de evaluación | 3.000 usuarios warm · NDCG@K, Precision@K, Coverage · K={5,10,20} |

---

## Estado del proyecto

```
✅ EDA de eventos                (NB01)
✅ EDA de ítems y categorías     (NB02)
✅ Análisis de funnel            (NB03)
✅ Merge pipeline                (NB04)
✅ Datos demográficos sintéticos (NB05)
✅ Feature engineering           (NB06)
✅ Modelado base (9 modelos)     (NB07) ← SVD Opt NDCG@10=0.0081
✅ Mejoras + métricas ROI        (NB08) ← SVD+TD+IPS NDCG@10=0.0093
✅ Modelos avanzados CF          (NB09) ← RP3beta NDCG@10=0.0258
✅ Mult-VAE^PR challenger        (NB10) ← NDCG@10=0.0255
✅ Optimización ensemble         (NB11) ← RP3beta+TD optimizado
✅ SASRec warm users             (NB12) ← SASRec < RP3beta (sparsidad alta)
✅ Ensemble avanzado             (NB13) ← RP3+TD baseline: NDCG@10=0.02859
✅ Estrategias avanzadas         (NB14) ← Ensemble Spearman: NDCG@10=0.04069 (+42.3%)
✅ Champion NB15v2               (NB15) ← Ensemble Optimizado: NDCG@10=0.04310 (+50.8%) ★
✅ API REST                            ← FastAPI · 7 endpoints · Swagger UI
✅ Dashboard interactivo               ← Streamlit · 5 páginas · NEXUS AI integrado
```

---

## Referencias principales

| Paper | Modelo |
|-------|--------|
| Paudel et al. (2017). *Updatable, Accurate, Diverse, and Scalable Recommendations*. TIST. | RP3beta |
| Steck (2019). *Embarrassingly Shallow Autoencoders for Sparse Data*. WWW 2019. | EASE^R |
| Liang et al. (2018). *Variational Autoencoders for Collaborative Filtering*. WWW 2018. | Mult-VAE^PR |
| Dacrema et al. (2019). *Are We Really Making Much Progress?* RecSys 2019. | Benchmark |
| Rendle et al. (2009). *BPR: Bayesian Personalized Ranking*. UAI 2009. | BPR-MF |
| Kang & McAuley (2018). *Self-Attentive Sequential Recommendation*. ICDM 2018. | SASRec |

Ver justificación técnica completa en [docs/model_justification.md](docs/model_justification.md)

---

## API REST y Dashboard

### Lanzar la API

```bash
# Desde la raíz del proyecto
uvicorn api.main:app --reload --port 8000

# Documentación interactiva Swagger UI:
http://localhost:8000/docs
```

### Lanzar el Dashboard

```bash
# Configurar el LLM (opcional — NEXUS AI)
cp .env.example .env
# Editar .env y completar GROQ_API_KEY

# Lanzar Streamlit
streamlit run dashboard/app.py
# Abre en http://localhost:8501
```

Ver documentación completa de la API en [docs/API_DOCS.md](docs/API_DOCS.md) y del dashboard en [dashboard/README.md](dashboard/README.md).
