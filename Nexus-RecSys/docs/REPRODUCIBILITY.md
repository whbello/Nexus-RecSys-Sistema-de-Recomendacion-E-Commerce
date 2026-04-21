# Reproducibilidad — nexus-recsys

**Proyecto:** Sistema de Recomendación de Productos  
**Dataset:** RetailRocket E-Commerce (Kaggle)  
**Modelo ganador:** Mega-Ensemble NB15v2 · NDCG@10 = **0.04310**

---

## Requisitos del sistema

| Componente | Mínimo recomendado |
|---|---|
| Python | 3.9+ (testeado en 3.11) |
| RAM | 16 GB (para cargar interaction_matrix ~2 GB en memoria) |
| Disco | 5 GB libres (datos + score cache) |
| CPU | 4 núcleos (entrenamiento RP3beta tarda ~6s en 4 cores) |
| GPU | No requerida (todos los modelos corren en CPU) |
| OS | Linux / macOS / Windows (WSL recomendado para el script bash) |

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd nexus-recsys
```

### 2. Crear entorno virtual

```bash
# Opción A — venv estándar
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows

# Opción B — conda
conda create -n nexus python=3.11
conda activate nexus
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar datos raw (solo si se quiere reproducir desde cero)

Los datos originales de RetailRocket están disponibles en Kaggle:
```
https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
```
Descargar y colocar en `data/raw/`:
```
data/raw/
├── events.csv                         (~2.75 M filas)
├── item_properties_part1.csv          (~10 M filas)
├── item_properties_part2.csv          (~10 M filas)
└── category_tree.csv                  (~1.6 K filas)
```

Si los archivos procesados ya están en `data/processed/` (como ocurre en el
repositorio), este paso puede omitirse.

---

## Ejecución completa (pipeline end-to-end)

```bash
# Desde la raíz del proyecto
bash scripts/run_pipeline.sh
```

Este comando ejecuta secuencialmente:
1. Verifica e instala dependencias
2. Valida datos de entrada
3. EDA y ETL (NB01–NB04)
4. Feature Engineering (NB05–NB06)
5. Modelado base y avanzado (NB07–NB11, NB13–NB14)
6. Ensemble champion NB15v2
7. Validación de artefactos

**Tiempo estimado:** 45–90 minutos en hardware estándar.

---

## Ejecución parcial (solo inferencia con el modelo guardado)

Si los artefactos ya están generados (archivos `.npy` en `scripts/_score_cache/`),
se puede usar la API directamente sin reentrenar:

```bash
# Verificar que los artefactos están listos
python scripts/validate_artifacts.py

# Iniciar la API REST
uvicorn api.main:app --reload --port 8000

# Acceder a la documentación interactiva
# http://localhost:8000/docs
```

---

## Ejecución del dashboard

```bash
# Iniciar el dashboard Streamlit
streamlit run dashboard/app.py

# Acceder en el navegador
# http://localhost:8501
```

---

## Reproductibilidad total

Todas las semillas aleatorias están fijadas en `random_state=42` / `seed=42`:
- División de usuarios train/validation/test
- Selección del subconjunto de 3000 usuarios de evaluación
- Trials de Optuna (sampler TPE con seed=42)

Los archivos de score cache en `scripts/_score_cache/*.npy` son deterministas
dado el mismo `interaction_matrix.csv` y el mismo código de entrenamiento.

---

## Ejecución notebook por notebook

Para explorar el proyecto paso a paso:

```bash
# Análisis exploratorio
jupyter notebook notebooks/01_eda_events.ipynb
jupyter notebook notebooks/02_eda_items_categories.ipynb
jupyter notebook notebooks/03_funnel_analysis.ipynb

# Pipeline de datos
jupyter notebook notebooks/04_merge_pipeline.ipynb
jupyter notebook notebooks/05_synthetic_demographics.ipynb
jupyter notebook notebooks/06_feature_engineering.ipynb

# Modelado base
jupyter notebook notebooks/07_modeling.ipynb

# Métricas de negocio
jupyter notebook notebooks/08_business_metrics_roi.ipynb

# Modelos avanzados (EASE^R, RP3beta, NCF, SASRec)
jupyter notebook notebooks/09_advanced_models.ipynb

# Mult-VAE
jupyter notebook notebooks/10_multivae.ipynb

# Optimización y ensemble básico
jupyter notebook notebooks/11_optimization_ensemble.ipynb

# SASRec sobre warm users
jupyter notebook notebooks/12_sasrec_warm.ipynb

# Ensemble avanzado (NB14 Spearman ensemble)
jupyter notebook notebooks/13_ensemble_advanced.ipynb

# Champion: Mega-Ensemble NB15v2
python scripts/_nb15v2_ensemble.py
```

---

## Troubleshooting de errores comunes

### Error: `ModuleNotFoundError: No module named 'fastapi'`
```bash
pip install fastapi uvicorn pydantic
```

### Error: `ModuleNotFoundError: No module named 'streamlit'`
```bash
pip install streamlit plotly
```

### Error: `MemoryError` al cargar interaction_matrix
- Asegurar al menos 16 GB de RAM disponibles
- Cerrar otras aplicaciones que consuman memoria
- Alternativa: usar una muestra del dataset con `pd.read_csv(..., nrows=500_000)`

### Error: `FileNotFoundError` en score cache
```bash
# Regenerar score cache ejecutando el pipeline de entrenamiento
python scripts/_nb14v3_run.py     # genera rp3_mb_td, ease_*, rp3_td
```

### Error de codificación en Windows (UnicodeDecodeError)
```bash
# Agregar al inicio del script problemático:
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
```

### Los notebooks no ejecutan con nbconvert
```bash
# Verificar que el kernel esté registrado
python -m ipykernel install --user --name=python3
```

---

## Variables de entorno opcionales

```bash
# Directorio de datos (por defecto: <ROOT>/data)
export NEXUS_DATA_DIR=/ruta/alternativa/data

# Puerto de la API (por defecto: 8000)
export NEXUS_API_PORT=8080

# Puerto del dashboard (por defecto: 8501)
export NEXUS_DASHBOARD_PORT=8502
```

---

*Documento: docs/REPRODUCIBILITY.md | nexus-recsys v1.0 | Abril 2026*
