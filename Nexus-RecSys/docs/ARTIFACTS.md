# Nexus RecSys — Manifiesto de Artifacts

**Versión:** 3.0 — Proyecto completo (NB01–NB15 · API REST · Dashboard) | Champion: NDCG@10 = **0.04310** (+50.8% vs baseline)

Este documento clasifica cada archivo del repositorio según su origen:  
**A** = creado/editado a mano (commitear en git) | **G** = generado por notebook/script (reproducible, en `.gitignore`)

---

## Archivos comprometidos en Git (origen manual)

### Raíz y configuración
| Archivo | Descripción |
|---|---|
| `README.md` | Documentación principal del proyecto (pipeline completo NB01–NB15 + API + Dashboard) |
| `requirements.txt` | Dependencias Python del entorno virtual |
| `.gitignore` | Reglas de exclusión para Git |
| `.env.example` | Plantilla pública de variables de entorno (sin credenciales) |

### docs/ — documentación técnica (versionada)
| Archivo | Descripción |
|---|---|
| `docs/model_justification.md` | Justificación técnica del modelo final v7.0 (NB01–NB15, champion NB15v2) |
| `docs/TRADE_OFFS.md` | Decisiones de diseño, trade-offs y alternativas consideradas |
| `docs/REPRODUCIBILITY.md` | Guía de reproducibilidad end-to-end con comandos paso a paso |
| `docs/validation_plan.md` | Plan de validación y protocolo de evaluación (split temporal, métricas) |
| `docs/API_DOCS.md` | Documentación de la API REST (endpoints, schemas, ejemplos curl) |
| `docs/ARTIFACTS.md` | Este manifiesto |
| `docs/model_comparison_final.csv` | Tabla curada de progresión: RP3+TD → Ensemble NB14 → Champion NB15v2 |

### scripts/ — generadores, modelos y pipeline final
| Archivo | Tipo | Descripción |
|---|---|---|
| `scripts/generate_modeling_notebook.py` | A | Generador del notebook NB07 (modelado base) |
| `scripts/generate_08_notebook.py` | A | Generador del notebook NB08 (métricas ROI) |
| `scripts/generate_09_notebook.py` | A | Generador del notebook NB09 (EASE^R, RP3beta, BPR, NCF, SASRec) |
| `scripts/generate_10_notebook.py` | A | Generador del notebook NB10 (Mult-VAE^PR) |
| `scripts/generate_11_notebook.py` | A | Generador del notebook NB11 (optimización ensemble) |
| `scripts/generate_12_notebook.py` | A | Generador del notebook NB12 (SASRec warm users) |
| `scripts/build_nb13.py` | A | Builder del notebook NB13 (ensemble avanzado) |
| `scripts/multivae_model.py` | A | Implementación PyTorch de Mult-VAE^PR (Liang et al. WWW 2018) |
| `scripts/sasrec_model.py` | A | Implementación SASRec (transformer secuencial, Wang et al. 2018) |
| `scripts/build_product_catalog.py` | A | Genera `data/processed/product_catalog.json` con nombres de productos |
| `scripts/validate_data.py` | A | Valida integridad y formato de los datos raw antes del pipeline |
| `scripts/validate_artifacts.py` | A | Verifica que todos los artefactos requeridos existan antes de inferencia |
| `scripts/run_pipeline.sh` | A | Script bash de ejecución secuencial completa NB01–NB15 |
| `scripts/_nb14v3_run.py` | A | Pipeline NB14 avanzado: IPS, Multi-Behavior, LightGCN, Ensemble Spearman |
| `scripts/_nb14v3_results.json` | A | Resultados NB14: NDCG@10=0.04069 (Ensemble Spearman) |
| `scripts/_nb15v2_ensemble.py` | A | **Champion script NB15v2**: greedy selection + 100 Optuna trials |
| `scripts/_nb15v2_results.json` | A | **Resultados champion**: NDCG@10=0.04310 (+50.8% vs baseline) |

### api/ — API REST (FastAPI)
| Archivo | Tipo | Descripción |
|---|---|---|
| `api/main.py` | A | API REST FastAPI: health check, recomendaciones, similares, cold-start |
| `api/README.md` | A | Documentación de uso con ejemplos curl y descripción de endpoints |

### dashboard/ — Dashboard interactivo (Streamlit)
| Archivo | Tipo | Descripción |
|---|---|---|
| `dashboard/app.py` | A | Aplicación Streamlit principal (5 páginas) |
| `dashboard/catalog.py` | A | Gestión del catálogo de productos y nombres de ítems |
| `dashboard/llm_engine.py` | A | Motor de LLM: integración Groq (NEXUS AI) con fallback estático |
| `dashboard/plot_config.py` | A | Configuración de gráficos Plotly (paleta, temas, helpers) |
| `dashboard/styles.py` | A | CSS y estilos del dashboard |
| `dashboard/README.md` | A | Documentación del dashboard (páginas, requisitos, modo sin datos) |

### notebooks/ — pipeline completo de análisis y modelado
| Archivo | Descripción |
|---|---|
| `notebooks/01_eda_events.ipynb` | EDA de eventos del dataset RetailRocket |
| `notebooks/02_eda_items_categories.ipynb` | EDA de ítems y árbol de categorías |
| `notebooks/03_funnel_analysis.ipynb` | Análisis del embudo view→cart→transaction |
| `notebooks/04_merge_pipeline.ipynb` | Pipeline de unión y limpieza de fuentes |
| `notebooks/05_synthetic_demographics.ipynb` | Generación de demografía sintética (Faker) |
| `notebooks/06_feature_engineering.ipynb` | Ingeniería de features: user, item, interaction |
| `notebooks/07_modeling.ipynb` | Modelado base: 9 modelos (CF, CBF, Híbrido, LightGBM) |
| `notebooks/07_modeling_output.ipynb` | Variante con salida extendida de NB07 |
| `notebooks/08_business_metrics_roi.ipynb` | SVD+TD+IPS + métricas ROI + análisis de segmentos |
| `notebooks/09_advanced_models.ipynb` | EASE^R · RP3beta · BPR-MF · NCF · SASRec-lite |
| `notebooks/10_multivae.ipynb` | Mult-VAE^PR challenger vs RP3beta |
| `notebooks/11_optimization_ensemble.ipynb` | Optimización de hiperparámetros y ensemble básico |
| `notebooks/12_sasrec_warm.ipynb` | SASRec completo (warm users, split temporal real) |
| `notebooks/13_ensemble_advanced.ipynb` | Ensemble avanzado: correlación Spearman, RP3+MB+TD |
| `notebooks/14_strategies_advanced.ipynb` | IPS, Multi-Behavior, LightGCN, Ensemble Spearman → 0.04069 |

### data/ — marcadores de directorio
| Archivo | Descripción |
|---|---|
| `data/raw/.gitkeep` | Marcador — directorio vacío hasta descarga de Kaggle |
| `data/interim/.gitkeep` | Marcador — rellenado por NB01–NB05 |
| `data/processed/.gitkeep` | Marcador — rellenado por NB06–NB14 |

### encoders/ — marcador de directorio
| Archivo | Descripción |
|---|---|
| `encoders/.gitkeep` | Marcador — rellenado por NB06–NB14 |

---

## Archivos generados (NO comprometidos, reproducibles)

### data/raw/ — descarga manual de Kaggle
| Archivo | Descripción |
|---|---|
| `data/raw/events.csv` | Eventos de interacción (2.75M: view, addtocart, transaction) |
| `data/raw/item_properties_part1.csv` | Propiedades de ítems — parte 1 |
| `data/raw/item_properties_part2.csv` | Propiedades de ítems — parte 2 |
| `data/raw/category_tree.csv` | Árbol jerárquico de categorías |

> Descargar de: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

### data/interim/ — generados por NB01–NB05
| Archivo | Generado por |
|---|---|
| `data/interim/cp01_events_clean.parquet` | NB01 |
| `data/interim/cp02_items_flat.parquet` | NB02 |
| `data/interim/cp02_category_enriched.parquet` | NB02 |
| `data/interim/cp03_funnel_metrics.parquet` | NB03 |
| `data/interim/cp04_merged.parquet` | NB04 |
| `data/interim/cp05_with_demographics.parquet` | NB05 |

### data/processed/ — generados por NB06–NB14
| Archivo | Generado por | Descripción |
|---|---|---|
| `cp06_features_final.parquet` | NB06 | Dataset final con todas las features |
| `user_features.csv` | NB06 | Features de usuarios |
| `item_features.csv` | NB06 | Features de ítems |
| `interaction_matrix.csv` | NB06 | Matriz interacciones: 1.763.782 pares user×item |
| `train_test_split_info.json` | NB07 | Metadata del split temporal (cutoff 2015-08-22) |
| `model_comparison_final.csv` | NB07–NB15 | Tabla comparativa acumulada (copia en data/processed) |
| `model_comparison_nb11.csv` | NB11 | Comparativa NB11 |
| `model_comparison_nb12.csv` | NB12 | Comparativa NB12 SASRec |
| `model_comparison_nb13.csv` | NB13 | Comparativa NB13 ensemble avanzado |
| `model_comparison_nb14.csv` | NB14 | Comparativa NB14 estrategias avanzadas |
| `model_comparison_nb15.csv` | NB15 | Comparativa NB15 EASE multi-lambda e iALS |
| `warm_dataset_info.json` | NB12 | Info dataset warm users (SASRec) |
| `warm_item_mapping.pkl` | NB12 | Mapping ítem→id warm |
| `warm_user_mapping.pkl` | NB12 | Mapping usuario→id warm |
| `warm_sequences_train.pkl` | NB12 | Secuencias de train (SASRec) |
| `warm_sequences_val.pkl` | NB12 | Secuencias de validación (SASRec) |
| `warm_sequences_test.pkl` | NB12 | Secuencias de test (SASRec) |

### encoders/ — modelos serializados
| Archivo | Generado por | NDCG@10 | Descripción |
|---|---|---|---|
| `scaler_user.pkl` | NB06 | — | StandardScaler features de usuarios |
| `scaler_item.pkl` | NB06 | — | StandardScaler features de ítems |
| `label_encoders.pkl` | NB06 | — | LabelEncoders columnas categóricas |
| `lgb_model_opt.txt` | NB07 | — | Modelo LightGBM (re-ranker Stage 2) |
| `hybrid_model.pkl` | NB07 | — | Artefacto híbrido NB07 |
| `final_model_v4.pkl` | NB10 | 0.0255 | Mult-VAE^PR — metadata NB10 |
| `sasrec_warm_best.pt` | NB12 | — | Pesos SASRec completo (warm users) |
| `sasrec_warm_config.json` | NB12 | — | Hiperparámetros SASRec NB12 |
| `rp3beta_td_ips_meta.json` | NB14 | 0.02836 | Metadatos RP3+TD+IPS (γ=0.1) |
| `rp3beta_mb_td_meta.json` | NB14 | 0.01890 | Metadatos RP3+MB+TD (pesos Optuna) |

### scripts/ — artefactos generados (excluidos por .gitignore)
| Archivo/Directorio | Descripción |
|---|---|
| `scripts/_score_cache/` | Matrices de scores (18 archivos .npy, ~2GB total) |
| `scripts/_cat_mapping.json` | Mapping categoría→ítems (417K ítems, 758 cat.) |
| `scripts/_nb12_state.pkl` | Estado SASRec entrenado |
| `scripts/_nb12_*.pkl` | Checkpoints NB12 |
| `scripts/_nb12_*.pt` | Pesos modelo SASRec |
| `scripts/*_log.txt` | Logs de ejecución |
| `scripts/_nb10_run.py` | Runner NB10 (supersedido) |
| `scripts/_nb12_s*.py` | Scripts de ejecución por fases NB12 |
| `scripts/_nb14_run.py` | Runner NB14 v1 (supersedido por v3) |
| `scripts/_nb14_extracted.py` | Extracción temporal de NB14 |
| `scripts/_nb15_improvement.py` | NB15v1 exploración (supersedida por v2) |
| `scripts/_nb15_results.json` | Resultados NB15v1 (0.03950, regresión) |

### docs/ — figuras y tablas generadas (reproducibles)
| Archivo | Generado por |
|---|---|
| `docs/fig_dataset_stats.png` | NB07 |
| `docs/fig_model_comparison.png` | NB07 |
| `docs/fig_08_model_comparison_roi.png` | NB08 |
| `docs/fig_08_roi_business_impact.png` | NB08 |
| `docs/fig_08_segment_analysis.png` | NB08 |
| `docs/fig_09_model_comparison_advanced.png` | NB09 |
| `docs/fig_09_convergence_top.png` | NB09 |
| `docs/fig_10_multivae_comparison.png` | NB10 |
| `docs/fig_10_multivae_convergence.png` | NB10 |
| `docs/model_comparison_08_roi.csv` | NB08 |
| `docs/model_comparison_09_advanced.csv` | NB09 |
| `docs/model_comparison_10_multivae.csv` | NB10 |

---

## Progresión de resultados — historia del proyecto

| Notebook | Mejor modelo | NDCG@10 | Delta vs baseline |
|---|---|---|---|
| NB07 | SVD Optimizado | 0.0081 | — |
| NB08 | SVD+TD+IPS | 0.0093 | — |
| NB09 | RP3beta + EASE^R | 0.0258 | — |
| NB10 | Mult-VAE^PR | 0.0255 | — |
| NB11 | RP3beta+TD optimizado | ~0.028 | — |
| NB12 | SASRec warm | < RP3beta | regresión vs clásico |
| NB13-C | RP3+TD (baseline final) | **0.02859** | baseline |
| NB14-E4 | Ensemble Spearman | 0.04069 | +42.3% |
| **NB15v2** | **Ensemble Optimizado** | **0.04310** | **+50.8%** |

**Champion definitivo:** Ensemble (RP3+TD + EASE^R_500 + RP3+MB+TD) — pesos {0.023 / 0.021 / **0.956**} optimizados con 100 trials Optuna.

### Raíz y configuración
| Archivo | Descripción |
|---|---|
| `README.md` | Documentación principal del proyecto (pipeline completo NB01–NB10) |
| `requirements.txt` | Dependencias Python del entorno virtual |
| `.gitignore` | Reglas de exclusión para Git |

### docs/ — documentación técnica (versionada)
| Archivo | Descripción |
|---|---|
| `docs/model_justification.md` | Justificación técnica del modelo final v5.0 (16 modelos, NB07–NB10) |
| `docs/ARTIFACTS.md` | Este manifiesto |

### scripts/ — generadores y módulos reutilizables
| Archivo | Descripción |
|---|---|
| `scripts/generate_modeling_notebook.py` | Generador del notebook NB07 (modelado base) |
| `scripts/generate_08_notebook.py` | Generador del notebook NB08 (métricas ROI) |
| `scripts/generate_09_notebook.py` | Generador del notebook NB09 (EASE^R, RP3beta, BPR, NCF, SASRec) |
| `scripts/generate_10_notebook.py` | Generador del notebook NB10 (Mult-VAE^PR challenger) |
| `scripts/multivae_model.py` | Implementación PyTorch pura de Mult-VAE^PR (Liang et al. WWW 2018) |

### notebooks/ — pipeline de análisis y modelado
| Archivo | Descripción |
|---|---|
| `notebooks/01_eda_events.ipynb` | EDA de eventos del dataset RetailRocket |
| `notebooks/02_eda_items_categories.ipynb` | EDA de ítems y árbol de categorías |
| `notebooks/03_funnel_analysis.ipynb` | Análisis del embudo view→cart→transaction |
| `notebooks/04_merge_pipeline.ipynb` | Pipeline de unión y limpieza de fuentes |
| `notebooks/05_synthetic_demographics.ipynb` | Generación de demografía sintética (Faker) |
| `notebooks/06_feature_engineering.ipynb` | Ingeniería de features: user, item, interaction |
| `notebooks/07_modeling.ipynb` | Modelado base: 9 modelos (CF, CBF, Híbrido, LightGBM) |
| `notebooks/07_modeling_output.ipynb` | Variante con salida extendida de NB07 |
| `notebooks/08_business_metrics_roi.ipynb` | SVD+TD+IPS + métricas ROI + análisis de segmentos |
| `notebooks/09_advanced_models.ipynb` | EASE^R · RP3beta · BPR-MF · NCF · SASRec-lite |
| `notebooks/10_multivae.ipynb` | Mult-VAE^PR challenger vs RP3beta (NB10) |

### data/ — marcadores de directorio
| Archivo | Descripción |
|---|---|
| `data/raw/.gitkeep` | Marcador — directorio vacío hasta descarga de Kaggle |
| `data/interim/.gitkeep` | Marcador — rellenado por NB01–NB05 |
| `data/processed/.gitkeep` | Marcador — rellenado por NB06 |
| `encoders/.gitkeep` | Marcador — rellenado por NB07–NB10 |

---

## Archivos generados (NO comprometidos, reproducibles)

### data/raw/ — descarga manual de Kaggle
| Archivo | Descripción |
|---|---|
| `data/raw/events.csv` | Eventos de interacción (view, addtocart, transaction) |
| `data/raw/item_properties_part1.csv` | Propiedades de ítems — parte 1 |
| `data/raw/item_properties_part2.csv` | Propiedades de ítems — parte 2 |
| `data/raw/category_tree.csv` | Árbol jerárquico de categorías |

> Descargar de: https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

### data/interim/ — generados por NB01–NB05
| Archivo | Generado por |
|---|---|
| `data/interim/cp01_events_clean.parquet` | NB01 |
| `data/interim/cp02_items_flat.parquet` | NB02 |
| `data/interim/cp02_category_enriched.parquet` | NB02 |
| `data/interim/cp03_funnel_metrics.parquet` | NB03 |
| `data/interim/cp04_merged.parquet` | NB04 |
| `data/interim/cp05_with_demographics.parquet` | NB05 |

### data/processed/ — generados por NB06–NB10
| Archivo | Generado por | Descripción |
|---|---|---|
| `cp06_features_final.parquet` | NB06 | Dataset final con todas las features |
| `user_features.csv` | NB06 | Features de usuarios |
| `item_features.csv` | NB06 | Features de ítems |
| `interaction_matrix.csv` | NB06 | Matriz interacciones: 1.763.782 pares user×item |
| `train_test_split_info.json` | NB07 | Metadata del split temporal (cutoff 2015-08-22) |
| `model_comparison_final.csv` | NB07 | Tabla comparativa 9 modelos NB07 |
| `model_comparison_08_roi.csv` | NB08 | Tabla comparativa NB08 con métricas ROI |
| `model_comparison_09_advanced.csv` | NB09 | Tabla comparativa NB09: EASE^R, RP3beta, BPR, NCF, SASRec |
| `model_comparison_10_multivae.csv` | NB10 | Tabla comparativa NB10: todos los modelos + Mult-VAE^PR |

### encoders/ — modelos serializados (.pkl)
| Archivo | Generado por | NDCG@10 | Descripción |
|---|---|---|---|
| `scaler_user.pkl` | NB06 | — | StandardScaler features de usuarios |
| `scaler_item.pkl` | NB06 | — | StandardScaler features de ítems |
| `label_encoders.pkl` | NB06 | — | LabelEncoders columnas categóricas |
| `lgb_model_opt.txt` | NB07 | — | Modelo LightGBM (re-ranker Stage 2) |
| `final_model.pkl` | NB07 | 0.0081 | SVD Opt — mejor modelo NB07 |
| `hybrid_model.pkl` | NB07 | — | Artefacto híbrido NB07 |
| `final_model_v2.pkl` | NB08 | 0.0093 | SVD+TD+IPS — mejor NB08 |
| `final_model_v3.pkl` | NB09 | **0.0258** | **RP3beta + EASE^R — modelo activo NB09** |
| `final_model_v4.pkl` | NB10 | 0.0255 | Mult-VAE^PR — metadata NB10 |
| `sasrec_warm_best.pt` | NB12 | — | Pesos SASRec completo (warm users) |
| `sasrec_warm_config.json` | NB12 | — | Hyperparámetros SASRec NB12 |
| **`rp3beta_td_ips_meta.json`** | **NB14** | **0.02836** | **Metadatos RP3+TD+IPS (γ=0.1)** |
| **`rp3beta_mb_td_meta.json`** | **NB14** | **0.01890** | **Metadatos RP3+MB+TD (pesos Optuna)** |

### docs/ — figuras y tablas generadas
| Archivo | Generado por | Descripción |
|---|---|---|
| `fig_dataset_stats.png` | NB07 | Estadísticas del dataset |
| `fig_model_comparison.png` | NB07 | Comparativa visual NB07 |
| `fig_08_model_comparison_roi.png` | NB08 | Comparativa con métricas ROI |
| `fig_08_roi_business_impact.png` | NB08 | Impacto de negocio estimado |
| `fig_08_segment_analysis.png` | NB08 | Análisis por segmento de usuario |
| `fig_09_model_comparison_advanced.png` | NB09 | Comparativa visual NB09 |
| `fig_09_convergence_top.png` | NB09 | Curvas de convergencia modelos NB09 |
| `fig_10_multivae_comparison.png` | NB10 | Barplot + scatter NDCG vs Coverage NB10 |
| `fig_10_multivae_convergence.png` | NB10 | Curvas loss/KL/beta del training Mult-VAE^PR |
| `model_comparison_final.csv` | NB07 | Copia en docs/ de la tabla NB07 |
| `model_comparison_08_roi.csv` | NB08 | Copia en docs/ de la tabla NB08 |
| `model_comparison_09_advanced.csv` | NB09 | Tabla principal comparativa NB09 |
| `model_comparison_10_multivae.csv` | NB10 | Tabla principal comparativa NB10 |
| **`model_comparison_nb14.csv`** | **NB14** | **Tabla comparativa 4 estrategias NB14** |

---

## Reglas en `.gitignore`

```gitignore
data/raw/*             # descargado de Kaggle
data/interim/*         # generado por NB01-NB05
data/processed/*       # generado por NB06
encoders/*             # pkl serializados (varios GB)
docs/*.csv             # tablas generadas por notebooks
docs/*.png             # figuras generadas por notebooks
scripts/_*.py          # scripts de ejecución temporal
scripts/_*.txt         # logs de ejecución

# Whitelists — sí se versionan:
!docs/model_justification.md
!docs/ARTIFACTS.md
!data/raw/.gitkeep
!data/interim/.gitkeep
!data/processed/.gitkeep
!encoders/.gitkeep
```

---

## Pipeline de reproducción completa

```bash
# 1. Descargar datos de Kaggle
kaggle datasets download retailrocket/ecommerce-dataset -p data/raw/ --unzip

# 2. EDA y feature engineering (NB01-NB06)
for nb in 01_eda_events 02_eda_items_categories 03_funnel_analysis \
          04_merge_pipeline 05_synthetic_demographics 06_feature_engineering; do
  jupyter nbconvert --to notebook --execute "notebooks/${nb}.ipynb" --inplace
done

# 3. Modelado (NB07-NB10) — regenerar desde generadores antes de ejecutar
python scripts/generate_modeling_notebook.py  # → notebooks/07_modeling.ipynb
python scripts/generate_08_notebook.py        # → notebooks/08_business_metrics_roi.ipynb
python scripts/generate_09_notebook.py        # → notebooks/09_advanced_models.ipynb
python scripts/generate_10_notebook.py        # → notebooks/10_multivae.ipynb

for nb in "07_modeling" "08_business_metrics_roi" "09_advanced_models" "10_multivae"; do
  jupyter nbconvert --to notebook --execute "notebooks/${nb}.ipynb" \
    --inplace --ExecutePreprocessor.timeout=7200
done

# Tiempos estimados en CPU:
#   NB01–NB08: ~30 min total
#   NB09:      ~25 min (EASE^R ~130s + RP3beta ~70s + NCF/SASRec ~25min)
#   NB10:      ~120 min (Mult-VAE^PR 50 épocas × 150K usuarios)
```

---

## Resultados de evaluación por versión de modelo

Protocolo fijo en todos los notebooks: **3.000 usuarios warm**, split temporal cutoff `2015-08-22`,
evaluación NDCG@K sobre historial de test, sparsidad=0.999993.

| Versión | Notebook | Modelo | NDCG@10 | Mejora acumulada |
|---------|---------|--------|--------:|:----------------:|
| v1 | NB07 | SVD (k=50) optimizado | 0.0081 | baseline |
| v2 | NB08 | SVD + TD + IPS | 0.0093 | +14.8% |
| **v3** | **NB09** | **RP3beta (α=0.85, β=0.25)** | **0.0258** | **+219%** |
| v4 | NB10 | Mult-VAE^PR (enc=[600,200], z=64) | 0.0255 | +215% |

> **Modelo activo en producción:** `encoders/final_model_v3.pkl` — RP3beta
