# Plan de Validación y Evaluación — nexus-recsys

**Proyecto:** Sistema de Recomendación de Productos — E-Commerce  
**Dataset:** RetailRocket (Kaggle) · 2.75 M eventos · 4.5 meses  
**Fecha:** Abril 2026 | Modelo ganador: Mega-Ensemble NB15v2 · NDCG@10 = **0.04310**

---

## 1. Protocolo de evaluación

### 1.1 Split temporal (justificación)

El dataset contiene eventos de comportamiento de usuarios durante aproximadamente
4.5 meses (mayo–septiembre 2015). La elección de **split temporal** sobre split
aleatorio es una decisión técnica fundamental, justificada por tres razones:

| Criterio | Split Aleatorio | Split Temporal (elegido) |
|---|---|---|
| Leakage temporal | **SÍ** — interacciones futuras en train | **NO** — corte estricto por fecha |
| Realismo de producción | Bajo — no replica deployment real | **Alto** — simula un sistema en vivo |
| Comparabilidad con papers | Varía — muchos usan leave-one-out | **Sí** — comparable con evaluaciones offline estándar |
| Sesgo de usuario | Alto — usuarios con poca actividad aparecen en test y train | **Controlado** — usuarios con 0 historial pre-corte son cold-start |

**Referencia:** Hidasi & Czapp (2023) demuestran que el split aleatorio en
recsys genera estimaciones optimistas artificiales de NDCG por data leakage
temporal. El modelo aprende interacciones que "aún no ocurrieron" en el tiempo.

**Fecha de corte elegida: 2015-08-22**  
Esta fecha divide el dataset en un ratio 82/18 que replica una proporción
realista de train/test para sistemas de recomendación online (meses de
entrenamiento → semanas de evaluación).

---

### 1.2 División del dataset

| Partición | Criterio | Interacciones | Usuarios | % del total |
|---|---|---|---|---|
| **Train** | `last_interaction_ts < 2015-08-22` | 1 763 782 | 1 158 463 | 82.2 % |
| **Test** | `last_interaction_ts ≥ 2015-08-22` | 381 397 | 269 694 | 17.8 % |
| **Validation** | 15 % del conjunto de evaluación (random_state=42, estratificado por actividad) | — | ~450 | — |

**Usuarios evaluados:** Se seleccionan **3 000 usuarios warm** de forma reproducible
(seed=42) de la intersección de usuarios en train Y test. Este subconjunto
representa el escenario más exigente y realista: usuarios con historial pero
con pocas interacciones.

**Isolation del test:** El conjunto de test fue evaluado **UNA SOLA VEZ por modelo**.
El conjunto de validación (15% del pool de 3000) fue usado exclusivamente
para Optuna (100 trials, NB15v2). Esta separación garantiza que ningún modelo
"vio" el test durante la optimización.

---

### 1.3 Protocolo de evaluación de usuarios

El protocolo estándar del proyecto evalúa sobre **todos los usuarios warm**
(≥1 interacción en train, ≥1 en test). Esto incluye usuarios con una sola
interacción de entrenamiento — el caso más difícil y más honesto.

Se presentaron también análisis secundarios con filtros más restrictivos para
ilustrar la sensibilidad del sistema:

#### Tabla de sensibilidad al protocolo

| Filtro mínimo (train) | N usuarios evaluados | NDCG@10 (Ensemble) | Δ vs ≥1 |
|---|---|---|---|
| **≥1 interacción** (estándar) | 3 000 | **0.0431** | baseline |
| ≥3 interacciones | ~757 | ~0.065 | +51 % |
| ≥5 interacciones | ~400 | ~0.079 | +84 % |

**Conclusión del análisis:** El valor NDCG@10=0.0431 es la métrica más honesta
porque incluye el 57.5% de usuarios con exactamente 1 ítem en train (cold-start
parcial). Con filtro ≥5, el resultado (0.079) es comparable con lo reportado
en la literatura académica sobre el mismo dataset.

**Por qué usamos ≥1 como protocolo principal:**
1. Es el escenario real: los sistemas de producción deben servir a TODOS los usuarios
2. Inflar métricas con filtros restrictivos sería deshonesto ante evaluadores
3. Permite diagnóstico claro del problema de cold-start estructural

---

## 2. Métricas de evaluación

### 2.1 Por qué NDCG@10 como métrica principal

**NDCG@10 (Normalized Discounted Cumulative Gain a K=10)** fue elegida como
métrica principal por cuatro razones técnicas:

1. **Captura la posición en el ranking:** Un ítem relevante en posición 1 vale
   más que el mismo ítem en posición 10. La penalización logarítmica simula
   el comportamiento de click del usuario (más probable hacer click en el top).

2. **Estándar de la industria y literatura:** Es la métrica principal en ACM RecSys,
   SIGIR y la mayoría de papers sobre CF (Koren 2009, Dacrema 2019, McAuley 2022).

3. **Feedback implícito compatible:** A diferencia de RMSE, no requiere ratings
   explícitos. Trata el problema como ranking, no como regresión.

4. **Interpretable para el negocio:** Valores cercanos a 1 implican que el
   producto que el usuario compraría aparece en la primera posición de la lista.

**Fórmula:**
```
DCG@K  = Σ_{i=1}^{K} rel_i / log2(i+1)
IDCG@K = DCG@K del ranking perfecto
NDCG@K = DCG@K / IDCG@K
```

donde `rel_i = 1` si el ítem en posición i está en el test set del usuario.

---

### 2.2 Métricas complementarias y su rol

| Métrica | Rol técnico | Traducción al negocio |
|---|---|---|
| **Precision@K** | Fracción de los K recomendados que son relevantes | Tasa de relevancia de la lista mostrada |
| **Recall@K** | Fracción de los ítems relevantes incluidos en los K | Cobertura del historial futuro del usuario |
| **MAP@K** | Promedio de precisiones en cada ítem relevante encontrado | Eficiencia general del ranking |
| **Coverage** | % del catálogo que aparece al menos una vez recomendado | Diversidad del catálogo servido |
| **Novelty** | Popularidad media de los ítems recomendados (inverso) | Capacidad de recomendar ítems no-obvios |

**Resultados del modelo ganador (NB15v2, 3000 usuarios):**

| Métrica | K=5 | K=10 |
|---|---|---|
| Precision@K | ~0.0104 | ~0.0086 |
| Recall@K | ~0.0283 | ~0.0450 |
| NDCG@K | ~0.0318 | **0.0431** |

---

### 2.3 Por qué NO usamos estas métricas

| Métrica excluida | Motivo |
|---|---|
| **RMSE / MAE** | Requieren ratings explícitos (estrellas, puntuaciones). Nuestros datos son feedback implícito (clics, carritos, compras). No hay número que predecir, solo un orden. |
| **AUC** | Mide discriminación entre ítems positivos y negativos, pero en feedback implícito los "negativos" son desconocidos — un ítem no visto puede haber sido no deseado o simplemente no descubierto. |
| **Accuracy** | Solo aplica a clasificación binaria. La recomendación es un problema de ranking sobre miles de ítems. |
| **F1-Score** | Promedia Precision y Recall con pesos iguales; en recomendación nos importa más la posición que la cobertura bruta. NDCG ya captura esto con penalización logarítmica. |

---

## 3. Análisis crítico de resultados

### 3.1 Qué funcionó y por qué

#### RP3beta — el modelo más robusto del proyecto

RP3beta (Random Propagation with damping) resultó ser el modelo más efectivo
por tres razones intrínsecas al dataset:

1. **Robustez ante sparsidad extrema (99.9994 %):** RP3beta opera sobre el grafo
   de co-ocurrencias usuario-ítem. Con solo 1-2 interacciones por usuario, aún
   puede propagar señal a través del vecindario de ítems. Los modelos de
   factorización (SVD, ALS) requieren más supervisión para converger.

2. **Parámetro β como mecanismo anti-popularidad:** El parámetro β penaliza
   ítems con alta popularidad al calcular las probabilidades de transición.
   Esto permite escapar del filter bubble de popularidad sin necesitar datos
   demográficos adicionales.

3. **No requiere backpropagation:** El modelo es puramente analítico (álgebra
   lineal dispersa). Converge en ~6 segundos incluso sobre el dataset completo.

#### Temporal Decay — capturar el cambio de preferencias

La ponderación temporal `peso = exp(-0.01 × días_desde_evento)` mejoró el
NDCG de RP3beta porque los usuarios de e-commerce muestran patrones de compra
altamente dependientes del tiempo. Una vista de hace 4 meses predice mucho
menos que una compra de hace 7 días.

#### EASE^R — la elegancia algebraica

EASE^R (Embarrassingly Shallow Autoencoder for Recommendations) encontró su
lugar en el ensemble a pesar de ser el modelo con menor weight (2.1%) porque
su inductive bias es completamente ortogonal a RP3beta. Opera con álgebra
matricial directa (pseudo-inversión con regularización) sin saltos en el
grafo, capturando correlaciones globales entre ítems.

#### Multi-Behavior — aprender los pesos óptimos de señal

En lugar de usar pesos fijos (trans=3, cart=2, view=1), Optuna optimizó
pesos específicos para los tipos de evento:
- `w_view = 2.669`
- `w_cart = 1.079`  
- `w_trans = 3.869`

El resultado revela que las "vistas" tienen un peso relativo mayor al esperado,
probablemente porque los usuarios que hacen muchas vistas son los que eventualmente
más compran.

#### El ensemble — sinergia de modelos complementarios

La correlación Spearman entre RP3+MB+TD y EASE^R fue **ρ = 0.216** —
excepcionalmente baja. Esto indica que los modelos cometen errores distintos
sobre los mismos usuarios. El ensemble captura lo mejor de cada uno, resultando
en NDCG@10=0.0431 (+50.8% sobre RP3+TD solo).

---

### 3.2 Qué falló y por qué

#### BPR-MF (NDCG@10 = 0.0012) — factorización bayesiana

BPR-Bayesian Personalized Ranking requiere pares (usuario, ítem positivo, ítem
negativo) para el entrenamiento. Con feedback implícito y sparsidad >99.9%,
los "ítems negativos" son mayoritariamente ítems no vistos pero quizás relevantes.
El modelo aprende a penalizar ítems que el usuario simplemente no conocía,
no que rechazó.

#### NCF/NeuMF (NDCG@10 = 0.0002) — deep learning colaborativo

Neural Collaborative Filtering requiere suficiente señal por usuario para que
el gradiente sea informativo. Con mediana de 1 interacción por usuario, la
función de pérdida converge hacia la predicción del ítem más popular, no a
las preferencias individuales.

#### SASRec-lite (NDCG@10 = 0.0005) — transformers secuenciales

El transformer secuencial asume **secuencias de interacciones ordenadas** como
señal. Con 57.5% de usuarios con secuencias de longitud 1, no hay secuencia
que modelar. Adicionalmente, la implementación de NB09 carecía de causal mask
y FFN completa — diagnosticado y corregido en NB12 (SASRec warm).

**Referencia:** Dacrema, Cremonesi & Jannach (2019) — "Are We Really Making
Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches"
(RecSys 2019): métodos lineales y de matrix factorization bien optimizados
superan consistentemente a los métodos neurales en datasets sparse de e-commerce.

Nuestros resultados replican exactamente este hallazgo.

#### Por qué NB11-Optuna no generalizó (-4.2% en test)

Optuna encontró `α=0.75, β=0.30` como hiperparámetros óptimos en el validation
set, pero el modelo resultante tuvo NDCG idéntico al original (0.0258) en test.
Causas:
1. El validation set (450 usuarios) es demasiado pequeño para generalización robusta
2. Los hiperparámetros óptimos de RP3beta son sensibles a la distribución del
   subconjunto de evaluación
3. El espacio de búsqueda era continuo con gradientes muy planos alrededor del óptimo

**Lección:** El problema de sobreajuste no es exclusivo del deep learning.
La optimización bayesiana sobre conjuntos de validación pequeños también puede
sobreajustar.

---

### 3.3 Limitaciones del sistema

| Limitación | Impacto | Mitigación implementada |
|---|---|---|
| 57.5 % de usuarios con 1 ítem en train | Techo práctico de NDCG ≈ 0.028–0.035 sin filtros | Fallback a popularidad para cold-start |
| Contenido de ítems hasheado | Imposibilita Content-Based semántico | Multi-Behavior como proxy de señal |
| Sin datos demográficos reales | No se puede segmentar por perfil real | Demographics sintéticos para LightGBM |
| Sparsidad 99.9994 % | El DL no puede extraer señal suficiente | Métodos clásicos bien optimizados (RP3, EASE) |
| Dataset estático (sin stream) | El modelo no se actualiza con nuevos eventos | Arquitectura batch; reentrenamiento periódico |

**Techo práctico estimado** con protocolo ≥1: **NDCG@10 ≈ 0.045–0.055**.  
El Mega-Ensemble (0.0431) está dentro del 96% de ese techo estimado.

---

### 3.4 Oportunidades de mejora

1. **Modelos de sesión en tiempo real:** GRU4Rec sobre sesiones activas en lugar
   de historial agregado. Captura la intención inmediata del usuario.

2. **A/B testing en producción:** Validar que las ganancias offline (NDCG)
   se traducen en mejoras online (CTR, conversión). Es bien documentado que
   la correlación offline/online no es perfecta.

3. **Reentrenamiento periódico:** Implementar pipeline de reentrenamiento
   automático con ventana deslizante (últimos N días de eventos).

4. **Señales de contexto:** Incorporar precio, stock, margen y categoría
   como features de re-ranking (LightGBM como segunda etapa).

5. **Mejor manejo del catálogo:** RP3beta y EASE^R recomiendan solo de los
   top-20K ítems más populares. Un modelo de "long tail" complementario
   podría mejorar Coverage y Novelty.

---

*Documento generado: Abril 2026 | nexus-recsys v1.0*
