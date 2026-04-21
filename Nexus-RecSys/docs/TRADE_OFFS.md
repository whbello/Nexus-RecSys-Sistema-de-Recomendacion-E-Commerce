# Decisiones técnicas y trade-offs — nexus-recsys

Este documento justifica las decisiones de diseño del sistema Nexus RecSys.
Para cada decisión se explicita la alternativa considerada, la razón de la elección
y el costo asumido.

---

## 1. ¿Por qué RP3beta y no SASRec?

| | RP3beta (elegido) | SASRec (alternativa) |
|---|---|---|
| Complejidad | O(n·k²) analítico | O(L²·d) por epoch |
| Tiempo de entrenamiento | 6 segundos | 45 min / epoch |
| GPU requerida | No | Sí (recomendado) |
| Mínimo de interacciones | 1 | ≥5 (necesita secuencia) |
| Rendimiento en sparse | ★★★★★ | ★★☆☆☆ |

**Decisión tomada:** RP3beta + Temporal Decay + Multi-Behavior.

**Por qué:** RetailRocket tiene densidad de 0.0006%. El 57.5% de usuarios tiene
1 sola interacción de entrenamiento. SASRec requiere una secuencia de al menos
3-5 ítems para que el mecanismo de attention sea útil — con 1 ítem degenera
a popularidad ponderada. Dacrema et al. (2019) documenta formalmente que métodos
clásicos bien ajustados superan al DL en datasets de e-commerce con esta densidad.

**Qué perdemos:** SASRec captura orden temporal y efectos de sesión a largo plazo.
En producción con streams de clics en tiempo real, SASRec sería superior para
usuarios con sesiones activas largas (≥10 clics).

**Experimento realizado:** NB12 implementó SASRec completo (Transformer 2L, 128 heads).
Con warm users (≥5 interacciones), NDCG = 0.0431 (mismo que ensemble global).
El conjunto total de usuarios baja a NDCG ≈ 0.015 por el manejo deficiente de cold users.

---

## 2. ¿Por qué ensemble y no el mejor modelo individual?

| Modelo | NDCG@10 | Ventaja | Debilidad |
|---|---|---|---|
| RP3beta+MB+TD | 0.0411 | Maneja cold-start | Sesgado por popularidad local |
| EASE^R (λ=500) | 0.0382 | Captura co-ocurrencias globales | Sin decay temporal |
| RP3beta+TD | 0.0368 | Rápido y estable | Menos expresivo |
| **Ensemble NB15v2** | **0.0431** | **Lo mejor de cada uno** | Menos interpretable |

**Decisión tomada:** Combinación softmax-weighted con Optuna (100 trials, seed=42).

**Por qué:** La correlación Spearman inter-modelo es solo ρ=0.216 entre RP3 y EASE^R,
indicando que los modelos cometen errores distintos e independientes. Cuando uno
falla, el otro suele acertar. Con diversidad alta, el ensemble captura señal
que ningún modelo individual puede ver solo.

**Qué perdemos:** Interpretabilidad directa (no podemos decir "este ítem fue recomendado
porque…" de forma sencilla). Mayor latencia de inferencia (3 lookups vs 1).
Peor desempeño si los modelos están altamente correlacionados (no es el caso aquí).

**Alternativas rechazadas:**
- Stacking con LightGBM como meta-learner: overfitting al validation set con 3000 usuarios
- Rank fusion (CombSUM/CombMNZ): subóptimo, no aprende los pesos desde los datos
- Bagging de RP3beta: no suficiente diversidad (mismo architecture)

---

## 3. ¿Por qué split temporal y no aleatorio?

**Decisión tomada:** Corte temporal en 2015-08-22 — el 82.2% de las interacciones
(hasta esa fecha) como train y el 17.8% posterior como test.

**Alternativa rechazada:** K-fold cross-validation aleatorio estándar.

**Por qué el temporal es correcto:** Un sistema de recomendación en producción
siempre predice el futuro a partir del pasado. El split aleatorio contamina
el entrenamiento con interacciones del futuro, creando data leakage. Un modelo
que "ve" ítems de octubre 2015 para predecir qué comprará en julio 2015
tiene ventaja artificial — en producción ese conocimiento no existiría.

**Evidencia concreta:** En el NB04 de análisis exploratorio, identificamos
que las semanas finales del dataset tienen distribución de ítems levemente
diferente (nuevos productos lanzados). Un split aleatorio haría que el evaluador
no detecte drift temporal.

**Qué perdemos:** El split temporal usa cada ejemplo solo una vez (no podemos
hacer k-fold para estabilizar la varianza). Con k=5 aleatorio se reduciría
el error estándar de la métrica. Asumimos que un único corte es suficiente
dado la volumetría (381K eventos en test).

---

## 4. ¿Por qué NDCG y no Precision como métrica principal?

| Métrica | Qué mide | Cuándo usar |
|---|---|---|
| Precision@K | % de recomendados relevantes entre los K | Cuando el orden no importa |
| Recall@K | % de relevantes que están entre los K | Cuando el catálogo es pequeño |
| **NDCG@K** | Calidad del ranking (posición importa) | **Cuando el orden sí importa** |
| MAP | Precision promedio por rango de recall | Cuando hay múltiples elementos relevantes |
| RMSE | Error en predicción de rating | Solo con feedback explícito |

**Decisión tomada:** NDCG@10 como métrica primaria, con Precision@10, Recall@10 y
Coverage@10 como métricas secundarias evaluadas en NB08.

**Por qué NDCG:** En un carrusel de productos, el ítem en posición 1 tiene CTR
3-5× mayor que el de posición 10. Precision@K trata todas las posiciones igualmente —
no captura que el orden es crucial. NDCG descuenta logarítmicamente la relevancia
por posición: si el producto correcto está en posición 1 vale más que en posición 9.
Este DCG normalizado es el estándar en todos los sistemas de ranking modernos.

**Por qué no RMSE:** No tenemos ratings explícitos. Solo tenemos señal binaria
(interactuó / no interactuó). RMSE requeriría conocer el "valor real" de cada
usuario-ítem, que no existe en feedback implícito. Usar RMSE sobre datos binarios
introduce sesgo de popularidad severo y mide algo sin significado práctico.

---

## 5. ¿Por qué λ=500 en EASE^R y no el valor óptimo de Optuna?

**Contexto:** EASE^R tiene un único hiperparámetro regularizador λ (L2 sobre
la diagonal de la matriz de ítem embeddings). Optuna en NB11 sugirió λ=500
como óptimo con NDCG@10 = 0.0382 en validation.

**Evaluación de alternativas:**

| λ | NDCG@10 (val) | NDCG@10 (test) |
|---|---|---|
| 50 | 0.0301 | 0.0298 |
| 200 | 0.0371 | 0.0368 |
| **500** | **0.0382** | **0.0379** |
| 1000 | 0.0377 | 0.0374 |
| 3000 | 0.0341 | 0.0339 |

**Decisión tomada:** λ=500 (score óptimo en ambos splits).

**Por qué no λ=3000 o mayor:** Con λ demasiado alto, la regularización aplasta
los ítems poco populares y EASE^R degenera a recomendar siempre ítems populares.
λ=500 equilibra la captura de co-ocurrencias reales con evitar el overfitting.

**Qué perdemos:** λ=200 da mejor cobertura de catálogo (más diversidad) pero
peor NDCG. Para un objetivo de negocio de diversificación, λ=200 podría ser
preferible. Para maximizar relevancia individual, λ=500 es correcto.

---

## 6. ¿Por qué decay_rate=0.01 fijo?

**Fórmula:** `peso_evento = exp(−decay_rate × días_desde_evento)`

**Contexto:** Con decay_rate=0.01, un evento de hace 100 días vale `exp(−1)=0.368`
comparado con un evento de hoy (1.0). A 300 días vale `exp(−3)=0.050`.

**Evaluación de alternativas (NB13, Optuna 100 trials):**

| decay_rate | NDCG@10 (val) | Interpretación |
|---|---|---|
| 0.001 | 0.0392 | Casi sin decay (pasado vale igual que presente) |
| **0.01** | **0.0411** | Decay suave — eventos <200 días aún influyen |
| 0.05 | 0.0391 | Decay agresivo — solo últimas 4-5 semanas importan |
| 0.10 | 0.0378 | Solo últimas 2 semanas (demasiado agresivo) |

**Decisión tomada:** decay_rate=0.01 (óptimo de Optuna).

**Por qué:** El dataset cubre 135 días de actividad (jun-oct 2015). Con 0.01,
los eventos de más de 160 días tienen peso <0.20 (casi irrelevantes), pero
los de últimas 4 semanas retienen >75% de su peso. Captura la zona de mayor
predictividad sin descartar historia útil.

**Qué perdemos:** Un decay_rate adaptativo por categoría de producto podría
ser mejor (electrónica: decay rápido; libros: decay lento). Pero añadir ese
nivel de complejidad no estaba justificado por los datos disponibles.

---

## 7. ¿Por qué evaluar con ≥1 y no filtrar cold users?

**Decisión tomada:** Evaluar sobre usuarios con ≥1 interacción en train,
manteniendo también a los "semi-cold" (1-4 interacciones).

**Alternativa rechazada:** Filtrar a usuarios con ≥5 interacciones (protocolo
estándar de muchos papers).

**Por qué:** El protocolo honesto incluye a todos los usuarios que el sistema
deberá servir en producción. Si filtramos a ≥5 interacciones, excluimos el
57.5% de usuarios — el sistema real los recibiría igualmente y fallaría. Publicar
NDCG=0.079 (con filtro ≥5) cuando en producción sería NDCG=0.043 es deshonesto.
Preferimos la métrica dura pero real.

**Costo asumido:** Nuestro NDCG parece bajo comparado con papers que usan
filtros agresivos. Para contextualizar, siempre citamos las dos métricas:
NDCG@10 (all users) = 0.0431 y NDCG@10 (≥5 interactions) ≈ 0.079.

**Validación:** Esta decisión está alineada con las recomendaciones de
Dacrema et al. (2019) "Are We Really Making Much Progress?" que denuncia
precisamente el uso de filtros que inflan artificialmente las métricas.

---

## Resumen de trade-offs

| Dimensión | Elegimos | Sacrificamos |
|---|---|---|
| Arquitectura | Robustez (RP3) | Expresividad (DL) |
| Complejidad | Interpretabilidad parcial | Flexibilidad de ensemble |
| Evaluación | Honestidad (all users) | Métricas altas de paper |
| Split | Validez temporal | Estabilidad de K-fold |
| Regularización | Precisión (NDCG-max) | Diversidad de catálogo |
| Decay | Configurabilidad simple | Adaptación por categoría |

---

*Documento generado para nexus-recsys · Henry DS Bootcamp · Marzo 2026*
