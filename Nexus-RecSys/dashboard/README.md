# nexus-recsys Dashboard — Documentación

**Sistema de Recomendación de Productos — Nexus Data Co.**  
**Modelo:** Mega-Ensemble NB15v2 · NDCG@10 = **0.04310**

---

## Cómo ejecutar

```bash
# Desde la raíz del proyecto
streamlit run dashboard/app.py

# El dashboard se abre automáticamente en:
# http://localhost:8501
```

**Requisitos previos:**
```bash
pip install streamlit plotly pandas numpy scipy
```

---

## Páginas del dashboard

### 🏠 Página 1 — Vista General del Sistema
- KPIs del sistema: NDCG@10=0.0431, mejora vs baseline, escala del dataset
- Gráfico de evolución de modelos (baseline → campeón) con barras coloreadas
- Tarjetas de contexto: 2.75M eventos, 1.4M usuarios, 235K ítems
- Información del modelo ganador: pesos de cada componente del ensemble

### 🎯 Página 2 — Demo de Recomendaciones en Vivo
- Selector de usuario (manual o aleatorio del conjunto de evaluación)
- Controles: cantidad de recomendaciones (5–20), excluir vistos
- Diagnóstico automático del tipo de usuario (warm/cold/semi-warm)
- Tabla interactiva de recomendaciones con scores normalizados
- Historial de ítems vistos en train
- Fallback automático a popularidad para usuarios cold-start

### 📊 Página 3 — Comparativa de Modelos
- Tabla completa con todos los modelos evaluados (NB07–NB15)
- Gráfico comparativo de progresión NDCG@10
- Explicación de por qué el ensemble supera a los individuales

### 🔍 Página 4 — Análisis del Dataset
- Distribución de eventos por tipo (pie chart)
- Funnel de conversión: view → cart → transaction
- Histograma de interacciones por usuario (power law)
- Tabla de sensibilidad al protocolo de evaluación (≥1, ≥3, ≥5)
- Estadísticas de sparsidad y estructura del dataset

### 📈 Página 5 — Métricas del Modelo
- NDCG@10 por segmento de usuario (cold, semi-warm, warm)
- Pesos optimizados del ensemble (gráfico de torta)
- Matriz de correlación Spearman entre componentes
- Precision@K y Recall@K para K=5, 10, 20
- Explicación de cada métrica en términos de negocio

---

## Requisitos de datos

El dashboard funciona con los archivos en `data/processed/`:

| Archivo | Requerido | Descripción |
|---|---|---|
| `model_comparison_final.csv` | ✅ Sí | Tabla de modelos para comparativa |
| `interaction_matrix.csv` | Solo para Demo en Vivo | Datos de usuarios para inferencia |
| `model_comparison_nb*.csv` | Opcional | Para tabla detallada de todos los modelos |

El score cache en `scripts/_score_cache/*.npy` es necesario para la
**Página 2 (Demo en Vivo)**. Las demás páginas funcionan sin él.

---

## Uso sin datos de inferencia

Si no está disponible `interaction_matrix.csv` o el score cache, el dashboard
funciona en modo visualización: todas las páginas excepto la Demo en Vivo
mostrarán datos estáticos del proyecto.

La página de Demo mostrará un mensaje de error con instrucciones para
generar los datos necesarios.

---

## Performance

- La primera carga del modelo (~60s) carga `interaction_matrix.csv` en memoria.
- Las visitas siguientes son instantáneas gracias a `@st.cache_resource`.
- En hardware con 16 GB RAM, el dashboard ocupa ~2-3 GB en memoria.

---

*Documentación: dashboard/README.md | nexus-recsys v1.0 | Abril 2026*
