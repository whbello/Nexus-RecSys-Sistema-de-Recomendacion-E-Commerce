# nexus-recsys API — Documentación de uso

**Sistema de Recomendación de Productos — Nexus Data Co.**  
**Modelo:** Mega-Ensemble NB15v2 (RP3+MB+TD · EASE^R · RP3+TD)  
**NDCG@10 = 0.04310**

---

## Cómo ejecutar

```bash
# Desde la raíz del proyecto
uvicorn api.main:app --reload --port 8000

# Documentación interactiva (Swagger UI)
http://localhost:8000/docs

# Documentación alternativa (ReDoc)
http://localhost:8000/redoc
```

**Requisitos previos:**
1. Tener los archivos de score cache en `scripts/_score_cache/*.npy`
2. Tener `data/processed/interaction_matrix.csv`
3. `pip install fastapi uvicorn pydantic pandas numpy scipy`

---

## Endpoints disponibles

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Estado del sistema y métricas del modelo |
| `POST` | `/recommend` | Recomendaciones personalizadas (body JSON) |
| `GET` | `/recommend/{visitor_id}` | Recomendaciones (query params) |
| `GET` | `/items/{item_id}/similar` | Ítems similares a uno dado |
| `GET` | `/model/metrics` | Métricas y descripción del ensemble |
| `GET` | `/model/predict/cold-start/{visitor_id}` | Fallback explícito por popularidad |
| `GET` | `/usuarios/info/{visitor_id}` | Información de un usuario |

---

## Ejemplos de uso

### 1. Verificar estado del sistema

```bash
curl http://localhost:8000/
```

**Respuesta esperada:**
```json
{
  "status": "ok",
  "model": "Mega-Ensemble NB15v2 (rp3_mb_td · ease_500 · rp3_td)",
  "ndcg10": 0.0431,
  "version": "1.0.0",
  "usuarios_warm_servidos": 2550,
  "items_catalogo": 20000,
  "sistema_listo": true
}
```

---

### 2. Recomendaciones para un usuario warm (GET)

```bash
curl "http://localhost:8000/recommend/1150086?top_n=10&exclude_seen=true"
```

**Respuesta esperada:**
```json
{
  "visitor_id": 1150086,
  "recommendations": [
    {"item_id": 461686, "score": 1.0, "rank": 1, "popularidad": 523.0},
    {"item_id": 135397, "score": 0.87, "rank": 2, "popularidad": 441.0},
    ...
  ],
  "model_used": "Mega-Ensemble (rp3_mb_td·0.956 + rp3_td·0.023 + ease_500·0.021)",
  "ndcg10_model": 0.0431,
  "tipo_usuario": "warm",
  "n_interacciones_train": 7,
  "timestamp": "2026-04-05T10:30:00+00:00"
}
```

---

### 3. Recomendaciones via POST (body JSON)

```bash
curl -X POST "http://localhost:8000/recommend" \
     -H "Content-Type: application/json" \
     -d '{"visitor_id": 1150086, "top_n": 5, "exclude_seen": true}'
```

---

### 4. Usuario cold-start (sin historial)

```bash
curl "http://localhost:8000/recommend/9999999?top_n=5"
```

**Respuesta:** El campo `tipo_usuario` será `"cold-start"` y las recomendaciones
serán los ítems más populares del catálogo (no personalizadas).

---

### 5. Ítems similares

```bash
curl "http://localhost:8000/items/461686/similar?top_n=5"
```

---

### 6. Métricas del modelo

```bash
curl "http://localhost:8000/model/metrics"
```

**Respuesta:**
```json
{
  "modelo": "Mega-Ensemble NB15v2",
  "componentes": {
    "rp3_mb_td": {"peso_ensemble": 0.9556, "ndcg10_individual_test": 0.01890},
    "rp3_td":    {"peso_ensemble": 0.0231, "ndcg10_individual_test": 0.02859},
    "ease_500":  {"peso_ensemble": 0.0213, "ndcg10_individual_test": 0.01930}
  },
  "ensemble_ndcg10_test": 0.04310,
  "mejora_vs_baseline": "+50.8%"
}
```

---

### 7. Información de un usuario

```bash
curl "http://localhost:8000/usuarios/info/1150086"
```

---

## Parámetros y validaciones

### `POST /recommend` — body

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `visitor_id` | int | requerido | ID del usuario |
| `top_n` | int | 10 | Ítems a recomendar (1–50) |
| `exclude_seen` | bool | true | Excluir ítems ya vistos |

### Códigos de respuesta

| Código | Significado |
|---|---|
| `200` | Éxito |
| `404` | Usuario o ítem no encontrado en el catálogo |
| `422` | Parámetros inválidos (ej: top_n > 50) |
| `503` | Modelos aún cargando al inicio |

---

## Casos de uso cubiertos

| Caso | Endpoint | Comportamiento |
|---|---|---|
| Usuario warm (≥1 interacción) | `/recommend/{id}` | Ensemble personalizado |
| Usuario cold-start (nuevo) | `/recommend/{id}` | Fallback popularidad |
| Cold-start explícito | `/model/predict/cold-start/{id}` | Siempre popularidad |
| Ítems similares | `/items/{id}/similar` | Por co-popularidad |
| Consultar métricas | `/model/metrics` | Detalle del ensemble |
| Diagnóstico de usuario | `/usuarios/info/{id}` | Tipo + historial |

---

## Limitaciones conocidas

1. **Cobertura de personalización:** Solo están disponibles recomendaciones
   personalizadas para los ~2550 usuarios del conjunto de evaluación de NB15v2
   (seleccionados con seed=42). Para todos los demás usuarios, se usa el
   fallback de popularidad.

2. **Catálogo activo:** Solo los 20,000 ítems más populares del catálogo
   participan en el ensemble. Ítems del "long tail" no se recomiendan.

3. **Sin actualización en tiempo real:** El modelo es estático (batch). Las
   interacciones que ocurran después del corte (2015-08-22) no se incorporan.

4. **Tiempo de inicio:** La API tarda ~30-60 segundos en estar lista porque
   carga `interaction_matrix.csv` (~2 GB) en memoria al arrancar.

---

*Documentación: api/README.md | nexus-recsys v1.0 | Abril 2026*
