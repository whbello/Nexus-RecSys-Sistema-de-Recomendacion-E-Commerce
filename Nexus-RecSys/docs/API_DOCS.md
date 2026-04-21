# API Documentation — nexus-recsys

> **Base URL:** `http://localhost:8000`  
> **Docs interactivas (Swagger):** `http://localhost:8000/docs`  
> **Redoc:** `http://localhost:8000/redoc`  
> **Versión:** 2.0.0 · Modelo: Mega-Ensemble NB15v2 · NDCG@10 = 0.0431

---

## Inicio rápido

```bash
# Levantar la API
uvicorn api.main:app --reload --port 8000

# Verificar que está corriendo
curl http://localhost:8000/
```

---

## Endpoints

### 🟢 `GET /` — Health Check

Devuelve el estado del sistema y si los modelos están cargados.

**Response:**
```json
{
  "status": "ok",
  "modelo": "Mega-Ensemble NB15v2",
  "ndcg10": 0.0431,
  "warm_users": 3000,
  "items_activos": 20000,
  "modelo_cargado": true,
  "error_carga": null
}
```

---

### 🔵 `POST /recommend` — Recomendar (body JSON)

Genera recomendaciones para un usuario.

**Request body:**
```json
{
  "visitor_id": 442053,
  "top_n": 10,
  "exclude_seen": true
}
```

| Campo | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| `visitor_id` | int | requerido | ID del usuario |
| `top_n` | int | `10` | Cantidad de recomendaciones (1–100) |
| `exclude_seen` | bool | `true` | Excluir ítems vistos en train |

**Response `200 OK`:**
```json
{
  "visitor_id": 442053,
  "top_n": 10,
  "metodo": "ensemble",
  "n_interacciones_train": 7,
  "recomendaciones": [
    {
      "rank": 1,
      "item_id": 187946,
      "score": 1.0,
      "popularidad": 2912.0
    },
    ...
  ],
  "latencia_ms": 3.2
}
```

| Campo `metodo` | Cuándo aplica |
|----------------|---------------|
| `"ensemble"` | Usuario warm (presente en score cache) |
| `"cold-start"` | Usuario nuevo o sin score cache |

---

### 🔵 `GET /recommend/{visitor_id}` — Recomendar (URL)

Misma funcionalidad que `POST /recommend` pero via GET con query params.

**URL:** `/recommend/442053?top_n=10&exclude_seen=true`

**Parámetros:**
| Param | Tipo | Default |
|-------|------|---------|
| `visitor_id` | int (path) | requerido |
| `top_n` | int (query) | `10` |
| `exclude_seen` | bool (query) | `true` |

**Response:** Igual que `POST /recommend`.

**Ejemplo:**
```bash
curl "http://localhost:8000/recommend/442053?top_n=5&exclude_seen=true"
```

---

### 🟡 `GET /items/{item_id}/similar` — Ítems similares

Devuelve los ítems más frecuentemente co-vistos con el ítem dado.

**URL:** `/items/187946/similar?top_n=10`

**Response `200 OK`:**
```json
{
  "item_id": 187946,
  "top_n": 10,
  "similares": [
    {"item_id": 5411, "coocurrencias": 312, "score": 1.0},
    ...
  ],
  "latencia_ms": 15.4
}
```

> **Nota:** El cálculo de similaridad usa co-ocurrencia en transacciones del conjunto de entrenamiento.

---

### 🟣 `GET /model/metrics` — Métricas del modelo

Devuelve las métricas de evaluación del champion.

**Response `200 OK`:**
```json
{
  "modelo": "Mega-Ensemble NB15v2",
  "ndcg10_test": 0.0431,
  "ndcg10_baseline": 0.0286,
  "delta_pct": 50.76,
  "precision10": 0.0412,
  "recall10": 0.0387,
  "map10": 0.0298,
  "hr10": 0.3124,
  "n_usuarios_eval": 3000,
  "protocolo": "warm users (≥1 interacción en train), seed=42",
  "ensemble_weights": {
    "rp3_mb_td": 0.9556,
    "rp3_td": 0.0231,
    "ease_500": 0.0213
  }
}
```

---

### ❄️ `GET /model/predict/cold-start/{visitor_id}` — Cold-start explícito

Fuerza recomendaciones por popularidad global, independientemente de si el usuario es warm.

**URL:** `/model/predict/cold-start/999999?top_n=10`

**Response `200 OK`:**
```json
{
  "visitor_id": 999999,
  "metodo": "cold-start",
  "recomendaciones": [...],
  "latencia_ms": 0.8
}
```

---

### 👤 `GET /usuarios/info/{visitor_id}` — Info del usuario

Devuelve el perfil de un usuario: tipo (warm/cold), historial de ítems en train.

**URL:** `/usuarios/info/442053`

**Response `200 OK`:**
```json
{
  "visitor_id": 442053,
  "tipo_usuario": "warm",
  "n_interacciones_train": 7,
  "items_train": [187946, 5411, 370653],
  "en_eval_users": true,
  "row_en_score_cache": 142
}
```

---

## Códigos de error

| Código | Descripción |
|--------|-------------|
| `503 Service Unavailable` | Modelos no cargados aún (primeros segundos al iniciar) |
| `500 Internal Server Error` | Error durante la carga de los modelos (ver `error_carga` en `/`) |
| `422 Unprocessable Entity` | Parámetros inválidos (ej: `top_n < 1`) |

---

## Notas de rendimiento

| Operación | Latencia (CPU) |
|-----------|----------------|
| Startup (carga de modelos) | ~60–90 segundos |
| `/recommend` (usuario warm) | < 5 ms |
| `/recommend` (cold-start) | < 2 ms |
| `/items/{id}/similar` | 10–30 ms |
| `/model/metrics` | < 1 ms |

**RAM requerida:** ~800 MB para la interaction matrix + ~460 MB para el score cache (3000×20000×fp32).

---

## Uso desde Python

```python
import requests

BASE = "http://localhost:8000"

# Verificar estado
r = requests.get(f"{BASE}/")
print(r.json()["ndcg10"])  # 0.0431

# Recomendaciones para un usuario
r = requests.post(f"{BASE}/recommend", json={
    "visitor_id": 442053,
    "top_n": 10,
    "exclude_seen": True,
})
recs = r.json()["recomendaciones"]
for rec in recs:
    print(f"#{rec['rank']} item_id={rec['item_id']} score={rec['score']:.3f}")
```

---

## Variables de entorno relevantes

Ver [`.env.example`](../.env.example) para la configuración completa.

```bash
# No se requieren variables de entorno para la API base.
# La API carga los datos automáticamente desde data/processed/.
```
