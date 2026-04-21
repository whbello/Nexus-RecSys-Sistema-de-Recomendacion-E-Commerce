"""
generate_10_notebook.py
=======================
Genera notebooks/10_multivae.ipynb

Modelo implementado:
  Mult-VAE^PR (Variational Autoencoder para Recomendacion con Multinomial Likelihood)
  Referencia: Liang et al., "Variational Autoencoders for Collaborative Filtering", WWW 2018.

Este notebook compite directamente con RP3beta (NDCG@10=0.0258) usando:
  - Bag-of-items como input (no secuencial) -> adecuado para RetailRocket
  - Multinomial likelihood (correcto para feedback implicito)
  - KL annealing (robusto ante sparsity extrema)
  - Implementacion PyTorch pura (sin frameworks externos)
"""
import json, pathlib, uuid

ROOT    = pathlib.Path(__file__).parent.parent
NB_PATH = ROOT / "notebooks" / "10_multivae.ipynb"


def _cell_id():
    return uuid.uuid4().hex[:8]


def md(src):
    return {"cell_type": "markdown", "id": _cell_id(), "metadata": {}, "source": src}


def py(src):
    return {
        "cell_type": "code", "id": _cell_id(),
        "execution_count": None, "metadata": {}, "outputs": [], "source": src,
    }


cells = []

# =============================================================================
# TITULO
# =============================================================================
cells.append(md("""\
# 10 · Nexus RecSys — Mult-VAE^PR: Variational Autoencoder para Recomendacion

**Sistema de Recomendacion E-Commerce - RetailRocket Dataset**

---

## Objetivo

Superar el ganador actual **RP3beta (NDCG@10=0.0258)** con **Mult-VAE^PR** (Liang et al., WWW 2018).

### ¿Por qué MultiVAE puede superar a RP3beta aquí?

| Aspecto | RP3beta | Mult-VAE^PR |
|---------|---------|-------------|
| Tipo de input | Bag-of-items | Bag-of-items |
| Función de loss | Ninguna (algebraica) | Multinomial ELBO (teoricamente correcta para feedback implicito) |
| KL annealing | No aplica | ✅ Regulariza el espacio latente para sparsity extrema |
| Capacidad de generalización | Linear (propagación de grafos) | No-lineal (redes neuronales) |
| Memoria de ítems populares | Implícita (co-ocurrencia) | Aprendida (espacio latente) |
| Penalización de popularidad | ✅ Explícita (`pop^beta`) | Implícita (via dropout de entrada) |

### Historia del benchmark (NB07–NB09):

| Modelo | NDCG@10 | Mejora acumulada |
|--------|--------:|:----------------:|
| SVD (k=50) — NB07 baseline | 0.0081 | — |
| SVD+TD+IPS — NB08 | 0.0093 | +17% |
| EASE^R — NB09 | 0.0193 | +107% |
| **RP3beta — NB09 (actual ganador)** | **0.0258** | **+176%** |
| **Mult-VAE^PR — NB10 (objetivo)** | **?** | **objetivo: >0.0258** |

> Requiere NB06 + NB07 + NB08 + NB09 ejecutados previamente.
> `scripts/multivae_model.py` debe estar presente en el repo.
> PyTorch 2.10.0+cpu instalado en el venv.\
"""))

# =============================================================================
# 0 - SETUP
# =============================================================================
cells.append(md("## 0 - Setup y Parámetros Globales"))

cells.append(py("""\
import os, sys, time, json, pickle, warnings, logging, math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize as skl_normalize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── PyTorch (requerido para MultiVAE) ────────────────────────────────────────
try:
    import torch
    print(f"PyTorch: {torch.__version__}  device={'GPU' if torch.cuda.is_available() else 'CPU'}")
except ImportError:
    raise ImportError("PyTorch no disponible. Instalá con: pip install torch --index-url https://download.pytorch.org/whl/cpu")

# ── MultiVAE (implementacion propia) ─────────────────────────────────────────
SCRIPTS_DIR = Path().resolve().parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from multivae_model import (
    MultiVAE,
    train_multivae,
    build_scorer,
    run_multivae_pipeline,
    multivae_loss,
)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Parámetros de evaluación (idénticos a NB09 para comparación justa) ───────
RANDOM_STATE  = 42
K_VALUES      = [5, 10, 20]
N_EVAL_USERS  = 3_000
CUTOFF_DATE   = pd.Timestamp("2015-08-22", tz="UTC")
N_DAU         = 50_000
AVG_TICKET    = 45.0

# ── Hiperparámetros MultiVAE (grid buscado en NB10) ──────────────────────────
# Configuración BASE (reproducible, basada en el paper original para
# datasets de e-commerce con sparsity alta)
MVAE_TOP_ITEMS    = 20_000     # mismo subespacio que EASE^R y RP3beta
MVAE_ENC_DIMS     = [600, 200] # paper: [600] para ML-20M; [600,200] para más regularización
MVAE_LATENT_DIM   = 64        # espacio latente (más pequeño = más regularizado)
MVAE_DROPOUT      = 0.5        # dropout de entrada (paper recomienda 0.5)
MVAE_LR           = 1e-3
MVAE_L2_REG       = 1e-5
MVAE_EPOCHS       = 50         # el paper muestra convergencia en 50-200 epochs
MVAE_BATCH        = 512
MVAE_BETA_MAX     = 0.3        # beta_max < 1 es mejor en datasets sparse (partial regularization)
MVAE_ANNEAL_STEPS = 200_000    # pasos de annealing (de 0 a beta_max)
MVAE_MAX_TRAIN_USERS = 150_000 # usuarios de entrenamiento (subconjunto para CPU)
# Nota: 150K usuarios × 50 epochs × batch=512 ≈ 2.5 min/epoch en CPU

HERE     = Path().resolve()
ROOT     = HERE.parent if (HERE.parent / "data").exists() else HERE
DATA_DIR = ROOT / "data" / "processed"
ENC_DIR  = ROOT / "encoders"
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

print(f"Root         : {ROOT}")
print(f"MultiVAE cfg : top={MVAE_TOP_ITEMS}, enc={MVAE_ENC_DIMS}, z={MVAE_LATENT_DIM}")
print(f"               dropout={MVAE_DROPOUT}, lr={MVAE_LR}, epochs={MVAE_EPOCHS}")
print(f"               batch={MVAE_BATCH}, beta_max={MVAE_BETA_MAX}")
print(f"               anneal_steps={MVAE_ANNEAL_STEPS}, max_users={MVAE_MAX_TRAIN_USERS}")
"""))

# =============================================================================
# 1 - CARGA DE DATOS (idéntica a NB09)
# =============================================================================
cells.append(md("## 1 - Carga de Datos y Split Temporal"))

cells.append(py("""\
print("Cargando datos...")
t0 = time.time()
im  = pd.read_csv(DATA_DIR / "interaction_matrix.csv")
im["last_interaction_ts"] = pd.to_datetime(im["last_interaction_ts"], format="ISO8601", utc=True)
print(f"  IM: {im.shape}  [{time.time()-t0:.1f}s]")

train_mask = im["last_interaction_ts"] < CUTOFF_DATE
train_df   = im[train_mask].copy()
test_df    = im[~train_mask].copy()

warm_users = sorted(set(train_df["visitorid"].unique()) & set(test_df["visitorid"].unique()))
rng = np.random.default_rng(RANDOM_STATE)
eval_users = rng.choice(warm_users, size=min(N_EVAL_USERS, len(warm_users)), replace=False).tolist()

test_items_by_user  = test_df.groupby("visitorid")["itemid"].apply(set).to_dict()
train_items_by_user = train_df.groupby("visitorid")["itemid"].apply(set).to_dict()
test_tx_by_user = (
    test_df[test_df["last_interaction_type"] == "transaction"]
    .groupby("visitorid")["itemid"].apply(set).to_dict()
)

all_items_global = sorted(im["itemid"].unique())
n_items_global   = len(all_items_global)

n_test_tx   = len(test_df[test_df["last_interaction_type"] == "transaction"])
baseline_conv = n_test_tx / (len(set(test_df["visitorid"])) * n_items_global)

n_buyers = sum(1 for u in eval_users if test_tx_by_user.get(u))
print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")
print(f"Warm users: {len(warm_users):,}  Eval: {len(eval_users):,}  Compradores: {n_buyers}")
print(f"Baseline P(compra|aleatorio): {baseline_conv:.2e}")
"""))

cells.append(py("""\
# ── Índices y matriz sparse (igual que NB09) ──────────────────────────────────
all_train_users = sorted(train_df["visitorid"].unique())
all_train_items = sorted(train_df["itemid"].unique())
user2idx = {u: i for i, u in enumerate(all_train_users)}
item2idx = {it: i for i, it in enumerate(all_train_items)}
idx2user = {i: u for u, i in user2idx.items()}
idx2item = {i: it for it, i in item2idx.items()}
n_u = len(all_train_users)
n_i = len(all_train_items)

rows = train_df["visitorid"].map(user2idx).values
cols = train_df["itemid"].map(item2idx).values
vals = train_df["interaction_strength"].values.astype(np.float32)

R = sp.csr_matrix((vals, (rows, cols)), shape=(n_u, n_i), dtype=np.float32)
item_pop = np.asarray(R.sum(axis=0)).ravel()
item_pop_dict = {idx2item[i]: float(item_pop[i]) for i in range(n_i)}
n_total_train = float(R.sum())

print(f"R: {R.shape}  nnz={R.nnz:,}")
print(f"Sparsidad: {1 - R.nnz / (n_u * n_i):.6f}")

# Distribución de ítems por usuario
user_lens = np.asarray(R.sum(axis=1)).ravel()
print(f"Ítems/usuario: mean={user_lens.mean():.2f}  median={np.median(user_lens):.0f}  "
      f"p95={np.percentile(user_lens, 95):.0f}  max={user_lens.max():.0f}")
"""))

# =============================================================================
# 2 - BASELINE RP3beta (ganador NB09, para comparación directa)
# =============================================================================
cells.append(md("## 2 - Baseline RP3beta (ganador NB09)"))

cells.append(py("""\
# ── Cargar mappings del NB09 (más rápido que reconstruir) ────────────────────
print("Cargando final_model_v3.pkl (solo mappings)...")
t0 = time.time()
with open(ENC_DIR / "final_model_v3.pkl", "rb") as f:
    fm3 = pickle.load(f)
print(f"  Keys disponibles: {[k for k in fm3 if 'idx' in k or 'top' in k]}")
print(f"  [{time.time()-t0:.1f}s]")

# ── Recomputar W_rp3 (tarda ~6s, reproducible con los mismos datos) ──────────
print("Recomputando W_rp3 (RP3beta)...")
t_rp3 = time.time()
RP3_ALPHA = 0.85
RP3_BETA  = 0.25
RP3_TOP   = 20_000

top_items_rp3 = np.argpartition(item_pop, -RP3_TOP)[-RP3_TOP:]
top_items_rp3 = top_items_rp3[np.argsort(item_pop[top_items_rp3])[::-1]]
top_items_global_rp3 = [idx2item[i] for i in top_items_rp3]

X_rp3 = R[:, top_items_rp3].astype(np.float32).tocsr()  # (n_u, 20K)
P_ui  = skl_normalize(X_rp3, norm="l1", axis=1)          # (n_u, 20K) sparse
P_it  = skl_normalize(X_rp3.T.tocsr(), norm="l1", axis=1)  # (20K, n_u) sparse

# W_rp3 = (P_it @ P_ui)^alpha / pop^beta  — (20K, 20K) dense
W_dense = np.asarray((P_it @ P_ui).todense(), dtype=np.float32)
W_dense = np.power(np.abs(W_dense), RP3_ALPHA) * np.sign(W_dense)
pop_norm = item_pop[top_items_rp3].astype(np.float32)
pop_norm = np.maximum(pop_norm, 1.0)
W_rp3 = W_dense / (pop_norm[None, :] ** RP3_BETA)
del W_dense
print(f"  W_rp3: {W_rp3.shape}  [{time.time()-t_rp3:.1f}s]")

R_rp3 = X_rp3.copy()

def get_rp3(uid, n):
    if uid not in user2idx: return []
    ui = user2idx[uid]
    row = R_rp3.getrow(ui)
    x_u = np.asarray(row.todense(), dtype=np.float32).ravel()
    sc = x_u @ W_rp3
    sc[x_u > 0] = -np.inf
    top_local = np.argpartition(sc, -n)[-n:]
    top_local = top_local[np.argsort(sc[top_local])[::-1]]
    return [top_items_global_rp3[i] for i in top_local]

from multivae_model import _evaluate_compat
print("Evaluando RP3beta (baseline NB09)...")
t1 = time.time()
m_rp3 = _evaluate_compat(
    get_rp3, eval_users, test_items_by_user, test_tx_by_user,
    item_pop_dict, n_total_train, n_items_global, baseline_conv, ks=K_VALUES
)
print(f"  [{time.time()-t1:.1f}s] RP3beta NDCG@10={m_rp3.get('NDCG@10', 0):.4f}"
      f"  Coverage={m_rp3.get('Coverage', 0):.4f}")

all_results = {"RP3beta (NB09)": m_rp3}
"""))

# =============================================================================
# 3 - MultiVAE ENTRENAMIENTO
# =============================================================================
cells.append(md("""\
## 3 - Mult-VAE^PR: Entrenamiento

### Arquitectura implementada

```
Input: x ∈ {0,1}^n_items  (historial binarizado del usuario, tamaño top-20K)
  ↓  L2-norm + Dropout(0.5)
  ↓  Linear(20K, 600) → Tanh
  ↓  Linear(600, 200) → Tanh
  ↓  Linear(200, 64) → μ(z)  |  Linear(200, 64) → log σ²(z)
  ↓  z = μ + ε·σ  (reparameterization trick; en eval: z=μ)
  ↓  Linear(64, 200) → Tanh
  ↓  Linear(200, 600) → Tanh
  ↓  Linear(600, 20K)  →  logits
Output: logits ∈ ℝ^n_items (sin softmax; se aplica en la loss y en scoring)
```

### Función de pérdida ELBO Multinomial

$$\\mathcal{L} = \\underbrace{\\mathbb{E}_q[\\log p(x|z)]}_{\\text{recon: multinomial NLL}} - \\beta \\cdot \\underbrace{KL(q(z|x) \\| p(z))}_{\\text{regularización latente}}$$

donde $\\beta$ sube de 0 a $\\beta_{max}=0.3$ durante el entrenamiento (KL annealing).
$\\beta_{max} < 1$ implementa la "partial regularization" del paper (mejor para datos sparse).\
"""))

cells.append(py("""\
print("=" * 60)
print("Entrenando Mult-VAE^PR ...")
print(f"  top_items={MVAE_TOP_ITEMS}, enc={MVAE_ENC_DIMS}, z={MVAE_LATENT_DIM}")
print(f"  epochs={MVAE_EPOCHS}, batch={MVAE_BATCH}, beta_max={MVAE_BETA_MAX}")
print(f"  max_train_users={MVAE_MAX_TRAIN_USERS}")
print("=" * 60)

t_mvae = time.time()
mvae_model, mvae_history, m_mvae = run_multivae_pipeline(
    R_full=R,
    user2idx=user2idx,
    idx2item=idx2item,
    item_pop=item_pop,
    eval_users=eval_users,
    test_items_by_user=test_items_by_user,
    test_tx_by_user=test_tx_by_user,
    item_pop_dict=item_pop_dict,
    n_total_train=n_total_train,
    n_items_global=n_items_global,
    baseline_conv=baseline_conv,
    top_k_items=MVAE_TOP_ITEMS,
    enc_dims=MVAE_ENC_DIMS,
    latent_dim=MVAE_LATENT_DIM,
    dropout_rate=MVAE_DROPOUT,
    lr=MVAE_LR,
    l2_reg=MVAE_L2_REG,
    n_epochs=MVAE_EPOCHS,
    batch_size=MVAE_BATCH,
    beta_max=MVAE_BETA_MAX,
    total_anneal_steps=MVAE_ANNEAL_STEPS,
    max_train_users=MVAE_MAX_TRAIN_USERS,
    seed=RANDOM_STATE,
    verbose=True,
    ks=K_VALUES,
)

print(f"\\nMultiVAE total time: {time.time()-t_mvae:.1f}s")
print(f"NDCG@10 = {m_mvae.get('NDCG@10', 0):.4f}  (RP3beta baseline = {m_rp3.get('NDCG@10', 0):.4f})")
ndcg_rp3 = m_rp3.get("NDCG@10", 0)
ndcg_mvae = m_mvae.get("NDCG@10", 0)
if ndcg_rp3 > 0:
    delta = (ndcg_mvae - ndcg_rp3) / ndcg_rp3 * 100
    print(f"Delta vs RP3beta: {delta:+.1f}%")
all_results["Mult-VAE^PR"] = m_mvae
"""))

# =============================================================================
# 4 - CURVA DE CONVERGENCIA
# =============================================================================
cells.append(md("## 4 - Curva de Convergencia del VAE"))

cells.append(py("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

epochs_x = range(1, len(mvae_history["loss"]) + 1)

axes[0].plot(epochs_x, mvae_history["loss"], color="steelblue", lw=2)
axes[0].set_title("ELBO Loss (total)")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")

axes[1].plot(epochs_x, mvae_history["recon"], color="darkorange", lw=2, label="Recon (NLL)")
axes[1].plot(epochs_x, mvae_history["kl"],    color="purple", lw=2, linestyle="--", label="KL")
axes[1].set_title("Recon NLL vs KL")
axes[1].set_xlabel("Epoch"); axes[1].legend()

axes[2].plot(epochs_x, mvae_history["beta"], color="green", lw=2)
axes[2].set_title("Beta (KL annealing)")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("beta")
axes[2].axhline(y=MVAE_BETA_MAX, color="red", linestyle="--", label=f"beta_max={MVAE_BETA_MAX}")
axes[2].legend()

plt.suptitle("Mult-VAE^PR — Convergencia de Entrenamiento", fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(DOCS_DIR / "fig_10_multivae_convergence.png", dpi=150, bbox_inches="tight")
plt.show()
print("Guardado: fig_10_multivae_convergence.png")
"""))

# =============================================================================
# 5 - GRIDSEARCH LIGERO (beta_max y latent_dim)
# =============================================================================
cells.append(md("""\
## 5 - Grid Search Ligero (opcional)

Explora 4 configuraciones clave para optimizar el balance recon/KL.
Se ejecuta solo si `RUN_GRID=True` (por defecto False para ahorrar tiempo).\
"""))

cells.append(py("""\
RUN_GRID = False  # Poner True para re-ejecutar el grid

if RUN_GRID:
    grid_configs = [
        # (beta_max, latent_dim, label)
        (0.1,  64,  "beta=0.1, z=64"),
        (0.3,  64,  "beta=0.3, z=64"),  # <- configuración base
        (0.3, 128,  "beta=0.3, z=128"),
        (1.0,  64,  "beta=1.0, z=64 (full KL)"),
    ]

    grid_results = {}
    for beta, z, label in grid_configs:
        print(f"\\n--- Grid: {label} ---")
        _, _, m_g = run_multivae_pipeline(
            R_full=R, user2idx=user2idx, idx2item=idx2item,
            item_pop=item_pop, eval_users=eval_users,
            test_items_by_user=test_items_by_user,
            test_tx_by_user=test_tx_by_user,
            item_pop_dict=item_pop_dict, n_total_train=n_total_train,
            n_items_global=n_items_global, baseline_conv=baseline_conv,
            top_k_items=MVAE_TOP_ITEMS, enc_dims=MVAE_ENC_DIMS,
            latent_dim=z, dropout_rate=MVAE_DROPOUT,
            lr=MVAE_LR, l2_reg=MVAE_L2_REG,
            n_epochs=30,  # epochs reducidos para grid
            batch_size=MVAE_BATCH, beta_max=beta,
            total_anneal_steps=100_000,
            max_train_users=80_000,  # usuarios reducidos para grid
            seed=RANDOM_STATE, verbose=False, ks=[10],
        )
        ndcg = m_g.get("NDCG@10", 0)
        grid_results[label] = ndcg
        print(f"  NDCG@10={ndcg:.4f}")

    print("\\nGrid Search resultados:")
    for label, ndcg in sorted(grid_results.items(), key=lambda x: -x[1]):
        print(f"  {ndcg:.4f}  {label}")
else:
    print("Grid search desactivado (RUN_GRID=False).")
    print(f"Configuración base: beta={MVAE_BETA_MAX}, z={MVAE_LATENT_DIM}  ->  NDCG@10={m_mvae.get('NDCG@10', 0):.4f}")
"""))

# =============================================================================
# 6 - COMPARACION FINAL
# =============================================================================
cells.append(md("## 6 - Tabla Comparativa Final NB07–NB10"))

cells.append(py("""\
# Resultados históricos (de los notebooks anteriores)
historical = {
    "SVD (k=50) — NB07":         {"NDCG@10": 0.00808, "Coverage": 0.0041},
    "SVD+TD+IPS — NB08":         {"NDCG@10": 0.00934, "Coverage": 0.0115},
    "BPR-MF (k=64) — NB09":      {"NDCG@10": 0.00124, "Coverage": 0.0017},
    "EASE^R (λ=500) — NB09":     {"NDCG@10": 0.01931, "Coverage": 0.0496},
    "RP3beta (α=0.85) — NB09":   {"NDCG@10": 0.02576, "Coverage": 0.0600},
    "NCF (NeuMF) — NB09":        {"NDCG@10": 0.00017, "Coverage": 0.0163},
    "SASRec-lite — NB09":        {"NDCG@10": 0.00054, "Coverage": 0.0016},
}

all_hist_results = dict(historical)
all_hist_results["RP3beta (NB09) — reval"]   = m_rp3
all_hist_results["Mult-VAE^PR (NB10)"]       = m_mvae

rows_cmp = []
for name, metrics in all_hist_results.items():
    rows_cmp.append({
        "Modelo":        name,
        "NDCG@10":       metrics.get("NDCG@10", 0),
        "Precision@10":  metrics.get("Precision@10", 0),
        "Coverage":      metrics.get("Coverage", 0),
        "Recall@10":     metrics.get("Recall@10", 0),
    })

df_cmp = pd.DataFrame(rows_cmp).sort_values("NDCG@10", ascending=False).reset_index(drop=True)
df_cmp["Rank"] = df_cmp.index + 1
df_cmp["Mejora vs NB08"] = df_cmp["NDCG@10"].apply(
    lambda x: f"{(x - 0.00934) / 0.00934 * 100:+.1f}%" if x != 0.00934 else "—"
)

print("=" * 80)
print("TABLA COMPARATIVA NB07–NB10")
print("=" * 80)
print(df_cmp[["Rank", "Modelo", "NDCG@10", "Precision@10", "Coverage", "Mejora vs NB08"]].to_string(index=False))
print("=" * 80)
winner = df_cmp.iloc[0]["Modelo"]
winner_ndcg = df_cmp.iloc[0]["NDCG@10"]
rp3_row = df_cmp[df_cmp["Modelo"].str.contains("RP3beta.*reval")].iloc[0]
print(f"\\nGanador actual : {winner}")
print(f"NDCG@10        : {winner_ndcg:.4f}")
print(f"RP3beta NB09   : {rp3_row['NDCG@10']:.4f}")
if winner_ndcg > rp3_row['NDCG@10']:
    delta = (winner_ndcg - rp3_row['NDCG@10']) / rp3_row['NDCG@10'] * 100
    print(f"MultiVAE supera a RP3beta en {delta:+.1f}%  *** NUEVO GANADOR ***")
else:
    delta = (winner_ndcg - rp3_row['NDCG@10']) / rp3_row['NDCG@10'] * 100
    print(f"MultiVAE vs RP3beta: {delta:+.1f}%  (RP3beta mantiene la ventaja)")
"""))

# =============================================================================
# 7 - VISUALIZACION barplot
# =============================================================================
cells.append(md("## 7 - Visualización: NDCG@10 Global NB07–NB10"))

cells.append(py("""\
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ── Barplot NDCG@10 ──────────────────────────────────────────────────────────
colors = []
for m in df_cmp["Modelo"]:
    if "Mult-VAE" in m:            colors.append("crimson")
    elif "RP3beta" in m and "reval" not in m and "NB09" in m: colors.append("steelblue")
    elif "EASE^R" in m:            colors.append("darkorange")
    elif "NB08" in m:              colors.append("gray")
    else:                          colors.append("lightgray")

bars = ax1.barh(df_cmp["Modelo"][::-1], df_cmp["NDCG@10"][::-1], color=colors[::-1])
ax1.set_xlabel("NDCG@10", fontsize=12)
ax1.set_title("Comparativa NDCG@10 (NB07–NB10)", fontweight="bold")
ax1.axvline(x=0.0258, color="steelblue", linestyle="--", alpha=0.7, label="RP3beta (0.0258)")
for bar, val in zip(bars, df_cmp["NDCG@10"][::-1]):
    ax1.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height()/2,
             f"{val:.4f}", va="center", fontsize=9)
ax1.legend(fontsize=9)

# ── Scatter NDCG@10 vs Coverage ─────────────────────────────────────────────
sc_colors = {
    "Mult-VAE^PR (NB10)":       "crimson",
    "RP3beta (NB09) — reval":   "steelblue",
    "EASE^R (λ=500) — NB09":   "darkorange",
    "SVD+TD+IPS — NB08":        "gray",
}
for _, row in df_cmp.iterrows():
    c = sc_colors.get(row["Modelo"], "lightgray")
    ax2.scatter(row["Coverage"], row["NDCG@10"], color=c, s=120, zorder=5)
    ax2.annotate(row["Modelo"].split(" — ")[0],
                 (row["Coverage"], row["NDCG@10"]),
                 textcoords="offset points", xytext=(5, 3), fontsize=8)

ax2.set_xlabel("Coverage", fontsize=12)
ax2.set_ylabel("NDCG@10", fontsize=12)
ax2.set_title("Trade-off NDCG@10 vs Coverage", fontweight="bold")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(DOCS_DIR / "fig_10_multivae_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Guardado: fig_10_multivae_comparison.png")
"""))

# =============================================================================
# 8 - GUARDAR ARTEFACTOS
# =============================================================================
cells.append(md("## 8 - Guardar Artefactos"))

cells.append(py("""\
# ── CSV comparativo NB10 ──────────────────────────────────────────────────────
df_cmp.to_csv(DOCS_DIR / "model_comparison_10_multivae.csv", index=False)
df_cmp.to_csv(DATA_DIR / "model_comparison_10_multivae.csv", index=False)
print("Guardado: model_comparison_10_multivae.csv")

# ── Guardar modelo MultiVAE ───────────────────────────────────────────────────
# Seleccionar los top_items del subespacio MultiVAE
top_idx_mvae = np.argpartition(item_pop, -MVAE_TOP_ITEMS)[-MVAE_TOP_ITEMS:]
top_idx_mvae = top_idx_mvae[np.argsort(item_pop[top_idx_mvae])[::-1]]
top_items_global_mvae = [idx2item[i] for i in top_idx_mvae]

artifact = {
    "model_state":        mvae_model.state_dict(),
    "n_items":            MVAE_TOP_ITEMS,
    "enc_dims":           MVAE_ENC_DIMS,
    "latent_dim":         MVAE_LATENT_DIM,
    "dropout_rate":       MVAE_DROPOUT,
    "top_items_idx":      top_idx_mvae,
    "top_items_global":   top_items_global_mvae,
    "user2idx":           user2idx,
    "item2idx":           item2idx,
    "idx2user":           idx2user,
    "idx2item":           idx2item,
    "training_history":   mvae_history,
    "eval_metrics":       m_mvae,
    "hyperparams": {
        "top_k_items":    MVAE_TOP_ITEMS,
        "enc_dims":       MVAE_ENC_DIMS,
        "latent_dim":     MVAE_LATENT_DIM,
        "dropout_rate":   MVAE_DROPOUT,
        "lr":             MVAE_LR,
        "l2_reg":         MVAE_L2_REG,
        "n_epochs":       MVAE_EPOCHS,
        "batch_size":     MVAE_BATCH,
        "beta_max":       MVAE_BETA_MAX,
        "anneal_steps":   MVAE_ANNEAL_STEPS,
        "max_train_users": MVAE_MAX_TRAIN_USERS,
    },
    "ndcg10":             m_mvae.get("NDCG@10", 0),
    "notebook":           "10_multivae.ipynb",
}

out_path = ENC_DIR / "final_model_v4.pkl"
with open(out_path, "wb") as f:
    pickle.dump(artifact, f, protocol=4)

size_mb = out_path.stat().st_size / 1e6
print(f"Guardado: {out_path}  ({size_mb:.1f} MB)")
"""))

# =============================================================================
# 9 - RESUMEN FINAL
# =============================================================================
cells.append(md("## 9 - Resumen Final NB10"))

cells.append(py("""\
print("=" * 60)
print("RESUMEN FINAL NB10")
print("=" * 60)
ndcg_mvae_val = m_mvae.get("NDCG@10", 0)
ndcg_rp3_val  = m_rp3.get("NDCG@10", 0)
ndcg_nb08     = 0.00934

print(f"Arquitectura   : MultiVAE enc={MVAE_ENC_DIMS}, z={MVAE_LATENT_DIM}")
print(f"Beta annealing : 0 -> {MVAE_BETA_MAX} (en {MVAE_ANNEAL_STEPS:,} steps)")
print(f"Epochs         : {MVAE_EPOCHS}")
print(f"Train users    : {MVAE_MAX_TRAIN_USERS:,}")
print()
print(f"NDCG@10 (MultiVAE)  : {ndcg_mvae_val:.4f}")
print(f"NDCG@10 (RP3beta)   : {ndcg_rp3_val:.4f}  (NB09 ganador)")
print(f"NDCG@10 (NB08 base) : {ndcg_nb08:.4f}")
print()
if ndcg_nb08 > 0:
    print(f"MultiVAE vs NB08: {(ndcg_mvae_val - ndcg_nb08) / ndcg_nb08 * 100:+.1f}%")
if ndcg_rp3_val > 0:
    delta_rp3 = (ndcg_mvae_val - ndcg_rp3_val) / ndcg_rp3_val * 100
    print(f"MultiVAE vs RP3beta: {delta_rp3:+.1f}%")
print("=" * 60)
"""))

# =============================================================================
# GENERAR NOTEBOOK
# =============================================================================
NB_PATH.parent.mkdir(exist_ok=True)
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "cells": cells,
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Generado: {NB_PATH}")
print(f"  Celdas totales : {len(cells)}")
print(f"  Celdas codigo  : {sum(1 for c in cells if c['cell_type']=='code')}")
