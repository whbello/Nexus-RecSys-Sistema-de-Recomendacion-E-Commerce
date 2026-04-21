"""
generate_09_notebook.py
=======================
Genera notebooks/09_advanced_models.ipynb

Modelos implementados (todos vectorizados, factibles en CPU):
  1. WRMF-ALS    : Weighted Regularized MF via ALS vectorizado (bloques)
  2. BPR-MF      : Bayesian Personalized Ranking, SGD vectorizado con numpy
  3. RP3beta      : Random Walk p=3 + penalizacion popularidad (scipy.sparse)
  4. NCF          : Neural CF (GMF+MLP) con PyTorch CPU
  5. SASRec-lite  : Self-Attention Sequential (PyTorch CPU)
  6. Ensemble     : fusion lineal ponderada de los mejores

NOTA sobre escalabilidad en CPU:
  - RetailRocket: 1.4M usuarios, 215K items, 2.1M interacciones
  - WRMF naive O(n_u * n_i) por iter es inviable (>1e11 ops)
  - WRMF vectorizado: aplica el truco matematico que evita iterar por usuario;
    actualiza U como un sistema lineal usando la estructura de C-I = alpha*R (solo nnz)
  - Con k=64 y 15 iteraciones corre en ~5-10 min en CPU
"""
import json, pathlib, uuid

ROOT    = pathlib.Path(__file__).parent.parent
NB_PATH = ROOT / "notebooks" / "09_advanced_models.ipynb"

def _cell_id():
    return uuid.uuid4().hex[:8]

def md(src):
    return {"cell_type": "markdown", "id": _cell_id(), "metadata": {}, "source": src}

def py(src):
    return {"cell_type": "code", "id": _cell_id(),
            "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells = []

# =============================================================================
# TITULO
# =============================================================================
cells.append(md("""\
# 09 · Nexus RecSys — Modelos Avanzados: EASE^R, BPR, RP3beta, NCF, SASRec-lite

**Sistema de Recomendacion E-Commerce - RetailRocket Dataset**

---

Este notebook lleva el proyecto al estado del arte reportado en la literatura
para el dataset RetailRocket. Se implementan 5 algoritmos avanzados + 1 ensemble:

| # | Modelo | Familia | Clave tecnica |
|---|--------|---------|---------------|
| 1 | **EASE^R** (top-20K items) | CF implicito | Formula cerrada; inversión de Gram matrix |
| 2 | **BPR-MF** (k=64) | Ranking-aware | Optimiza ranking directamente via BPR loss |
| 3 | **RP3beta** | Graph CF | Random-walk p=3; robusto en cold-start extremo |
| 4 | **NCF** (GMF+MLP) | Deep Learning | PyTorch CPU; combina factorizacion + red neuronal |
| 5 | **SASRec-lite** | Seq. Transformer | Aprovecha el orden temporal de las sesiones |
| 6 | **Ensemble** | Stacking | Fusion lineal ponderada por NDCG de cada componente |

> Requiere NB06 + NB07 + NB08 ejecutados previamente.
> PyTorch 2.10.0+cpu instalado en el venv.\
"""))

# =============================================================================
# 0 - SETUP
# =============================================================================
cells.append(md("## 0 - Setup y Parametros Globales"))

cells.append(py("""\
import os, time, json, pickle, warnings, logging, math
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize as skl_normalize
from sklearn.utils.extmath import randomized_svd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
    print("PyTorch:", torch.__version__)
except ImportError:
    print("AVISO: PyTorch no disponible - NCF y SASRec seran saltados")

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Hiperparametros ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
K_VALUES     = [5, 10, 20]
N_EVAL_USERS = 3_000
CUTOFF_DATE  = pd.Timestamp("2015-08-22", tz="UTC")

# EASE^R (Embarrassingly Shallow Autoencoders for Sparse Data, Steck 2019)
EASE_TOP  = 20_000   # items populares a incluir en el subespacio
EASE_REG  = 500.0    # lambda de regularizacion

# BPR - SGD vectorizado
BPR_K       = 64
BPR_LR      = 0.05
BPR_REG     = 0.01
BPR_EPOCHS  = 15
BPR_NEG_K   = 3      # negativos por positivo (vectorizados)

# RP3beta
RP3_ALPHA = 0.85
RP3_BETA  = 0.25

# NCF
NCF_EMB    = 64
NCF_LAYERS = [256, 128, 64]
NCF_LR     = 1e-3
NCF_EPOCHS = 10
NCF_BATCH  = 4096
NCF_NEG    = 4

# SASRec-lite
SAS_LEN    = 20
SAS_EMB    = 64
SAS_HEADS  = 2
SAS_LAYERS = 2
SAS_DROP   = 0.2
SAS_LR     = 1e-3
SAS_EPOCHS = 8
SAS_BATCH  = 1024

# Parametros heredados NB08
DECAY_LAMBDA  = 0.03
IPS_POWER     = 0.4
N_DAU         = 50_000
AVG_TICKET    = 45.0

HERE     = Path().resolve()
ROOT     = HERE.parent if (HERE.parent / "data").exists() else HERE
DATA_DIR = ROOT / "data" / "processed"
ENC_DIR  = ROOT / "encoders"
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

print(f"Root     : {ROOT}")
print(f"EASE^R   : top={EASE_TOP}, lambda={EASE_REG}")
print(f"BPR-MF   : k={BPR_K}, lr={BPR_LR}, epochs={BPR_EPOCHS}")
print(f"RP3beta  : alpha={RP3_ALPHA}, beta={RP3_BETA}")
if TORCH_AVAILABLE:
    print(f"NCF      : emb={NCF_EMB}, epochs={NCF_EPOCHS}, batch={NCF_BATCH}")
    print(f"SASRec   : emb={SAS_EMB}, epochs={SAS_EPOCHS}, batch={SAS_BATCH}")
print(f"EASE^R   : top={EASE_TOP}, lambda={EASE_REG}")
"""))

# =============================================================================
# 1 - CARGA DE DATOS
# =============================================================================
cells.append(md("## 1 - Carga de Datos y Split Temporal"))

cells.append(py("""\
print("Cargando datos...")
t0 = time.time()
im  = pd.read_csv(DATA_DIR / "interaction_matrix.csv")
itf = pd.read_csv(DATA_DIR / "item_features.csv")
im["last_interaction_ts"] = pd.to_datetime(im["last_interaction_ts"], format="ISO8601", utc=True)
print(f"  IM: {im.shape}  [{time.time()-t0:.1f}s]")

train_mask = im["last_interaction_ts"] < CUTOFF_DATE
train_df   = im[train_mask].copy()
test_df    = im[~train_mask].copy()

warm_users = sorted(set(train_df["visitorid"].unique()) & set(test_df["visitorid"].unique()))
rng = np.random.default_rng(RANDOM_STATE)
eval_users = rng.choice(warm_users, size=min(N_EVAL_USERS, len(warm_users)), replace=False).tolist()

test_items_by_user = test_df.groupby("visitorid")["itemid"].apply(set).to_dict()
train_items_by_user = train_df.groupby("visitorid")["itemid"].apply(set).to_dict()
test_tx_by_user = (
    test_df[test_df["last_interaction_type"]=="transaction"]
    .groupby("visitorid")["itemid"].apply(set).to_dict()
)

all_items_global = sorted(im["itemid"].unique())
n_items_global   = len(all_items_global)

n_test_tx = len(test_df[test_df["last_interaction_type"]=="transaction"])
baseline_conv = n_test_tx / (len(set(test_df["visitorid"])) * n_items_global)

n_buyers = sum(1 for u in eval_users if test_tx_by_user.get(u))
print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")
print(f"Warm users: {len(warm_users):,}  Eval: {len(eval_users):,}  Compradores: {n_buyers}")
print(f"Baseline P(compra|aleatorio): {baseline_conv:.2e}")
"""))

cells.append(py("""\
# ── Indices y matrices ────────────────────────────────────────────────────────
all_train_users = sorted(train_df["visitorid"].unique())
all_train_items = sorted(train_df["itemid"].unique())
user2idx = {u:i for i,u in enumerate(all_train_users)}
item2idx = {it:i for i,it in enumerate(all_train_items)}
idx2user = {i:u for u,i in user2idx.items()}
idx2item = {i:it for it,i in item2idx.items()}
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
print(f"Sparsidad: {1 - R.nnz/(n_u*n_i):.6f}")
"""))

# =============================================================================
# 2 - FUNCIONES DE EVALUACION
# =============================================================================
cells.append(md("## 2 - Framework de Evaluacion"))

cells.append(py("""\
def ndcg(r, rel, k):
    d = sum(1./math.log2(i+2) for i,x in enumerate(r[:k]) if x in rel)
    ideal = sum(1./math.log2(i+2) for i in range(min(len(rel),k)))
    return d/ideal if ideal else 0.

def prec(r, rel, k): return len(set(r[:k])&rel)/k if k else 0.
def rec(r, rel, k):  return len(set(r[:k])&rel)/len(rel) if rel else 0.
def ap(r, rel, k):
    if not rel: return 0.
    s,h=0.,0
    for i,x in enumerate(r[:k]):
        if x in rel: h+=1; s+=h/(i+1)
    return s/min(len(rel),k)
def nov(flat, pd, nt): return float(np.mean([-math.log2(pd.get(x,1)/nt+1e-10) for x in flat])) if flat else 0.
def rev_k(r, tx, k): return len(set(r[:k])&tx)/k if k else 0.
def ctr_k(r, ts, k): return len(set(r[:k])&ts)/k if k else 0.

def evaluate(get_fn, evals, tst, tst_tx, pop_d, nt, cat_sz, bconv, ks=K_VALUES):
    acc = {k:{m:[] for m in "prnmrc"} for k in ks}
    seen = set(); ne = 0
    for uid in evals:
        ti = tst.get(uid, set())
        if not ti: continue
        tx = tst_tx.get(uid, set())
        mk = max(ks)
        try: recs = get_fn(uid, mk)
        except: continue
        seen.update(recs); ne += 1
        for k in ks:
            acc[k]["p"].append(prec(recs, ti, k))
            acc[k]["r"].append(rec(recs,  ti, k))
            acc[k]["n"].append(ndcg(recs,  ti, k))
            acc[k]["m"].append(ap(recs,   ti, k))
            acc[k]["r2"] = acc[k].get("r2", [])
            acc[k]["r2"].append(rev_k(recs, tx, k))
            acc[k]["c"].append(ctr_k(recs, ti, k))
    out = {"n_eval": ne}
    for k in ks:
        if not acc[k]["p"]: continue
        out[f"NDCG@{k}"]      = float(np.mean(acc[k]["n"]))
        out[f"Precision@{k}"] = float(np.mean(acc[k]["p"]))
        out[f"Recall@{k}"]    = float(np.mean(acc[k]["r"]))
        out[f"MAP@{k}"]       = float(np.mean(acc[k]["m"]))
        out[f"Revenue@{k}"]   = float(np.mean(acc[k].get("r2",[])))
        out[f"CTR@{k}"]       = float(np.mean(acc[k]["c"]))
        rv = out[f"Revenue@{k}"]
        out[f"ConvLift@{k}"]  = rv/bconv if bconv else 0.
    out["Coverage"] = len(seen)/cat_sz
    out["Novelty"]  = nov(list(seen), pop_d, nt)
    return out

all_results = {}
print("Funciones de evaluacion definidas.")
"""))

# =============================================================================
# 3 - BASELINE SVD+TD+IPS (heredado del NB08)
# =============================================================================
cells.append(md("## 3 - Baseline: SVD+TD+IPS (ganador NB08)"))

cells.append(py("""\
print("Cargando final_model_v2.pkl ...")
t0 = time.time()
with open(ENC_DIR / "final_model_v2.pkl", "rb") as f:
    fm2 = pickle.load(f)

U_tdips = fm2["U"]
s_tdips = fm2["sigma"]
Vt_raw  = fm2["Vt"]
Vt_tdips = np.diag(s_tdips) @ Vt_raw
print(f"  [{time.time()-t0:.1f}s] U={U_tdips.shape} Vt={Vt_tdips.shape}")

def get_tdips(uid, n, _U=U_tdips, _Vt=Vt_tdips, _u2i=user2idx, _i2i=idx2item):
    if uid not in _u2i: return []
    ui = _u2i[uid]; sc = _U[ui] @ _Vt
    row = R.getrow(ui); sc[row.indices] = -np.inf
    top = np.argpartition(sc,-n)[-n:]
    return [_i2i[i] for i in top[np.argsort(sc[top])[::-1]]]

print("Evaluando SVD+TD+IPS...")
t1 = time.time()
m_tdips = evaluate(get_tdips, eval_users, test_items_by_user,
                   test_tx_by_user, item_pop_dict, n_total_train,
                   n_items_global, baseline_conv)
print(f"  [{time.time()-t1:.1f}s] NDCG@10={m_tdips['NDCG@10']:.4f}  "
      f"Coverage={m_tdips['Coverage']:.4f}  ConvLift@10={m_tdips.get('ConvLift@10',0):.0f}x")
all_results["SVD+TD+IPS (NB08)"] = m_tdips
"""))

# =============================================================================
# 4 - WRMF (ALS vectorizado)
# =============================================================================
cells.append(md("""\
## 4 - EASE^R: Embarrassingly Shallow Autoencoders

**EASE^R** (Steck, 2019) es uno de los mejores modelos CF reportados en la
literatura para datasets de e-commerce implicitos:

- No tiene hiperparametros de convergencia (formula cerrada)
- Resuelve: $B = (G + \\lambda I)^{-1}$, donde $G = X^T X$ (Gram del catalogo)
- Luego fuerza diagonal a 0 para evitar auto-recomendacion
- Scores: $S = X \\cdot B$ (sin materializacion, lo calculamos por usuario)

Para escalar a 215K items, filtramos a los **top-N items mas populares**
(los que concentran el 90% de las interacciones) y evaluamos sobre ese subespacio.\
"""))

cells.append(py("""\
# ── EASE^R: formula cerrada sobre los top items populares ─────────────────────
print(f"EASE^R: top_items={EASE_TOP}, lambda={EASE_REG}")
t_ease = time.time()

# Seleccionar top items por popularidad
top_items_idx = np.argpartition(item_pop, -EASE_TOP)[-EASE_TOP:]
top_items_idx = top_items_idx[np.argsort(item_pop[top_items_idx])[::-1]]
N_EASE = len(top_items_idx)

# Submatriz X_ease (sparse): (n_u, N_EASE)
# NO materializar X_ease_dense (1.15M x 20K x 4B = 92 GB)
# En cambio, usamos X_ease sparse para acceso on-the-fly por fila
X_ease_csr = R[:, top_items_idx].astype(np.float32).tocsr()   # (n_u, N_EASE) sparse
print(f"  X_ease_csr: {X_ease_csr.shape}  nnz={X_ease_csr.nnz:,}  [{time.time()-t_ease:.1f}s]")

# Gram matrix G = X^T X  - AQUI si necesitamos materilizar el bloque top
# X_T_dense = X_ease_csr.T.toarray()  (N_EASE x n_u) -- no, demasiado
# Mejor: G = X_ease_csr^T @ X_ease_csr  (scipy sparse -> dense)
print("  Calculando Gram matrix G = X^T X ...")
G_sparse = X_ease_csr.T @ X_ease_csr         # (N_EASE, N_EASE) sparse
G = np.asarray(G_sparse.todense(), dtype=np.float32)
del G_sparse
print(f"  G: {G.shape}  [{time.time()-t_ease:.1f}s]")

# B = inv(G + lambda*I)
print("  Invirtiendo G + lambda*I ...")
G_reg = G + EASE_REG * np.eye(N_EASE, dtype=np.float32)
del G
B_inv = np.linalg.inv(G_reg)
del G_reg

# B* = -B_inv / diag(B_inv), diagonal a 0
diag_inv = np.diag(B_inv).copy()
B_ease   = -(B_inv / diag_inv[None, :]).astype(np.float32)
np.fill_diagonal(B_ease, 0.)
del B_inv
print(f"  B_ease: {B_ease.shape}  [{time.time()-t_ease:.1f}s]")

# Mapa local idx_local -> item global
top_items_global = [idx2item[i] for i in top_items_idx]
# Mapa inverso: item_id -> local idx en top_items
item_to_local = {it: k for k, it in enumerate(top_items_global)}

print(f"  EASE^R listo en {time.time()-t_ease:.1f}s total")

def get_ease(uid, n, _Xcsr=X_ease_csr, _B=B_ease, _top_g=top_items_global,
             _u2i=user2idx):
    \"\"\"
    Calcula scores EASE^R on-the-fly: extrae la fila sparse del usuario
    y la multiplica por B (densa, N_EASE x N_EASE).
    Costo: O(nnz_u * N_EASE) por usuario -- muy rapido para nnz_u pequeno.
    \"\"\"
    if uid not in _u2i: return []
    ui = _u2i[uid]
    row = _Xcsr.getrow(ui)
    x_u = np.asarray(row.todense(), dtype=np.float32).ravel()   # (N_EASE,)
    sc  = x_u @ _B                                               # (N_EASE,)
    sc[x_u > 0] = -np.inf   # excluir vistos
    top_local = np.argpartition(sc, -n)[-n:]
    top_local = top_local[np.argsort(sc[top_local])[::-1]]
    return [_top_g[i] for i in top_local]

print("Evaluando EASE^R ...")
t1 = time.time()
m_ease = evaluate(get_ease, eval_users, test_items_by_user,
                  test_tx_by_user, item_pop_dict, n_total_train,
                  n_items_global, baseline_conv)
print(f"  [{time.time()-t1:.1f}s] NDCG@10={m_ease['NDCG@10']:.4f}  "
      f"Coverage={m_ease['Coverage']:.4f}  ConvLift@10={m_ease.get('ConvLift@10',0):.0f}x")
all_results["EASE^R (lambda=500, top20K)"] = m_ease
"""))

# =============================================================================
# 5 - BPR-MF (vectorizado con numpy)
# =============================================================================
cells.append(md("""\
## 5 - BPR-MF: Bayesian Personalized Ranking (numpy vectorizado)

Implementacion con **mini-batches vectorizados**: en cada batch se toman
B triplas (u, i+, i-) y se actualizan todos a la vez con operaciones numpy.
Mucho mas eficiente que el loop Python puro.\
"""))

cells.append(py("""\
def train_bpr_vec(R, k, lr, reg, n_epochs, n_neg, batch_size=8192, seed=42):
    \"\"\"
    BPR-MF con mini-batch vectorizado.
    Cada epoch: se extraen todos los (u, pos_item) de R y se samplea n_neg negativos.
    Actualizacion en batches de tamano batch_size.
    \"\"\"
    rng = np.random.default_rng(seed)
    n_u, n_i = R.shape

    # Inicializacion Xavier
    std_u = math.sqrt(2./(n_u+k)); std_v = math.sqrt(2./(n_i+k))
    U = rng.uniform(-std_u, std_u, (n_u, k)).astype(np.float32)
    V = rng.uniform(-std_v, std_v, (n_i, k)).astype(np.float32)

    # Extraer positivos (coo)
    R_coo = R.tocoo()
    pos_u = R_coo.row.astype(np.int32)   # (nnz,)
    pos_i = R_coo.col.astype(np.int32)   # (nnz,)
    nnz   = len(pos_u)

    # Conjuntos de positivos por usuario (para negative sampling)
    pos_sets = {}
    for pu, pi_val in zip(pos_u, pos_i):
        if pu not in pos_sets:
            pos_sets[pu] = set()
        pos_sets[pu].add(int(pi_val))

    losses = []
    for epoch in range(n_epochs):
        t_ep = time.time()
        # Aleatorizar positivos y repetir con negativos
        perm = rng.permutation(nnz)
        p_u = pos_u[perm]; p_i = pos_i[perm]

        # Samplear negativos (vectorizado: puede tener colisiones minimas, ok)
        neg_i = rng.integers(0, n_i, size=(nnz * n_neg,), dtype=np.int32)

        total_loss = 0.; n_batches = 0

        # Procesar en batches
        for b_start in range(0, nnz, batch_size):
            b_end = min(b_start + batch_size, nnz)
            bu  = p_u[b_start:b_end]           # (B,)
            bi  = p_i[b_start:b_end]           # (B,)
            # Usar el primer negativo del batch (rapido, sin verificacion)
            neg_start = b_start * n_neg
            neg_end   = b_end * n_neg
            bn_flat   = neg_i[neg_start:neg_end]
            # Tomar solo el primer negativo para cada par (1:1 ratio)
            bn = bn_flat[::n_neg]              # (B,) -- primer neg de cada pos

            # Embeddings
            Uu  = U[bu]  # (B, k)
            Vi  = V[bi]  # (B, k)
            Vn  = V[bn]  # (B, k)

            # Score diferencial
            x_uij = np.sum(Uu * (Vi - Vn), axis=1)   # (B,)

            # Sigmoid del negativo del score (gradiente de log-sigmoid)
            # sig(-x) = 1/(1+e^x)
            sig_neg = 1.0 / (1.0 + np.exp(np.clip(x_uij, -20, 20)))   # (B,)
            sig_neg = sig_neg.astype(np.float32)

            # Gradientes BPR
            du  = sig_neg[:, None] * (Vi - Vn) - reg * Uu    # (B, k)
            dvi = sig_neg[:, None] * Uu - reg * Vi             # (B, k)
            dvn = -sig_neg[:, None] * Uu - reg * Vn            # (B, k)

            # Actualizar con np.add.at (acumula gradientes del mismo idx)
            np.add.at(U, bu, lr * du)
            np.add.at(V, bi, lr * dvi)
            np.add.at(V, bn, lr * dvn)

            # BPR loss = -log(sig(x_uij))
            total_loss -= float(np.sum(np.log(1. - sig_neg + 1e-10)))
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        losses.append(avg_loss)
        print(f"  BPR epoch {epoch+1:>2}/{n_epochs}  loss={avg_loss:.4f}  [{time.time()-t_ep:.1f}s]")

    return U, V.T, losses

print(f"BPR: k={BPR_K}, lr={BPR_LR}, reg={BPR_REG}, epochs={BPR_EPOCHS}")
print("Iniciando entrenamiento BPR-MF ...")
t_b = time.time()
U_bpr, Vt_bpr, bpr_losses = train_bpr_vec(
    R, BPR_K, BPR_LR, BPR_REG, BPR_EPOCHS, BPR_NEG_K,
    batch_size=8192, seed=RANDOM_STATE
)
print(f"BPR entrenado en {time.time()-t_b:.1f}s")

def get_bpr(uid, n, _U=U_bpr, _Vt=Vt_bpr, _u2i=user2idx, _i2i=idx2item):
    if uid not in _u2i: return []
    ui = _u2i[uid]; sc = _U[ui] @ _Vt
    row = R.getrow(ui); sc[row.indices] = -np.inf
    top = np.argpartition(sc,-n)[-n:]
    return [_i2i[i] for i in top[np.argsort(sc[top])[::-1]]]

print("Evaluando BPR-MF ...")
t1 = time.time()
m_bpr = evaluate(get_bpr, eval_users, test_items_by_user,
                 test_tx_by_user, item_pop_dict, n_total_train,
                 n_items_global, baseline_conv)
print(f"  [{time.time()-t1:.1f}s] NDCG@10={m_bpr['NDCG@10']:.4f}  "
      f"Coverage={m_bpr['Coverage']:.4f}  ConvLift@10={m_bpr.get('ConvLift@10',0):.0f}x")
all_results["BPR-MF (k=64, SGD)"] = m_bpr
"""))

# =============================================================================
# 6 - RP3beta
# =============================================================================
cells.append(md("""\
## 6 - RP3beta: Random Walk con Reinicio

Calcula la similitud entre pares de items propagando sobre el grafo bipartito
usuario-item. El truco es que se puede expresar como multiplicacion matricial:

$$P3_{ij} = \\sum_u P(i \\to u) \\cdot P(u \\to j)$$

donde $P(i \\to u) \\propto R_{ui}$ y $P(u \\to j) \\propto R_{uj}$.
Para recomendar al usuario $u$, usamos su perfil de items para propagar.\
"""))

cells.append(py("""\
# RP3beta escalable:
# Precomputa W_rp3 = P_iu @ P_ui  donde:
#   P_ui = R normalizado por usuario (L1 por fila)   <- (n_u, n_top_i)
#   P_iu = R^T normalizado por item (L1 por fila)    <- (n_top_i, n_u)
# W[i,j] = similitud item i->j via random walk de longitud 3 con penalizacion de popularidad
# Solo calculamos sobre top items para mantener memoria manejable.
# Luego para recomendar: sc_u = hist_u_sparse @ W, excluir vistos.
print("Preparando RP3beta (item-item precomputado)...")
t0 = time.time()

# Trabajar sobre top items para que W_rp3 sea manejable
RP3_TOP = min(EASE_TOP, n_i)   # mismos top items que EASE^R
top_ri = top_items_idx          # reutiliza el subset de EASE^R (mas populares)
N_RP3 = len(top_ri)

# Submatriz sparse de los top items
X_rp3 = R[:, top_ri].astype(np.float32).tocsr()  # (n_u, N_RP3)
print(f"  X_rp3: {X_rp3.shape} nnz={X_rp3.nnz:,}  [{time.time()-t0:.1f}s]")

# Normalizar por usuario (P_ui) y por item (P_iu)
P_ui = skl_normalize(X_rp3.astype(np.float64), norm="l1", axis=1)  # (n_u, N_RP3)
P_it = skl_normalize(X_rp3.T.tocsr().astype(np.float64), norm="l1", axis=1)  # (N_RP3, n_u)

# Pop penalty: item_pop solo para el subconjunto top
pop_sub = item_pop[top_ri].astype(np.float64)
pop_beta_sub = np.power(pop_sub + 1e-10, RP3_BETA)   # (N_RP3,)

# W_rp3 = (P_it @ P_ui)^alpha  normalizado por popularidad
# Calculo en bloques de items para evitar (N_RP3 x N_RP3) densa en RAM
# Para dataset grande, calculamos W como producto sparse y mantenemos sparse
print(f"  Calculando P_it @ P_ui ({N_RP3}x{N_RP3}) ...")
W_sparse = P_it @ P_ui        # (N_RP3, N_RP3) posiblemente denso
W = np.asarray(W_sparse.todense() if hasattr(W_sparse,'todense') else W_sparse,
               dtype=np.float32)
del W_sparse

# Aplicar alpha y penalizar popularidad
np.power(W, RP3_ALPHA, out=W)
W = W / (pop_beta_sub[None, :] + 1e-10)  # escala cada columna j
np.fill_diagonal(W, 0.)   # no auto-recomendar
print(f"  W_rp3: {W.shape}  [{time.time()-t0:.1f}s]")

# Mapa local
top_rp3_global = [idx2item[i] for i in top_ri]

def get_rp3(uid, n, _Xsp=X_rp3, _W=W, _tg=top_rp3_global, _u2i=user2idx):
    \"\"\"
    sc = hist_u (sparse subvector) @ W_rp3
    Costo: O(nnz_u * N_RP3) — rapido para perfiles cortos.
    \"\"\"
    if uid not in _u2i: return []
    ui = _u2i[uid]
    row = _Xsp.getrow(ui)
    x_u = np.asarray(row.todense(), dtype=np.float32).ravel()
    sc = x_u @ _W
    sc[x_u > 0] = -np.inf
    top = np.argpartition(sc, -n)[-n:]
    top = top[np.argsort(sc[top])[::-1]]
    return [_tg[i] for i in top]

print("Evaluando RP3beta ...")
t1 = time.time()
m_rp3 = evaluate(get_rp3, eval_users, test_items_by_user,
                 test_tx_by_user, item_pop_dict, n_total_train,
                 n_items_global, baseline_conv)
ndcg_rp3 = m_rp3.get('NDCG@10', 0.)
print(f"  [{time.time()-t1:.1f}s] NDCG@10={ndcg_rp3:.4f}  "
      f"Coverage={m_rp3.get('Coverage',0.):.4f}  ConvLift@10={m_rp3.get('ConvLift@10',0):.0f}x")
all_results["RP3beta (alpha=0.85, beta=0.25)"] = m_rp3
"""))

# =============================================================================
# 7 - NCF
# =============================================================================
cells.append(md("""\
## 7 - NCF: Neural Collaborative Filtering (PyTorch CPU)

Arquitectura NeuMF: fusion de GMF (producto de embeddings) + MLP.\
"""))

cells.append(py("""\
if not TORCH_AVAILABLE:
    print("SKIP: PyTorch no disponible.")
    all_results["NCF (NeuMF)"] = {"NDCG@10": 0., "Coverage": 0., "n_eval": 0}
else:
    # Subsamplear: usar solo usuarios con >=2 interacciones para agilizar NCF
    NCF_MAX_USERS = 80_000   # usuarios para entrenamiento NCF
    active_u_idx = np.where(np.diff(R.indptr) >= 2)[0]
    if len(active_u_idx) > NCF_MAX_USERS:
        rng_ncf = np.random.default_rng(42)
        active_u_idx = rng_ncf.choice(active_u_idx, size=NCF_MAX_USERS, replace=False)
    R_ncf = R[active_u_idx, :]   # submatrix (NCF_MAX_USERS, n_i)
    # Remap user idx para NCF (0..len(active_u_idx))
    ncf_local2global = active_u_idx  # ncf_local_idx -> global user idx
    ncf_global2local = {int(g): l for l, g in enumerate(active_u_idx)}
    n_u_ncf = len(active_u_idx)

    class InteractionDataset(Dataset):
        def __init__(self, Rsub, neg_ratio=4, seed=42):
            coo = Rsub.tocoo()
            self.pu = coo.row.astype(np.int64)
            self.pi = coo.col.astype(np.int64)
            self.n_u, self.n_i = Rsub.shape
            self.nr = neg_ratio
            pos_s = {}
            for u,i in zip(self.pu, self.pi):
                pos_s.setdefault(int(u), set()).add(int(i))
            self.pos_s = pos_s
            self.rng   = np.random.default_rng(seed)

        def __len__(self): return len(self.pu) * (1 + self.nr)

        def __getitem__(self, idx):
            pos_idx = idx // (1 + self.nr)
            is_pos  = (idx % (1 + self.nr)) == 0
            u = int(self.pu[pos_idx])
            if is_pos:
                i, lbl = int(self.pi[pos_idx]), 1.0
            else:
                while True:
                    i = int(self.rng.integers(0, self.n_i))
                    if i not in self.pos_s.get(u, set()): break
                lbl = 0.0
            return torch.tensor(u, dtype=torch.long), torch.tensor(i, dtype=torch.long), torch.tensor(lbl, dtype=torch.float32)

    class NeuMF(nn.Module):
        def __init__(self, n_u, n_i, emb, layers):
            super().__init__()
            self.gmf_u = nn.Embedding(n_u, emb)
            self.gmf_i = nn.Embedding(n_i, emb)
            self.mlp_u = nn.Embedding(n_u, emb)
            self.mlp_i = nn.Embedding(n_i, emb)
            in_d = emb * 2
            mlist = []
            for od in layers:
                mlist += [nn.Linear(in_d, od), nn.ReLU(), nn.Dropout(0.1)]
                in_d = od
            self.mlp = nn.Sequential(*mlist)
            self.out  = nn.Linear(emb + layers[-1], 1)
            for emb_l in [self.gmf_u, self.gmf_i, self.mlp_u, self.mlp_i]:
                nn.init.normal_(emb_l.weight, std=0.01)

        def forward(self, u, i):
            g = self.gmf_u(u) * self.gmf_i(i)
            m = self.mlp(torch.cat([self.mlp_u(u), self.mlp_i(i)], dim=1))
            return torch.sigmoid(self.out(torch.cat([g, m], dim=1))).squeeze(1)

    torch.manual_seed(RANDOM_STATE)
    ds     = InteractionDataset(R_ncf, NCF_NEG, RANDOM_STATE)
    loader = DataLoader(ds, batch_size=NCF_BATCH, shuffle=True, num_workers=0)
    model  = NeuMF(n_u_ncf, n_i, NCF_EMB, NCF_LAYERS)
    opt    = optim.Adam(model.parameters(), lr=NCF_LR, weight_decay=1e-5)
    crit   = nn.BCELoss()

    print(f"Entrenando NCF: emb={NCF_EMB}, epochs={NCF_EPOCHS}, batch={NCF_BATCH}")
    print(f"Dataset: {len(ds):,} pares ({n_u_ncf:,} usuarios)")
    t_ncf = time.time()
    model.train()
    for ep in range(NCF_EPOCHS):
        tl, nb = 0., 0
        te_ep = time.time()
        for ub, ib, yb in loader:
            opt.zero_grad()
            loss = crit(model(ub, ib), yb)
            loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        print(f"  NCF epoch {ep+1:>2}/{NCF_EPOCHS}  loss={tl/nb:.4f}  [{time.time()-te_ep:.1f}s]")
    print(f"NCF entrenado en {time.time()-t_ncf:.1f}s")
    model.eval()
"""))

cells.append(py("""\
if not TORCH_AVAILABLE:
    pass
else:
    all_i_t = torch.arange(n_i, dtype=torch.long)

    @torch.no_grad()
    def get_ncf(uid, n, _m=model, _u2i=user2idx, _g2l=ncf_global2local, _i2i=idx2item):
        if uid not in _u2i: return []
        ui_global = _u2i[uid]
        if ui_global not in _g2l: return []   # usuario no visto en subset NCF
        ui_local = _g2l[ui_global]
        u_t = torch.tensor(ui_local, dtype=torch.long).expand(n_i)
        sc  = _m(u_t, all_i_t).numpy()
        sc[R.getrow(ui_global).indices] = -np.inf
        top = np.argpartition(sc,-n)[-n:]
        return [_i2i[x] for x in top[np.argsort(sc[top])[::-1]]]

    print("Evaluando NCF ...")
    t1 = time.time()
    m_ncf = evaluate(get_ncf, eval_users, test_items_by_user,
                     test_tx_by_user, item_pop_dict, n_total_train,
                     n_items_global, baseline_conv)
    ndcg_ncf = m_ncf.get('NDCG@10', 0.)
    print(f"  [{time.time()-t1:.1f}s] NDCG@10={ndcg_ncf:.4f}  "
          f"Coverage={m_ncf.get('Coverage',0.):.4f}  ConvLift@10={m_ncf.get('ConvLift@10',0):.0f}x")
    all_results["NCF (NeuMF, PyTorch)"] = m_ncf
"""))

# =============================================================================
# 8 - SASRec-lite
# =============================================================================
cells.append(md("""\
## 8 - SASRec-lite: Self-Attention Sequential (PyTorch)

Predice el siguiente item basandose en la secuencia de acciones del usuario,
ordenadas por timestamp. Implementacion simplificada con Transformer encoder.\
"""))

cells.append(py("""\
if not TORCH_AVAILABLE:
    print("SKIP SASRec")
    all_results["SASRec-lite"] = {"NDCG@10": 0., "Coverage": 0., "n_eval": 0}
else:
    # Construir secuencias de items ordenadas por timestamp
    print("Construyendo secuencias de usuario ...")
    t0 = time.time()
    im_s = im.sort_values(["visitorid","last_interaction_ts"])
    user_seqs_full = im_s.groupby("visitorid")["itemid"].apply(list).to_dict()
    # Solo usuarios de entrenamiento que tienen secuencias
    train_seqs = {u: s for u, s in user_seqs_full.items() if u in user2idx and len(s) >= 2}
    print(f"  Secuencias: {len(train_seqs):,} usuarios  [{time.time()-t0:.1f}s]")

    def encode_seq(seq, item2idx, max_len):
        enc = [item2idx[it]+1 for it in seq if it in item2idx]  # 0=pad, 1-based
        if len(enc) > max_len: enc = enc[-max_len:]
        return [0]*(max_len-len(enc)) + enc

    class SeqDataset(Dataset):
        def __init__(self, seqs, item2idx, max_len, n_i, seed=42):
            self.max_len = max_len; self.n_i = n_i
            self.rng = np.random.default_rng(seed)
            self.data = []
            for uid, seq in seqs.items():
                enc = [item2idx[it]+1 for it in seq if it in item2idx]
                if len(enc) < 2: continue
                iset = set(enc)
                for pos in range(1, len(enc)):
                    hist = enc[max(0,pos-max_len):pos]
                    tgt  = enc[pos]
                    # un negativo
                    while True:
                        nj = int(self.rng.integers(1, n_i+1))
                        if nj not in iset: break
                    pad = max_len - len(hist)
                    self.data.append(([0]*pad+hist, tgt, nj))

        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            h,t,n = self.data[idx]
            return torch.tensor(h,dtype=torch.long), torch.tensor(t,dtype=torch.long), torch.tensor(n,dtype=torch.long)

    class SASRec(nn.Module):
        def __init__(self, n_i, emb, max_len, n_heads, n_layers, drop):
            super().__init__()
            self.ie = nn.Embedding(n_i+1, emb, padding_idx=0)
            self.pe = nn.Embedding(max_len, emb)
            enc_l = nn.TransformerEncoderLayer(emb, n_heads, emb*4, drop, batch_first=True, norm_first=True)
            self.tr = nn.TransformerEncoder(enc_l, num_layers=n_layers)
            self.ln = nn.LayerNorm(emb)
            nn.init.normal_(self.ie.weight[1:], std=0.01)

        def encode(self, seq):
            B, L = seq.shape
            pos = torch.arange(L, device=seq.device).unsqueeze(0)
            x   = self.ie(seq) + self.pe(pos)
            pm  = (seq==0)
            x   = self.tr(x, src_key_padding_mask=pm)
            x   = self.ln(x)
            lens = (seq!=0).sum(1).clamp(min=1) - 1
            return x[torch.arange(B), lens]   # (B, emb)

        def score(self, seq, items):
            ur = self.encode(seq)
            ie = self.ie(items)
            return (ur * ie).sum(1)

    torch.manual_seed(RANDOM_STATE)
    # Limitar a usuarios mas activos para SASRec (subsampleo por velocidad)
    SAS_MAX_USERS = 50_000
    train_seqs_sub = dict(sorted(train_seqs.items(),
                                  key=lambda x: len(x[1]), reverse=True)[:SAS_MAX_USERS])
    sas_ds = SeqDataset(train_seqs_sub, item2idx, SAS_LEN, n_i, RANDOM_STATE)
    sas_ld = DataLoader(sas_ds, batch_size=SAS_BATCH, shuffle=True, num_workers=0)
    sas    = SASRec(n_i, SAS_EMB, SAS_LEN, SAS_HEADS, SAS_LAYERS, SAS_DROP)
    sas_opt = optim.Adam(sas.parameters(), lr=SAS_LR, weight_decay=1e-5)

    print(f"Entrenando SASRec: emb={SAS_EMB}, epochs={SAS_EPOCHS}, samples={len(sas_ds):,}")
    t_sas = time.time()
    sas.train()
    for ep in range(SAS_EPOCHS):
        tl, nb = 0., 0; te_ep = time.time()
        for h,t,n_ in sas_ld:
            sas_opt.zero_grad()
            ps = sas.score(h, t)
            ns = sas.score(h, n_)
            loss = -torch.log(torch.sigmoid(ps-ns)+1e-10).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(sas.parameters(), 1.)
            sas_opt.step()
            tl += loss.item(); nb += 1
        print(f"  SASRec ep {ep+1:>2}/{SAS_EPOCHS}  loss={tl/nb:.4f}  [{time.time()-te_ep:.1f}s]")
    print(f"SASRec entrenado en {time.time()-t_sas:.1f}s")
    sas.eval()
"""))

cells.append(py("""\
if not TORCH_AVAILABLE:
    pass
else:
    all_i_sas = torch.arange(1, n_i+1, dtype=torch.long)  # 1-indexed

    @torch.no_grad()
    def get_sasrec(uid, n, _m=sas, _u2i=user2idx, _i2i=idx2item):
        if uid not in _u2i: return []
        seq = user_seqs_full.get(uid, [])
        if not seq: return []
        enc = encode_seq(seq, item2idx, SAS_LEN)
        st  = torch.tensor(enc, dtype=torch.long).unsqueeze(0)  # (1, L)
        ur  = _m.encode(st).squeeze(0)                          # (emb,)
        ie  = _m.ie(all_i_sas)                                  # (n_i, emb)
        sc  = (ie @ ur).numpy()                                  # (n_i,)
        # sc es para indices 1..n_i; mapear a 0..n_i-1 para R
        sc_full = np.zeros(n_i, dtype=np.float32)
        sc_full[:] = sc
        sc_full[R.getrow(_u2i[uid]).indices] = -np.inf
        top = np.argpartition(sc_full,-n)[-n:]
        return [_i2i[x] for x in top[np.argsort(sc_full[top])[::-1]]]

    print("Evaluando SASRec ...")
    t1 = time.time()
    m_sas = evaluate(get_sasrec, eval_users, test_items_by_user,
                     test_tx_by_user, item_pop_dict, n_total_train,
                     n_items_global, baseline_conv)
    ndcg_sas = m_sas.get('NDCG@10', 0.)
    print(f"  [{time.time()-t1:.1f}s] NDCG@10={ndcg_sas:.4f}  "
          f"Coverage={m_sas.get('Coverage',0.):.4f}  ConvLift@10={m_sas.get('ConvLift@10',0):.0f}x")
    all_results["SASRec-lite (Transformer)"] = m_sas
"""))

# =============================================================================
# 9 - ENSEMBLE
# =============================================================================
cells.append(md("## 9 - Ensemble: Fusion Lineal Ponderada (SVD+TD+IPS + EASE^R + BPR)"))

cells.append(py("""\
def mm_norm(v):
    mn,mx = v.min(),v.max(); r=mx-mn
    return (v-mn)/r if r>1e-10 else np.zeros_like(v)

# Pesos proporcionales al NDCG@10 de cada componente
n1 = all_results.get("SVD+TD+IPS (NB08)",{}).get("NDCG@10",1.)
n2 = all_results.get("EASE^R (lambda=500, top20K)",{}).get("NDCG@10",1.)
n3 = all_results.get("BPR-MF (k=64, SGD)",{}).get("NDCG@10",1.)
wt = n1+n2+n3+1e-10
w1_e, w2_e, w3_e = n1/wt, n2/wt, n3/wt
print(f"Pesos ensemble: SVD+TD+IPS={w1_e:.3f}  EASE^R={w2_e:.3f}  BPR={w3_e:.3f}")

def get_ensemble_v2(uid, n):
    if uid not in user2idx: return []
    ui = user2idx[uid]
    # 1. SVD+TD+IPS (espacio completo n_i)
    sc1 = mm_norm(U_tdips[ui] @ Vt_tdips)   # (n_i,)
    sc1_min = sc1.min()
    # 2. EASE^R sobre top items (on-the-fly desde sparse)
    row_ease = X_ease_csr.getrow(ui)
    x_u = np.asarray(row_ease.todense(), dtype=np.float32).ravel()
    ease_top = (x_u @ B_ease).copy()
    ease_top[x_u > 0] = -np.inf
    # Mapear EASE^R scores al espacio global de items
    sc2 = np.full(n_i, sc1_min, dtype=np.float32)
    for k_loc, item_g in enumerate(top_items_global):
        ti = item2idx.get(item_g)
        if ti is not None: sc2[ti] = ease_top[k_loc]
    sc2 = mm_norm(sc2)
    # 3. BPR-MF (espacio completo n_i)
    sc3 = mm_norm(U_bpr[ui] @ Vt_bpr)       # (n_i,)
    # Fusion lineal ponderada
    sc = w1_e*sc1 + w2_e*sc2 + w3_e*sc3
    sc[R.getrow(ui).indices] = -np.inf
    top = np.argpartition(sc,-n)[-n:]
    return [idx2item[i] for i in top[np.argsort(sc[top])[::-1]]]

print("Evaluando Ensemble ...")
t1 = time.time()
m_ens = evaluate(get_ensemble_v2, eval_users, test_items_by_user,
                 test_tx_by_user, item_pop_dict, n_total_train,
                 n_items_global, baseline_conv)
print(f"  [{time.time()-t1:.1f}s] NDCG@10={m_ens['NDCG@10']:.4f}  "
      f"Coverage={m_ens['Coverage']:.4f}  ConvLift@10={m_ens.get('ConvLift@10',0):.0f}x")
all_results["Ensemble (TDIPS+EASE+BPR)"] = m_ens
"""))

# =============================================================================
# 10 - TABLA COMPARATIVA
# =============================================================================
cells.append(md("## 10 - Comparacion Completa de Modelos (NB08 + NB09)"))

cells.append(py("""\
rows_c = []
for name, m in all_results.items():
    row = {"Modelo": name}
    for k in [10, 20]:
        for met in ["NDCG","Precision","Recall","MAP","Revenue","ConvLift"]:
            row[f"{met}@{k}"] = round(m.get(f"{met}@{k}", 0.), 5)
    row["Coverage"] = round(m.get("Coverage",0.), 4)
    row["Novelty"]  = round(m.get("Novelty",0.), 2)
    row["N_eval"]   = m.get("n_eval", m.get("n_evaluated", 0))
    rows_c.append(row)

df_cmp = pd.DataFrame(rows_c).sort_values("NDCG@10", ascending=False).reset_index(drop=True)
df_cmp.insert(0,"Rank", range(1, len(df_cmp)+1))

print("="*80)
print("TABLA COMPARATIVA COMPLETA (NB08 + NB09)")
print("="*80)
print(df_cmp[["Rank","Modelo","NDCG@10","Precision@10","Coverage","ConvLift@10"]].to_string(index=False))
print()
winner_name = df_cmp.iloc[0]["Modelo"]
winner_ndcg = df_cmp.iloc[0]["NDCG@10"]
nb08_ndcg   = all_results.get("SVD+TD+IPS (NB08)",{}).get("NDCG@10", 0.0093)
print(f"Ganador NB09 : {winner_name}")
print(f"NDCG@10      : {winner_ndcg:.4f}  (NB08 baseline: {nb08_ndcg:.4f})")
print(f"Mejora vs NB08: {(winner_ndcg-nb08_ndcg)/nb08_ndcg*100:+.1f}%")

df_cmp.to_csv(DOCS_DIR / "model_comparison_09_advanced.csv", index=False)
df_cmp.to_csv(DATA_DIR / "model_comparison_09_advanced.csv", index=False)
print("Guardado CSV en docs/ y data/processed/")
"""))

# =============================================================================
# 11 - VISUALIZACIONES
# =============================================================================
cells.append(md("## 11 - Visualizaciones"))

cells.append(py("""\
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
models_p = df_cmp["Modelo"].tolist()
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(models_p)))[::-1]

for ax, col, title in zip(axes,
    ["NDCG@10","Coverage","ConvLift@10"],
    ["NDCG@10","Coverage del Catalogo","Conversion Lift@10"]):
    vals = df_cmp[col].tolist()
    bars = ax.barh(models_p[::-1], vals[::-1], color=colors[::-1])
    ax.set_title(title); ax.set_xlabel(col)
    for b,v in zip(bars, vals[::-1]):
        ax.text(b.get_width()+abs(b.get_width())*0.01, b.get_y()+b.get_height()/2,
                f"{v:.4g}", va="center", fontsize=8)

plt.suptitle("NB09 — Comparativa: NB08 baseline + WRMF + BPR + RP3beta + NCF + SASRec + Ensemble",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(DOCS_DIR / "fig_09_model_comparison_advanced.png", dpi=150, bbox_inches="tight")
plt.show()
print("Guardado: fig_09_model_comparison_advanced.png")
"""))

cells.append(py("""\
# Curva de convergencia BPR
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Loss BPR
if bpr_losses:
    axes[0].plot(range(1, len(bpr_losses)+1), bpr_losses, marker="o", color="#4C72B0")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BPR Loss")
    axes[0].set_title("Convergencia BPR-MF")

# NDCG@10 comparativa top-6
top6 = df_cmp.head(min(6, len(df_cmp)))
axes[1].barh(top6["Modelo"].tolist()[::-1], top6["NDCG@10"].tolist()[::-1],
             color="#55A868")
axes[1].axvline(nb08_ndcg, color="red", linestyle="--", label=f"NB08 baseline: {nb08_ndcg:.4f}")
axes[1].set_title("NDCG@10 — Top Modelos")
axes[1].set_xlabel("NDCG@10"); axes[1].legend()

plt.tight_layout()
plt.savefig(DOCS_DIR / "fig_09_convergence_top.png", dpi=150, bbox_inches="tight")
plt.show()
print("Guardado: fig_09_convergence_top.png")
"""))

# =============================================================================
# 12 - GUARDAR MODELO GANADOR
# =============================================================================
cells.append(md("## 12 - Guardar Artefactos del Modelo Ganador (NB09)"))

cells.append(py("""\
artifact = {
    "model_name":  winner_name,
    "ndcg10":      winner_ndcg,
    "all_metrics": all_results,
    "hyperparams": {
        "EASE_TOP": EASE_TOP, "EASE_REG": EASE_REG,
        "BPR_K": BPR_K, "BPR_LR": BPR_LR, "BPR_EPOCHS": BPR_EPOCHS,
        "NCF_EMB": NCF_EMB if TORCH_AVAILABLE else None,
        "SAS_EMB": SAS_EMB if TORCH_AVAILABLE else None,
        "DECAY_LAMBDA": DECAY_LAMBDA, "IPS_POWER": IPS_POWER,
    },
    # Embeddings SVD+TD+IPS (baseline)
    "U_tdips": U_tdips, "Vt_tdips": Vt_tdips, "s_tdips": s_tdips,
    # EASE^R matrices y subespacio
    "B_ease":           B_ease,
    "X_ease_csr":       X_ease_csr,
    "top_items_global": top_items_global,
    "top_items_idx":    top_items_idx,
    # BPR-MF
    "U_bpr": U_bpr, "Vt_bpr": Vt_bpr,
    # Indexing
    "user2idx": user2idx, "item2idx": item2idx,
    "idx2item": idx2item, "idx2user": idx2user,
    "ensemble_weights": {"tdips": w1_e, "ease": w2_e, "bpr": w3_e},
}

# Agregar modelos deep si disponibles
if TORCH_AVAILABLE and "NCF (NeuMF, PyTorch)" in all_results:
    artifact["ncf_state_dict"]    = model.state_dict()
    artifact["ncf_config"] = {"emb": NCF_EMB, "layers": NCF_LAYERS}
if TORCH_AVAILABLE and "SASRec-lite (Transformer)" in all_results:
    artifact["sasrec_state_dict"] = sas.state_dict()
    artifact["sasrec_config"] = {"emb": SAS_EMB, "max_len": SAS_LEN,
                                 "n_heads": SAS_HEADS, "n_layers": SAS_LAYERS}

out = ENC_DIR / "final_model_v3.pkl"
with open(out, "wb") as f:
    pickle.dump(artifact, f, protocol=4)

import os
print(f"Guardado: {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
print()
print("="*60)
print("RESUMEN FINAL NB09")
print("="*60)
print(f"Modelos evaluados : {len(all_results)}")
print(f"NB08 baseline     : NDCG@10 = {nb08_ndcg:.4f}")
print(f"NB09 winner       : {winner_name}")
print(f"NB09 NDCG@10      : {winner_ndcg:.4f}  ({(winner_ndcg-nb08_ndcg)/nb08_ndcg*100:+.1f}%)")
print()
for row in df_cmp[["Rank","Modelo","NDCG@10","Coverage","ConvLift@10"]].itertuples(index=False):
    print(f"  {row.Rank}. {row.Modelo:45s}  NDCG@10={row._2:.4f}  Cov={row._3:.4f}  Lift={row._4:.0f}x")
"""))

# =============================================================================
# ESCRIBIR NOTEBOOK
# =============================================================================
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.5"},
    },
    "cells": cells,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook generado: {NB_PATH}")
print(f"  Celdas totales : {len(cells)}")
print(f"  Celdas codigo  : {sum(1 for c in cells if c['cell_type']=='code')}")
