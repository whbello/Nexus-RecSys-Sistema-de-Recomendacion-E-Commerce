"""
Script para generar notebooks/13_ensemble_advanced.ipynb usando nbformat.
Ejecutar desde la raíz del proyecto:
    python scripts/build_nb13.py
"""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "notebooks" / "13_ensemble_advanced.ipynb"

nb = nbf.v4.new_notebook()
cells = []

# ── Celda 0: Markdown título ──────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""# NB13 — Ensemble Avanzado: 4 Levers hacia NDCG@10 ≥ 0.030

**Objetivo:** Superar NDCG@10 = 0.026026 (Ensemble RP3opt+EASE, NB11) apuntando a ≥ 0.030 (+15%).

## Auditoría pre-código
| Artefacto | Estado | Notas |
|---|---|---|
| `encoders/rp3beta_optimized.pkl` | ✅ Existe | α=0.75, β=0.30 (Optuna 50 trials NB11) |
| `encoders/final_model_v4.pkl` | ✅ Existe | MultiVAE state_dict + top_items_global |
| `encoders/ease_optimized.pkl` | ❌ Pendiente | Se crea en Lever A |
| Score matrices | ❌ No guardadas | Se recalculan por usuario on-the-fly |
| `data/processed/interaction_matrix.csv` | ✅ Existe | Fuente canónica |

## Plan de 4 levers
- **Lever A:** EASE^R λ Optuna (λ=500 nunca fue tuneado; precomputa G una sola vez)
- **Lever B:** Ensemble 3-way (RP3opt + EASE-tuned + MultiVAE) con Optuna de pesos
- **Lever C:** Temporal Decay sobre RP3beta (ponderar interacciones por recencia)
- **Lever D:** Ensemble final con mejor componente RP3

**Disciplina de evaluación:**
- `val_users` (~450, 15% estratificado) → hyperparameter search  
- `test_users_b` (~2550, 85%) → evaluación final UNA SOLA VEZ por modelo
- Split idéntico al de NB11 (seed=42, grupos cold/tepid/warm)"""))

# ── Celda 1: Imports y constantes ─────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""import sys
import time
import math
import pickle
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from sklearn.preprocessing import normalize as skl_normalize

import torch
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Rutas ────────────────────────────────────────────────────────────────────
HERE     = Path().resolve()
ROOT     = HERE.parent if (HERE.parent / "data").exists() else HERE
DATA_DIR = ROOT / "data" / "processed"
ENC_DIR  = ROOT / "encoders"
DOCS_DIR = ROOT / "docs"
SCRIPTS  = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
DOCS_DIR.mkdir(exist_ok=True)

# ── Constantes globales (idénticas a NB11) ───────────────────────────────────
RANDOM_STATE  = 42
K_VALUES      = [5, 10, 20]
N_EVAL_USERS  = 3_000
CUTOFF_DATE   = pd.Timestamp("2015-08-22", tz="UTC")
EASE_TOP      = 20_000
EASE_REG_BASE = 500.0      # λ original NB11 (nunca optimizado)

# Parámetros RP3beta óptimos de NB11
RP3_ALPHA_OPT = 0.75
RP3_BETA_OPT  = 0.30

# Optuna trials por lever
N_TRIALS_EASE     = 25   # Lever A — EASE λ
N_TRIALS_ENSEMBLE = 50   # Lever B — 3-way ensemble
N_TRIALS_FINAL    = 30   # Lever D — ensemble final

# NDCG@10 de referencia NB11
NDCG_NB11_ENS2 = 0.026026
TARGET_NDCG    = 0.030

print(f"Root     : {ROOT}")
print(f"Optuna   : {optuna.__version__}")
print(f"PyTorch  : {torch.__version__}")
print(f"EASE_TOP : {EASE_TOP}  lambda_base: {EASE_REG_BASE}")
print(f"RP3opt   : alpha={RP3_ALPHA_OPT}  beta={RP3_BETA_OPT}")"""))

# ── Celda 2: Markdown sección 1 ───────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 1 — Carga de datos, split temporal y val/test split (idéntico a NB11)"))

# ── Celda 3: Carga de datos ───────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""print("Cargando interaction_matrix.csv...")
t0 = time.time()
im  = pd.read_csv(DATA_DIR / "interaction_matrix.csv")
im["last_interaction_ts"] = pd.to_datetime(im["last_interaction_ts"], format="ISO8601", utc=True)
print(f"  IM: {im.shape}  [{time.time()-t0:.1f}s]")

# ── Split temporal (idéntico a NB09/NB11) ────────────────────────────────────
train_mask = im["last_interaction_ts"] < CUTOFF_DATE
train_df   = im[train_mask].copy()
test_df    = im[~train_mask].copy()

# Usuarios warm: tienen al menos 1 ítem en train Y 1 en test
warm_users = sorted(set(train_df["visitorid"].unique()) & set(test_df["visitorid"].unique()))

# Muestrear N_EVAL_USERS (seed=42, idéntico a NB09/NB11)
rng = np.random.default_rng(RANDOM_STATE)
eval_users = rng.choice(warm_users, size=min(N_EVAL_USERS, len(warm_users)), replace=False).tolist()

# Diccionarios de test
test_items_by_user  = test_df.groupby("visitorid")["itemid"].apply(set).to_dict()
train_items_by_user = train_df.groupby("visitorid")["itemid"].apply(set).to_dict()
test_tx_by_user     = (
    test_df[test_df["last_interaction_type"] == "transaction"]
    .groupby("visitorid")["itemid"].apply(set).to_dict()
)

all_items_global = sorted(im["itemid"].unique())
n_items_global   = len(all_items_global)
n_test_tx        = len(test_df[test_df["last_interaction_type"] == "transaction"])
baseline_conv    = n_test_tx / (len(set(test_df["visitorid"])) * n_items_global)

print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")
print(f"Warm users total: {len(warm_users):,}")
print(f"Eval users:       {len(eval_users):,}")

# ── Val/Test split estratificado (IDÉNTICO a NB11) ───────────────────────────
activity_groups = []
for u in eval_users:
    cnt = len(train_items_by_user.get(u, set()))
    if   cnt == 1: activity_groups.append(0)   # cold
    elif cnt <= 4: activity_groups.append(1)   # tepid
    else:          activity_groups.append(2)   # warm
activity_groups = np.array(activity_groups)

rng_split = np.random.default_rng(RANDOM_STATE)
val_mask  = np.zeros(len(eval_users), dtype=bool)
for g in [0, 1, 2]:
    idx_g = np.where(activity_groups == g)[0]
    if len(idx_g) == 0:
        continue
    n_val  = max(1, int(len(idx_g) * 0.15))
    chosen = rng_split.choice(idx_g, size=n_val, replace=False)
    val_mask[chosen] = True

eval_arr     = np.array(eval_users)
val_users    = eval_arr[val_mask].tolist()       # ~450 usuarios (15%)
test_users_b = eval_arr[~val_mask].tolist()      # ~2550 usuarios (85%)

print(f"val_users    : {len(val_users):,}  (15% estratificado)")
print(f"test_users_b : {len(test_users_b):,}  (85% test definitivo)")
cnt_cold  = int((activity_groups == 0).sum())
cnt_tepid = int((activity_groups == 1).sum())
cnt_warm  = int((activity_groups == 2).sum())
print(f"Grupos — cold:{cnt_cold}  tepid:{cnt_tepid}  warm:{cnt_warm}")"""))

# ── Celda 4: Markdown sección 2 ───────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 2 — Construcción de matrices y funciones base"))

# ── Celda 5: Construcción de matrices ─────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# ── Construir R (CSR) sobre train ─────────────────────────────────────────────
all_train_users = sorted(train_df["visitorid"].unique())
all_train_items = sorted(train_df["itemid"].unique())
user2idx = {u: i for i, u in enumerate(all_train_users)}
item2idx = {it: i for i, it in enumerate(all_train_items)}
idx2item = {i: it for it, i in item2idx.items()}
n_u = len(all_train_users)
n_i = len(all_train_items)

rows_r = train_df["visitorid"].map(user2idx).values
cols_r = train_df["itemid"].map(item2idx).values
vals_r = train_df["interaction_strength"].values.astype(np.float32)
R = sp.csr_matrix((vals_r, (rows_r, cols_r)), shape=(n_u, n_i), dtype=np.float32)

item_pop      = np.asarray(R.sum(axis=0)).ravel()
item_pop_dict = {idx2item[i]: float(item_pop[i]) for i in range(n_i)}
n_total_train = float(R.sum())

print(f"R: {R.shape}  nnz={R.nnz:,}")

# ── Subespacio top-20K ítems (idéntico a NB11) ────────────────────────────────
top_items_idx = np.argpartition(item_pop, -EASE_TOP)[-EASE_TOP:]
top_items_idx = top_items_idx[np.argsort(item_pop[top_items_idx])[::-1]]
N_TOP         = len(top_items_idx)
top_items_global = [idx2item[i] for i in top_items_idx]

X_top_csr = R[:, top_items_idx].astype(np.float32).tocsr()  # (n_u, N_TOP)
pop_sub   = item_pop[top_items_idx].astype(np.float32)

print(f"X_top_csr: {X_top_csr.shape}  nnz={X_top_csr.nnz:,}")
print(f"pop_sub  : min={pop_sub.min():.0f}  max={pop_sub.max():.0f}")"""))

# ── Celda 6: Funciones de evaluación y modelos ────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# ── Framework de evaluación (idéntico a NB11) ─────────────────────────────────
def ndcg(r, rel, k):
    r = r[:k]; dcg = sum((1/math.log2(i+2)) for i,x in enumerate(r) if x in rel)
    idcg = sum(1/math.log2(i+2) for i in range(min(len(rel), k))); return dcg/idcg if idcg else 0.
def prec(r, rel, k):  return len(set(r[:k]) & rel) / k if k else 0.
def rec(r, rel, k):   return len(set(r[:k]) & rel) / len(rel) if rel else 0.
def ap(r, rel, k):
    if not rel: return 0.
    s, h = 0., 0
    for i, x in enumerate(r[:k]):
        if x in rel: h += 1; s += h / (i + 1)
    return s / min(len(rel), k)
def nov(flat, pd_, nt):
    return float(np.mean([-math.log2(pd_.get(x, 1) / nt + 1e-10) for x in flat])) if flat else 0.
def rev_k(r, tx, k): return len(set(r[:k]) & tx) / k if k else 0.
def ctr_k(r, ts, k): return len(set(r[:k]) & ts) / k if k else 0.

def evaluate(get_fn, evals, tst, tst_tx, pop_d, nt, cat_sz, bconv, ks=K_VALUES):
    acc = {k: {m: [] for m in "prnm"} for k in ks}
    for k in ks: acc[k]["r2"] = []; acc[k]["c"] = []
    seen = set(); ne = 0
    for uid in evals:
        ti = tst.get(uid, set())
        if not ti: continue
        tx = tst_tx.get(uid, set())
        mk = max(ks)
        try: recs = get_fn(uid, mk)
        except Exception: continue
        seen.update(recs); ne += 1
        for k in ks:
            acc[k]["p"].append(prec(recs, ti, k))
            acc[k]["r"].append(rec(recs,  ti, k))
            acc[k]["n"].append(ndcg(recs,  ti, k))
            acc[k]["m"].append(ap(recs,   ti, k))
            acc[k]["r2"].append(rev_k(recs, tx, k))
            acc[k]["c"].append(ctr_k(recs, ti, k))
    out = {"n_eval": ne}
    for k in ks:
        if not acc[k]["p"]: continue
        out[f"NDCG@{k}"]      = float(np.mean(acc[k]["n"]))
        out[f"Precision@{k}"] = float(np.mean(acc[k]["p"]))
        out[f"Recall@{k}"]    = float(np.mean(acc[k]["r"]))
        out[f"MAP@{k}"]       = float(np.mean(acc[k]["m"]))
        out[f"Revenue@{k}"]   = float(np.mean(acc[k]["r2"]))
        out[f"CTR@{k}"]       = float(np.mean(acc[k]["c"]))
        rv = out[f"Revenue@{k}"]
        out[f"ConvLift@{k}"]  = rv / bconv if bconv else 0.
    out["Coverage"] = len(seen) / cat_sz
    out["Novelty"]  = nov(list(seen), pop_d, nt)
    return out

# ── RP3beta builder ───────────────────────────────────────────────────────────
def build_rp3(alpha, beta, X_csr, pop_arr):
    P_ui = skl_normalize(X_csr.astype(np.float64), norm="l1", axis=1)
    P_it = skl_normalize(X_csr.T.tocsr().astype(np.float64), norm="l1", axis=1)
    pop_beta = np.power(pop_arr + 1e-10, beta)
    W_raw = P_it @ P_ui
    W = np.asarray(W_raw.todense() if hasattr(W_raw, "todense") else W_raw, dtype=np.float32)
    del W_raw
    np.power(W, alpha, out=W)
    W /= pop_beta[:, np.newaxis]
    np.fill_diagonal(W, 0.)
    return W

def make_get_rp3(W, X_csr, top_items_list):
    def get_fn(uid, n):
        if uid not in user2idx: return []
        ui  = user2idx[uid]
        row = X_csr.getrow(ui)
        x_u = np.asarray(row.todense(), dtype=np.float32).ravel()
        sc  = x_u @ W
        sc[x_u > 0] = -1.
        top = np.argpartition(sc, -n)[-n:]
        top = top[np.argsort(sc[top])[::-1]]
        return [top_items_list[i] for i in top]
    return get_fn

def minmax_norm(v):
    vmin, vmax = v.min(), v.max()
    rng = vmax - vmin
    return (v - vmin) / rng if rng > 1e-12 else np.zeros_like(v)

print("Funciones definidas: evaluate, build_rp3, make_get_rp3, minmax_norm")"""))

# ── Celda 7: Markdown sección 3 ───────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## 3 — Carga de modelos existentes y baseline"))

# ── Celda 8: Cargar RP3 ───────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# ── Cargar RP3beta optimizado (NB11) ─────────────────────────────────────────
print("Cargando rp3beta_optimized.pkl...")
t0 = time.time()
with open(ENC_DIR / "rp3beta_optimized.pkl", "rb") as f:
    art_rp3 = pickle.load(f)

W_rp3opt = art_rp3["W_rp3"]            # (N_TOP, N_TOP) float32
assert W_rp3opt.shape == (N_TOP, N_TOP), f"Shape mismatch: {W_rp3opt.shape}"
get_rp3opt = make_get_rp3(W_rp3opt, X_top_csr, top_items_global)
print(f"  RP3opt cargado  alpha={art_rp3['alpha']:.2f}  beta={art_rp3['beta']:.2f}  "
      f"NDCG@10_test={art_rp3['ndcg10_test']:.5f}  [{time.time()-t0:.1f}s]")"""))

# ── Celda 9: Cargar MultiVAE ──────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# ── Cargar MultiVAE (NB10) ────────────────────────────────────────────────────
from multivae_model import MultiVAE, build_scorer

print("Cargando final_model_v4.pkl...")
t0 = time.time()
with open(ENC_DIR / "final_model_v4.pkl", "rb") as f:
    art_vae = pickle.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")

mvae = MultiVAE(
    n_items      = art_vae["n_items"],
    enc_dims     = art_vae["enc_dims"],
    latent_dim   = art_vae["latent_dim"],
    dropout_rate = art_vae["dropout_rate"],
).to(device)
mvae.load_state_dict(art_vae["model_state"])
mvae.eval()

get_multivae = build_scorer(mvae, X_top_csr, user2idx, top_items_global, device=device)
ndcg_vae_saved = art_vae.get("ndcg10", art_vae.get("ndcg@10", "N/A"))
print(f"  MultiVAE cargado  n_items={art_vae['n_items']}  "
      f"enc_dims={art_vae['enc_dims']}  ldim={art_vae['latent_dim']}  "
      f"NDCG@10_guardado={ndcg_vae_saved}  [{time.time()-t0:.1f}s]")"""))

# ── Celda 10: EASE baseline ───────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# ── Construir EASE^R baseline (lambda=500) y precomputar G ──────────────────
print(f"Construyendo EASE^R baseline (lambda={EASE_REG_BASE}) y precomputando G...")
t0 = time.time()
G_sparse = X_top_csr.T @ X_top_csr
G_dense  = np.asarray(G_sparse.todense(), dtype=np.float32)
del G_sparse
print(f"  G_dense: {G_dense.shape}  [{time.time()-t0:.1f}s]")

# Guardar G para Lever A (evitar recomputo)
G_for_optuna = G_dense.copy()
diag_idx     = np.arange(N_TOP)

t1 = time.time()
G_reg_base = G_dense + EASE_REG_BASE * np.eye(N_TOP, dtype=np.float32)
B_inv_base = np.linalg.inv(G_reg_base)
del G_reg_base
diag_inv_base = np.diag(B_inv_base).copy()
B_ease_base   = -(B_inv_base / diag_inv_base[np.newaxis, :]).astype(np.float32)
np.fill_diagonal(B_ease_base, 0.)
del B_inv_base
print(f"  EASE^R(lambda={EASE_REG_BASE}) construido  [{time.time()-t1:.1f}s]")

def make_get_ease(B):
    def get_fn(uid, n):
        if uid not in user2idx: return []
        ui  = user2idx[uid]
        row = X_top_csr.getrow(ui)
        x_u = np.asarray(row.todense(), dtype=np.float32).ravel()
        sc  = x_u @ B
        sc[x_u > 0] = -1.
        top = np.argpartition(sc, -n)[-n:]
        top = top[np.argsort(sc[top])[::-1]]
        return [top_items_global[i] for i in top]
    return get_fn

get_ease_base = make_get_ease(B_ease_base)
print("EASE^R baseline listo.")"""))

# ── Celda 11: Baseline eval ───────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# ── Evaluación baseline sobre val_users ──────────────────────────────────────
print(f"Evaluando modelos base sobre {len(val_users)} val_users...")

t0 = time.time()
m_rp3_val = evaluate(
    get_rp3opt, val_users,
    test_items_by_user, test_tx_by_user,
    item_pop_dict, n_total_train, n_items_global, baseline_conv
)
print(f"  RP3opt    NDCG@10={m_rp3_val['NDCG@10']:.5f}  [{time.time()-t0:.1f}s]")

t0 = time.time()
m_ease_val = evaluate(
    get_ease_base, val_users,
    test_items_by_user, test_tx_by_user,
    item_pop_dict, n_total_train, n_items_global, baseline_conv
)
print(f"  EASE(500) NDCG@10={m_ease_val['NDCG@10']:.5f}  [{time.time()-t0:.1f}s]")

t0 = time.time()
m_vae_val = evaluate(
    get_multivae, val_users,
    test_items_by_user, test_tx_by_user,
    item_pop_dict, n_total_train, n_items_global, baseline_conv
)
print(f"  MultiVAE  NDCG@10={m_vae_val['NDCG@10']:.5f}  [{time.time()-t0:.1f}s]")

ndcg_rp3_val  = m_rp3_val["NDCG@10"]
ndcg_ease_val = m_ease_val["NDCG@10"]
ndcg_vae_val  = m_vae_val["NDCG@10"]
print(f"\\nBaseline val — RP3:{ndcg_rp3_val:.5f}  EASE:{ndcg_ease_val:.5f}  VAE:{ndcg_vae_val:.5f}")"""))

# ── Celda 12: Markdown Lever A ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Lever A — EASE^R λ Optuna

El λ=500 de NB11 nunca fue optimizado para este dataset.  
**Truco de eficiencia:** G = X^T @ X ya está precomputado; cada trial sólo recomputa inv(G + λI)."""))

# ── Celda 13: Lever A Optuna ──────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""print("=" * 60)
print(f"LEVER A — EASE^R lambda Optuna ({N_TRIALS_EASE} trials)")
print(f"  G precomputado: {G_for_optuna.shape}  search lambda en [10, 5000] log-scale")
print(f"  Evaluacion sobre {len(val_users)} val_users")
print("=" * 60)

def objective_ease(trial):
    lam = trial.suggest_float("lambda", 10.0, 5000.0, log=True)
    G_reg = G_for_optuna.copy()
    G_reg[diag_idx, diag_idx] += lam
    B_inv_ = np.linalg.inv(G_reg)
    del G_reg
    d_inv = np.diag(B_inv_).copy()
    B = -(B_inv_ / d_inv[np.newaxis, :]).astype(np.float32)
    np.fill_diagonal(B, 0.)
    del B_inv_
    get_fn = make_get_ease(B)
    m = evaluate(
        get_fn, val_users,
        test_items_by_user, test_tx_by_user,
        item_pop_dict, n_total_train, n_items_global, baseline_conv
    )
    return m.get("NDCG@10", 0.)

t_opt = time.time()
study_ease = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
)
study_ease.optimize(objective_ease, n_trials=N_TRIALS_EASE, show_progress_bar=True)
print(f"\\nOptuna EASE finalizado en {time.time()-t_opt:.0f}s")

trials_ease_df = pd.DataFrame([
    {"Rank": 0, "lambda": t.params["lambda"], "NDCG@10 val": t.value}
    for t in study_ease.trials if t.value is not None
]).sort_values("NDCG@10 val", ascending=False).reset_index(drop=True)
trials_ease_df["Rank"] = range(1, len(trials_ease_df) + 1)
trials_ease_df["lambda"]      = trials_ease_df["lambda"].round(1)
trials_ease_df["NDCG@10 val"] = trials_ease_df["NDCG@10 val"].round(6)
print("\\nTop 10 trials EASE^R lambda:")
print(trials_ease_df.head(10).to_string(index=False))

best_lambda    = study_ease.best_params["lambda"]
best_ease_val  = study_ease.best_value
delta_ease     = (best_ease_val - ndcg_ease_val) / ndcg_ease_val * 100
sign = "+" if delta_ease >= 0 else ""
print(f"\\nlambda optimo: {best_lambda:.1f}  (lambda_base={EASE_REG_BASE})")
print(f"NDCG@10 val: {ndcg_ease_val:.5f} -> {best_ease_val:.5f}  ({sign}{delta_ease:.1f}%)")"""))

# ── Celda 14: Lever A test ────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# ── Lever A — Evaluacion FINAL sobre test (UNA SOLA VEZ) ─────────────────────
print(f"Construyendo EASE^R(lambda={best_lambda:.1f})...")
t0 = time.time()
G_reg_opt = G_for_optuna.copy()
G_reg_opt[diag_idx, diag_idx] += best_lambda
B_inv_opt  = np.linalg.inv(G_reg_opt)
del G_reg_opt
d_inv_opt  = np.diag(B_inv_opt).copy()
B_ease_opt = -(B_inv_opt / d_inv_opt[np.newaxis, :]).astype(np.float32)
np.fill_diagonal(B_ease_opt, 0.)
del B_inv_opt
get_ease_opt = make_get_ease(B_ease_opt)
print(f"  Construido en {time.time()-t0:.1f}s")

print(f"Evaluando EASE^R(lambda={best_lambda:.1f}) sobre test ({len(test_users_b)})...")
t0 = time.time()
m_ease_opt_test = evaluate(
    get_ease_opt, test_users_b,
    test_items_by_user, test_tx_by_user,
    item_pop_dict, n_total_train, n_items_global, baseline_conv
)
print(f"  [{time.time()-t0:.1f}s]")

print(f"Evaluando EASE^R(lambda={EASE_REG_BASE}) sobre test ({len(test_users_b)})...")
t0 = time.time()
m_ease_base_test = evaluate(
    get_ease_base, test_users_b,
    test_items_by_user, test_tx_by_user,
    item_pop_dict, n_total_train, n_items_global, baseline_conv
)
print(f"  [{time.time()-t0:.1f}s]")

ndcg_ease_base_test = m_ease_base_test["NDCG@10"]
ndcg_ease_opt_test  = m_ease_opt_test["NDCG@10"]
delta_a = (ndcg_ease_opt_test - ndcg_ease_base_test) / ndcg_ease_base_test * 100
sign = "+" if delta_a >= 0 else ""
print(f"\\nLEVER A — Resultados test:")
print(f"  EASE(lambda={EASE_REG_BASE}) : NDCG@10={ndcg_ease_base_test:.5f}")
print(f"  EASE(lambda={best_lambda:.1f}) : NDCG@10={ndcg_ease_opt_test:.5f}  ({sign}{delta_a:.1f}%)")

# Guardar ease_optimized.pkl
import os
artifact_ease = {
    "model_name":        f"EASE^R optimizado (lambda={best_lambda:.1f})",
    "lambda":             best_lambda,
    "ndcg10_test":        ndcg_ease_opt_test,
    "ndcg10_val":         best_ease_val,
    "B_ease":             B_ease_opt,
    "top_items_global":   top_items_global,
    "top_items_idx":      top_items_idx,
    "user2idx":           user2idx,
    "optuna_best_params": study_ease.best_params,
    "optuna_n_trials":    N_TRIALS_EASE,
}
ease_out = ENC_DIR / "ease_optimized.pkl"
with open(ease_out, "wb") as f:
    pickle.dump(artifact_ease, f, protocol=4)
print(f"Guardado: {ease_out}  ({os.path.getsize(ease_out)/1e6:.1f} MB)")"""))

# ── Celda 15: Markdown Lever B ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Lever B — Ensemble 3-way (RP3opt + EASE-tuned + MultiVAE)

Optuna sobre pesos w1 (RP3), w2 (EASE), w3=1−w1−w2 (MultiVAE),
con restricción w1+w2 ≤ 1 y todos ≥ 0."""))

# ── Celda 16: Lever B Optuna ──────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""print("=" * 60)
print(f"LEVER B — Ensemble 3-way Optuna ({N_TRIALS_ENSEMBLE} trials)")
print(f"  Modelos: RP3opt | EASE(lambda={best_lambda:.1f}) | MultiVAE")
print(f"  Evaluacion sobre {len(val_users)} val_users")
print("=" * 60)

def make_3way_ensemble(w1, w2, W_rp3_, B_ease_, X_rp3_):
    w3 = 1.0 - w1 - w2
    def get_fn(uid, n):
        if uid not in user2idx: return []
        ui   = user2idx[uid]
        row  = X_rp3_.getrow(ui)
        x_rp3 = np.asarray(row.todense(), dtype=np.float32).ravel()
        row_e = X_top_csr.getrow(ui)
        x_ease = np.asarray(row_e.todense(), dtype=np.float32).ravel()

        sc_rp3  = minmax_norm(x_rp3  @ W_rp3_)
        sc_ease = minmax_norm(x_ease @ B_ease_)

        x_bin = (x_ease > 0).astype(np.float32)
        x_t   = torch.from_numpy(x_bin).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _, _ = mvae(x_t)
        sc_vae = minmax_norm(logits.squeeze(0).cpu().numpy())

        sc = w1 * sc_rp3 + w2 * sc_ease + w3 * sc_vae
        seen_mask = (x_rp3 > 0) | (x_ease > 0)
        sc[seen_mask] = -1.
        top = np.argpartition(sc, -n)[-n:]
        top = top[np.argsort(sc[top])[::-1]]
        return [top_items_global[i] for i in top]
    return get_fn

def objective_ensemble(trial):
    w1 = trial.suggest_float("w_rp3",  0.30, 0.80)
    w2 = trial.suggest_float("w_ease", 0.05, 0.40)
    if w1 + w2 > 1.0:
        raise optuna.exceptions.TrialPruned()
    get_fn = make_3way_ensemble(w1, w2, W_rp3opt, B_ease_opt, X_top_csr)
    m = evaluate(
        get_fn, val_users,
        test_items_by_user, test_tx_by_user,
        item_pop_dict, n_total_train, n_items_global, baseline_conv
    )
    return m.get("NDCG@10", 0.)

t_opt = time.time()
study_ens = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
)
study_ens.optimize(objective_ensemble, n_trials=N_TRIALS_ENSEMBLE, show_progress_bar=True)
print(f"\\nOptuna ensemble finalizado en {time.time()-t_opt:.0f}s")

trials_ens_df = pd.DataFrame([
    {"Rank": 0,
     "w_rp3":  round(t.params["w_rp3"], 3),
     "w_ease": round(t.params["w_ease"], 3),
     "w_vae":  round(1 - t.params["w_rp3"] - t.params["w_ease"], 3),
     "NDCG@10 val": round(t.value, 6)}
    for t in study_ens.trials if t.value is not None and t.state.name == "COMPLETE"
]).sort_values("NDCG@10 val", ascending=False).reset_index(drop=True)
trials_ens_df["Rank"] = range(1, len(trials_ens_df) + 1)
print("Top 10 trials ensemble 3-way:")
print(trials_ens_df.head(10).to_string(index=False))

best_w1      = study_ens.best_params["w_rp3"]
best_w2      = study_ens.best_params["w_ease"]
best_w3      = 1.0 - best_w1 - best_w2
best_ens_val = study_ens.best_value
print(f"\\nPesos optimos: w_rp3={best_w1:.3f}  w_ease={best_w2:.3f}  w_vae={best_w3:.3f}")
print(f"NDCG@10 val: {best_ens_val:.5f}")"""))

# ── Celda 17: Lever B test ────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# ── Lever B — Evaluacion FINAL en test (UNA SOLA VEZ) ────────────────────────
print(f"Evaluando ensemble 3-way en test ({len(test_users_b)})...")
t0 = time.time()
get_ens3 = make_3way_ensemble(best_w1, best_w2, W_rp3opt, B_ease_opt, X_top_csr)
m_ens3_test = evaluate(
    get_ens3, test_users_b,
    test_items_by_user, test_tx_by_user,
    item_pop_dict, n_total_train, n_items_global, baseline_conv
)
print(f"  [{time.time()-t0:.1f}s]")

ndcg_ens3_test = m_ens3_test["NDCG@10"]
delta_b = (ndcg_ens3_test - NDCG_NB11_ENS2) / NDCG_NB11_ENS2 * 100
sign = "+" if delta_b >= 0 else ""
print(f"\\nLEVER B — Resultados test:")
print(f"  Ensemble 3-way  : NDCG@10={ndcg_ens3_test:.5f}")
print(f"  NB11 Ens2 (ref) : NDCG@10={NDCG_NB11_ENS2:.5f}")
print(f"  Mejora          : {sign}{delta_b:.1f}%")"""))

# ── Celda 18: Markdown Lever C ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""## Lever C — Temporal Decay sobre RP3beta

Ponderar interacciones por **recencia**: `w_i = exp(-decay_rate × dias_antes_cutoff)`.  
Grilla de 6 valores de `decay_rate` en [0.0001, 0.05]. Reconstruye R_td → X_top_td → RP3beta(α,β)."""))

# ── Celda 19: Lever C grilla ──────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""print("=" * 60)
print("LEVER C — Temporal Decay RP3beta")
print(f"  alpha={RP3_ALPHA_OPT}  beta={RP3_BETA_OPT}  (optimos de NB11)")
print("=" * 60)

# Calcular dias al cutoff
train_df_td = train_df.copy()
train_df_td["days_before"] = (
    (CUTOFF_DATE - train_df_td["last_interaction_ts"]).dt.total_seconds() / 86400.0
).clip(lower=0.0)

# Precomputar mapeos para R_td
rows_td_arr = train_df_td["visitorid"].map(user2idx).values
cols_td_arr = train_df_td["itemid"].map(item2idx).values
vals_base   = train_df_td["interaction_strength"].values.astype(np.float32)
days_arr    = train_df_td["days_before"].values.astype(np.float32)

decay_rates  = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
results_decay = []

print(f"  {'decay_rate':>12}  {'NDCG@10 val':>12}  {'vs RP3opt%':>12}  {'t(s)':>6}")
print("-" * 50)

for dr in decay_rates:
    t0 = time.time()
    decay_weights = np.exp(-dr * days_arr)
    vals_td = vals_base * decay_weights
    R_td = sp.csr_matrix((vals_td, (rows_td_arr, cols_td_arr)),
                          shape=(n_u, n_i), dtype=np.float32)
    X_top_td = R_td[:, top_items_idx].astype(np.float32).tocsr()
    pop_td   = np.asarray(R_td.sum(axis=0)).ravel()[top_items_idx].astype(np.float32)
    W_td     = build_rp3(RP3_ALPHA_OPT, RP3_BETA_OPT, X_top_td, pop_td)
    get_rp3_td = make_get_rp3(W_td, X_top_td, top_items_global)
    m = evaluate(
        get_rp3_td, val_users,
        test_items_by_user, test_tx_by_user,
        item_pop_dict, n_total_train, n_items_global, baseline_conv
    )
    elapsed  = time.time() - t0
    ndcg_td  = m.get("NDCG@10", 0.)
    delta_vs = (ndcg_td - ndcg_rp3_val) / ndcg_rp3_val * 100
    results_decay.append({
        "decay_rate":  dr,
        "NDCG@10 val": ndcg_td,
        "delta_%":     round(delta_vs, 2),
        "W_td":        W_td,
        "X_top_td":    X_top_td,
        "time_s":      round(elapsed, 1),
    })
    sign = "+" if delta_vs >= 0 else ""
    print(f"  {dr:12.4f}  {ndcg_td:12.5f}  {sign}{delta_vs:>10.1f}%  {elapsed:5.0f}s")

df_decay = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ("W_td", "X_top_td")}
    for r in results_decay
])
best_decay_row  = df_decay.loc[df_decay["NDCG@10 val"].idxmax()]
best_decay_rate = float(best_decay_row["decay_rate"])
best_decay_val  = float(best_decay_row["NDCG@10 val"])
print(f"\\nMejor decay_rate: {best_decay_rate}  NDCG@10 val={best_decay_val:.5f}")
print(f"RP3opt sin decay : {ndcg_rp3_val:.5f}  -> mejora: {best_decay_val - ndcg_rp3_val:+.5f}")"""))

# ── Celda 20: Lever C test ────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# ── Lever C — Evaluacion FINAL en test (UNA SOLA VEZ) ───────────────────────
best_td_entry = next(r for r in results_decay if r["decay_rate"] == best_decay_rate)
W_best_td      = best_td_entry["W_td"]
X_top_td_best  = best_td_entry["X_top_td"]
get_rp3_td_best = make_get_rp3(W_best_td, X_top_td_best, top_items_global)

print(f"Evaluando RP3+TD(decay={best_decay_rate}) en test ({len(test_users_b)})...")
t0 = time.time()
m_rp3_td_test = evaluate(
    get_rp3_td_best, test_users_b,
    test_items_by_user, test_tx_by_user,
    item_pop_dict, n_total_train, n_items_global, baseline_conv
)
print(f"  [{time.time()-t0:.1f}s]")

print(f"Evaluando RP3opt (sin TD) en test ({len(test_users_b)})...")
t0 = time.time()
m_rp3opt_test = evaluate(
    get_rp3opt, test_users_b,
    test_items_by_user, test_tx_by_user,
    item_pop_dict, n_total_train, n_items_global, baseline_conv
)
print(f"  [{time.time()-t0:.1f}s]")

ndcg_rp3opt_test  = m_rp3opt_test["NDCG@10"]
ndcg_rp3_td_test  = m_rp3_td_test["NDCG@10"]
delta_c = (ndcg_rp3_td_test - ndcg_rp3opt_test) / ndcg_rp3opt_test * 100
sign = "+" if delta_c >= 0 else ""

print(f"\\nLEVER C — Resultados test:")
print(f"  RP3opt sin TD      : NDCG@10={ndcg_rp3opt_test:.5f}")
print(f"  RP3+TD({best_decay_rate}) : NDCG@10={ndcg_rp3_td_test:.5f}  ({sign}{delta_c:.1f}%)")

# Seleccionar mejor componente RP3 para Lever D
use_td_in_final = ndcg_rp3_td_test > ndcg_rp3opt_test
if use_td_in_final:
    W_rp3_final = W_best_td
    X_rp3_final = X_top_td_best
    rp3_label   = f"RP3+TD(d={best_decay_rate})"
    ndcg_rp3_final_test = ndcg_rp3_td_test
else:
    W_rp3_final = W_rp3opt
    X_rp3_final = X_top_csr
    rp3_label   = "RP3opt"
    ndcg_rp3_final_test = ndcg_rp3opt_test
print(f"\\n-> Componente RP3 para Lever D: {rp3_label}  NDCG@10={ndcg_rp3_final_test:.5f}")"""))

# ── Celda 21: Markdown Lever D ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## Lever D — Ensemble final con mejor componente RP3"))

# ── Celda 22: Lever D Optuna ──────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""print("=" * 60)
print(f"LEVER D — Ensemble final ({N_TRIALS_FINAL} trials)")
print(f"  Componente RP3: {rp3_label}")
print("=" * 60)

def objective_final(trial):
    w1 = trial.suggest_float("w_rp3",  0.20, 0.80)
    w2 = trial.suggest_float("w_ease", 0.05, 0.50)
    if w1 + w2 > 1.0:
        raise optuna.exceptions.TrialPruned()
    get_fn = make_3way_ensemble(w1, w2, W_rp3_final, B_ease_opt, X_rp3_final)
    m = evaluate(
        get_fn, val_users,
        test_items_by_user, test_tx_by_user,
        item_pop_dict, n_total_train, n_items_global, baseline_conv
    )
    return m.get("NDCG@10", 0.)

t_opt = time.time()
study_final = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
)
study_final.optimize(objective_final, n_trials=N_TRIALS_FINAL, show_progress_bar=True)
print(f"\\nOptuna final finalizado en {time.time()-t_opt:.0f}s")

trials_final_df = pd.DataFrame([
    {"Rank": 0,
     "w_rp3":  round(t.params["w_rp3"], 3),
     "w_ease": round(t.params["w_ease"], 3),
     "w_vae":  round(1 - t.params["w_rp3"] - t.params["w_ease"], 3),
     "NDCG@10 val": round(t.value, 6)}
    for t in study_final.trials if t.value is not None and t.state.name == "COMPLETE"
]).sort_values("NDCG@10 val", ascending=False).reset_index(drop=True)
trials_final_df["Rank"] = range(1, len(trials_final_df) + 1)
print("Top 10 trials ensemble final:")
print(trials_final_df.head(10).to_string(index=False))

final_w1   = study_final.best_params["w_rp3"]
final_w2   = study_final.best_params["w_ease"]
final_w3   = 1.0 - final_w1 - final_w2
final_val  = study_final.best_value
print(f"\\nPesos optimos finales: w_rp3={final_w1:.3f}  w_ease={final_w2:.3f}  w_vae={final_w3:.3f}")
print(f"NDCG@10 val: {final_val:.5f}")"""))

# ── Celda 23: Lever D test ────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# ── Lever D — Evaluacion FINAL en test (UNA SOLA VEZ) ────────────────────────
print(f"Evaluando ensemble FINAL en test ({len(test_users_b)})...")
t0 = time.time()
get_final_ens = make_3way_ensemble(final_w1, final_w2, W_rp3_final, B_ease_opt, X_rp3_final)
m_final_test = evaluate(
    get_final_ens, test_users_b,
    test_items_by_user, test_tx_by_user,
    item_pop_dict, n_total_train, n_items_global, baseline_conv
)
print(f"  [{time.time()-t0:.1f}s]")

ndcg_final_test = m_final_test["NDCG@10"]
delta_d = (ndcg_final_test - NDCG_NB11_ENS2) / NDCG_NB11_ENS2 * 100
sign = "+" if delta_d >= 0 else ""

print(f"\\nLEVER D — Resultados test:")
print(f"  Ensemble Final  : NDCG@10={ndcg_final_test:.5f}")
print(f"  NB11 Ens2 (ref) : NDCG@10={NDCG_NB11_ENS2:.5f}")
print(f"  Mejora          : {sign}{delta_d:.1f}%")
target_ok = ndcg_final_test >= TARGET_NDCG
print(f"  Target {TARGET_NDCG}  : {'SUPERADO' if target_ok else 'No alcanzado'}")"""))

# ── Celda 24: Markdown resumen ────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("## E — Tabla comparativa NB13 y actualización de artefactos"))

# ── Celda 25: Tabla resumen ───────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# ── Compilar tabla comparativa ────────────────────────────────────────────────
summary_rows = [
    {"Notebook": "NB11",  "Modelo": f"EASE^R (lambda={EASE_REG_BASE})",            "NDCG@10": ndcg_ease_base_test,   "Prec@10": m_ease_base_test.get("Precision@10", float("nan")),   "Recall@10": m_ease_base_test.get("Recall@10", float("nan")),   "Coverage": m_ease_base_test.get("Coverage", float("nan")),   "Novelty": m_ease_base_test.get("Novelty", float("nan"))},
    {"Notebook": "NB11",  "Modelo": "RP3beta opt (alpha=0.75,beta=0.30)",            "NDCG@10": ndcg_rp3opt_test,      "Prec@10": m_rp3opt_test.get("Precision@10", float("nan")),      "Recall@10": m_rp3opt_test.get("Recall@10", float("nan")),      "Coverage": m_rp3opt_test.get("Coverage", float("nan")),      "Novelty": m_rp3opt_test.get("Novelty", float("nan"))},
    {"Notebook": "NB13-A","Modelo": f"EASE^R opt (lambda={best_lambda:.1f})",       "NDCG@10": ndcg_ease_opt_test,    "Prec@10": m_ease_opt_test.get("Precision@10", float("nan")),    "Recall@10": m_ease_opt_test.get("Recall@10", float("nan")),    "Coverage": m_ease_opt_test.get("Coverage", float("nan")),    "Novelty": m_ease_opt_test.get("Novelty", float("nan"))},
    {"Notebook": "NB13-B","Modelo": f"Ensemble 3-way (w1={best_w1:.2f},w2={best_w2:.2f},w3={best_w3:.2f})", "NDCG@10": ndcg_ens3_test, "Prec@10": m_ens3_test.get("Precision@10", float("nan")), "Recall@10": m_ens3_test.get("Recall@10", float("nan")), "Coverage": m_ens3_test.get("Coverage", float("nan")), "Novelty": m_ens3_test.get("Novelty", float("nan"))},
    {"Notebook": "NB13-C","Modelo": f"RP3+TD decay={best_decay_rate}",              "NDCG@10": ndcg_rp3_td_test,      "Prec@10": m_rp3_td_test.get("Precision@10", float("nan")),      "Recall@10": m_rp3_td_test.get("Recall@10", float("nan")),      "Coverage": m_rp3_td_test.get("Coverage", float("nan")),      "Novelty": m_rp3_td_test.get("Novelty", float("nan"))},
    {"Notebook": "NB13-D","Modelo": f"EnsembleFinal ({rp3_label}+EASE+VAE)",        "NDCG@10": ndcg_final_test,       "Prec@10": m_final_test.get("Precision@10", float("nan")),       "Recall@10": m_final_test.get("Recall@10", float("nan")),       "Coverage": m_final_test.get("Coverage", float("nan")),       "Novelty": m_final_test.get("Novelty", float("nan"))},
]

df_nb13 = pd.DataFrame(summary_rows)
for col in ["NDCG@10", "Prec@10", "Recall@10"]:
    df_nb13[col] = df_nb13[col].round(6)
df_nb13["Coverage"] = df_nb13["Coverage"].round(4)
df_nb13["Novelty"]  = df_nb13["Novelty"].round(2)

print("\\n" + "=" * 90)
print("TABLA NB13 — Comparativa de modelos (test set)")
print("=" * 90)
print(df_nb13.to_string(index=False))
print()

out_csv = DATA_DIR / "model_comparison_nb13.csv"
df_nb13.to_csv(out_csv, index=False)
print(f"CSV guardado: {out_csv}")"""))

# ── Celda 26: Actualizar model_comparison_final.csv ───────────────────────────
cells.append(nbf.v4.new_code_cell("""# ── Actualizar model_comparison_final.csv si hay nuevo campeon ───────────────
final_csv = DATA_DIR / "model_comparison_final.csv"
df_final  = pd.read_csv(final_csv)

best_nb13_idx   = df_nb13["NDCG@10"].idxmax()
best_nb13_row   = df_nb13.loc[best_nb13_idx]
best_nb13_ndcg  = best_nb13_row["NDCG@10"]

# Detectar columna NDCG@10
ndcg_col = "NDCG@10" if "NDCG@10" in df_final.columns else df_final.columns[1]
current_best_ndcg = float(df_final[ndcg_col].max())

print(f"Mejor NB13 : {best_nb13_row['Modelo']}  NDCG@10={best_nb13_ndcg:.5f}")
print(f"Actual best: NDCG@10={current_best_ndcg:.5f}")

if best_nb13_ndcg > current_best_ndcg:
    new_row = {col: None for col in df_final.columns}
    new_row["Notebook"] = "NB13"
    new_row["Modelo"]   = best_nb13_row["Modelo"]
    new_row[ndcg_col]   = best_nb13_ndcg
    for col in ["Prec@10", "Recall@10", "Coverage", "Novelty"]:
        if col in df_final.columns and col in best_nb13_row.index:
            new_row[col] = best_nb13_row[col]
    df_final = pd.concat([df_final, pd.DataFrame([new_row])], ignore_index=True)
    df_final.to_csv(final_csv, index=False)
    print(f"model_comparison_final.csv actualizado — nuevo campeon NDCG@10={best_nb13_ndcg:.5f}")
else:
    print(f"NB11 sigue siendo campeon (NDCG@10={current_best_ndcg:.5f}), no se actualiza")"""))

# ── Celda 27: Actualizar docs ──────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# ── Actualizar docs/model_justification.md ────────────────────────────────────
import re

doc_path = DOCS_DIR / "model_justification.md"
doc_text = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""

best_overall = max(ndcg_ease_opt_test, ndcg_ens3_test, ndcg_rp3_td_test, ndcg_final_test)

section56 = f\"\"\"

---

### 5.6 Notebook 13 — Ensemble Avanzado (4 Levers)

**Objetivo:** Superar NDCG@10 = {NDCG_NB11_ENS2:.5f} apuntando a >= {TARGET_NDCG}.

| Lever | Descripcion | NDCG@10 test | Mejora vs NB11 |
|---|---|---|---|
| A – EASE^R lambda opt | lambda={best_lambda:.1f} (era 500) | {ndcg_ease_opt_test:.5f} | {(ndcg_ease_opt_test-NDCG_NB11_ENS2)/NDCG_NB11_ENS2*100:+.1f}% |
| B – Ensemble 3-way | RP3+EASE+MultiVAE | {ndcg_ens3_test:.5f} | {(ndcg_ens3_test-NDCG_NB11_ENS2)/NDCG_NB11_ENS2*100:+.1f}% |
| C – Temporal Decay | decay_rate={best_decay_rate} | {ndcg_rp3_td_test:.5f} | {(ndcg_rp3_td_test-NDCG_NB11_ENS2)/NDCG_NB11_ENS2*100:+.1f}% |
| D – Ensemble Final | {rp3_label}+EASE+VAE | {ndcg_final_test:.5f} | {(ndcg_final_test-NDCG_NB11_ENS2)/NDCG_NB11_ENS2*100:+.1f}% |

**Mejor resultado NB13:** NDCG@10 = {best_overall:.5f} ({'SUPERA target' if best_overall >= TARGET_NDCG else 'No alcanza target'} {TARGET_NDCG})

Artefactos generados:
- `encoders/ease_optimized.pkl` — EASE^R con lambda optimo
- `data/processed/model_comparison_nb13.csv` — tabla comparativa completa
\"\"\"

if "### 5.6" in doc_text:
    doc_text = re.sub(r"\\n---\\n\\n### 5\\.6.*", section56, doc_text, flags=re.DOTALL)
else:
    doc_text += section56

doc_path.write_text(doc_text, encoding="utf-8")
print(f"docs/model_justification.md actualizado — seccion 5.6 insertada.")"""))

# ── Celda 28: Reporte final ────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell("""# ── Reporte final ────────────────────────────────────────────────────────────
print("\\n" + "=" * 70)
print("NB13 — REPORTE FINAL")
print("=" * 70)
print(f"Punto de partida (NB11 Ens2): NDCG@10 = {NDCG_NB11_ENS2:.5f}")
print(f"Target                      : NDCG@10 = {TARGET_NDCG:.5f}")
print()
print(f"Lever A  EASE lambda opt    : {ndcg_ease_opt_test:.5f}  ({(ndcg_ease_opt_test-NDCG_NB11_ENS2)/NDCG_NB11_ENS2*100:+.1f}%)")
print(f"Lever B  Ensemble 3-way     : {ndcg_ens3_test:.5f}  ({(ndcg_ens3_test-NDCG_NB11_ENS2)/NDCG_NB11_ENS2*100:+.1f}%)")
print(f"Lever C  RP3+TD             : {ndcg_rp3_td_test:.5f}  ({(ndcg_rp3_td_test-NDCG_NB11_ENS2)/NDCG_NB11_ENS2*100:+.1f}%)")
print(f"Lever D  Ensemble Final     : {ndcg_final_test:.5f}  ({(ndcg_final_test-NDCG_NB11_ENS2)/NDCG_NB11_ENS2*100:+.1f}%)")
print()
best_nb13_overall = max(ndcg_ease_opt_test, ndcg_ens3_test, ndcg_rp3_td_test, ndcg_final_test)
print(f"MEJOR RESULTADO NB13        : NDCG@10 = {best_nb13_overall:.5f}")
target_ok = best_nb13_overall >= TARGET_NDCG
print(f"Target {TARGET_NDCG}          : {'SUPERADO' if target_ok else 'No alcanzado'}")
print("=" * 70)"""))

nb.cells = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3"
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.10.0"
}

nbf.write(nb, str(OUT))
print(f"Notebook generado: {OUT}")
print(f"Celdas: {len(nb.cells)}")
