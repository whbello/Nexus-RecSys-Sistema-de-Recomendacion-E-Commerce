"""
NB14 v3 — Script optimizado: salta Optuna IPS/MB (usa params ya conocidos),
salta LightGCN (documenta resultado epoch-1 obtenido en v2), va directo a
EASE^R(500) + Ensemble Spearman.

Tiempos estimados:
  - Setup matrices + RP3+TD scores: ~240s
  - E1 IPS γ=0.1 rebuild + scores:  ~370s
  - E2 MB (saved weights) + scores:  ~390s
  - E4 EASE^R(500) + scores:         ~540s
  - E4 Spearman + Ensemble Optuna:   ~300s
  Total: ~31 min
"""
import sys, time, math, gc, warnings, json, os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize as skl_normalize
import scipy.linalg as sla
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
np.random.seed(42)

def flush(msg=''):
    if msg: print(msg)
    sys.stdout.flush()

HERE     = Path(__file__).resolve().parent
ROOT     = HERE.parent
DATA_DIR = ROOT / 'data' / 'processed'
RAW_DIR  = ROOT / 'data' / 'raw'
ENC_DIR  = ROOT / 'encoders'
DOCS_DIR = ROOT / 'docs'

CUTOFF_DATE  = pd.Timestamp('2015-08-22', tz='UTC')
EASE_TOP     = 20_000
EASE_LAM_500 = 500.0
TD_DECAY     = 0.01
RP3_ALPHA    = 0.75
RP3_BETA     = 0.30
RANDOM_STATE = 42
K_VALUES     = [10]
NDCG_NB12B   = 0.02859
TARGET_NDCG  = 0.030

# Params ya conocidos de v2 (evita re-optimización)
IPS_BEST_GAMMA   = 0.1
MB_BEST_W_VIEW   = 2.669
MB_BEST_W_CART   = 1.079
MB_BEST_W_TRANS  = 3.869
# Resultado LightGCN epoch-1 de v2 (impracticable en CPU, documentado)
LGCN_EPOCH1_NDCG = 0.01018
LGCN_EPOCH1_TIME = 835.7  # segundos

flush(f'ROOT: {ROOT}')
flush(f'Params conocidos: IPS_gamma={IPS_BEST_GAMMA}, MB_weights=({MB_BEST_W_VIEW},{MB_BEST_W_CART},{MB_BEST_W_TRANS})')

# ============================================================
# [1] CARGA DE DATOS
# ============================================================
flush('\n[1] Cargando interaction_matrix.csv...')
t0 = time.time()
im = pd.read_csv(DATA_DIR / 'interaction_matrix.csv')
im['last_interaction_ts'] = pd.to_datetime(im['last_interaction_ts'], format='ISO8601', utc=True)
flush(f'  IM: {im.shape}  [{time.time()-t0:.1f}s]')

train_mask = im['last_interaction_ts'] < CUTOFF_DATE
train_df   = im[train_mask].copy()
test_df    = im[~train_mask].copy()

warm_users  = sorted(set(train_df['visitorid'].unique()) & set(test_df['visitorid'].unique()))
rng         = np.random.default_rng(RANDOM_STATE)
N_EVAL      = 3000
eval_users  = rng.choice(warm_users, size=min(N_EVAL, len(warm_users)), replace=False).tolist()

test_items_by_user  = test_df.groupby('visitorid')['itemid'].apply(set).to_dict()
train_items_by_user = train_df.groupby('visitorid')['itemid'].apply(set).to_dict()
test_tx_by_user     = (
    test_df[test_df['last_interaction_type'] == 'transaction']
    .groupby('visitorid')['itemid'].apply(set).to_dict()
)

all_items_global = sorted(im['itemid'].unique())
n_items_global   = len(all_items_global)
n_test_tx        = len(test_df[test_df['last_interaction_type'] == 'transaction'])
baseline_conv    = n_test_tx / (len(set(test_df['visitorid'])) * n_items_global)

activity_groups = np.array([
    0 if len(train_items_by_user.get(u, set())) == 1
    else (1 if len(train_items_by_user.get(u, set())) <= 4 else 2)
    for u in eval_users
])
rng_split = np.random.default_rng(RANDOM_STATE)
val_mask  = np.zeros(len(eval_users), dtype=bool)
for g in [0, 1, 2]:
    idx_g = np.where(activity_groups == g)[0]
    if len(idx_g) == 0: continue
    n_val  = max(1, int(len(idx_g) * 0.15))
    chosen = rng_split.choice(idx_g, size=n_val, replace=False)
    val_mask[chosen] = True

eval_arr     = np.array(eval_users)
val_users    = eval_arr[val_mask].tolist()
test_users_b = eval_arr[~val_mask].tolist()

flush(f'Train: {len(train_df):,}  Test: {len(test_df):,}')
flush(f'val_users: {len(val_users):,}  test_users_b: {len(test_users_b):,}')

# ============================================================
# [2] MATRICES BASE
# ============================================================
flush('\n[2] Construyendo matrices...')
all_train_users = sorted(train_df['visitorid'].unique())
all_train_items = sorted(train_df['itemid'].unique())
user2idx = {u: i for i, u in enumerate(all_train_users)}
item2idx = {it: i for i, it in enumerate(all_train_items)}
idx2item = {i: it for it, i in item2idx.items()}
n_u = len(all_train_users); n_i = len(all_train_items)

rows_r = train_df['visitorid'].map(user2idx).values
cols_r = train_df['itemid'].map(item2idx).values
vals_r = train_df['interaction_strength'].values.astype(np.float32)
R = sp.csr_matrix((vals_r, (rows_r, cols_r)), shape=(n_u, n_i), dtype=np.float32)

item_pop      = np.asarray(R.sum(axis=0)).ravel()
item_pop_dict = {idx2item[ix]: float(item_pop[ix]) for ix in range(n_i)}
n_total_train = float(R.sum())

top_items_idx    = np.argpartition(item_pop, -EASE_TOP)[-EASE_TOP:]
top_items_idx    = top_items_idx[np.argsort(item_pop[top_items_idx])[::-1]]
N_TOP            = len(top_items_idx)
top_items_global = [idx2item[ix] for ix in top_items_idx]
X_top_csr        = R[:, top_items_idx].astype(np.float32).tocsr()
pop_sub          = item_pop[top_items_idx].astype(np.float32)
flush(f'R: {R.shape}  X_top: {X_top_csr.shape}')

# ============================================================
# FUNCIONES BASE
# ============================================================
def ndcg_at_k(r, rel, k):
    r = r[:k]
    dcg  = sum(1/math.log2(ii+2) for ii, x in enumerate(r) if x in rel)
    idcg = sum(1/math.log2(ii+2) for ii in range(min(len(rel), k)))
    return dcg/idcg if idcg else 0.

def prec_at_k(r, rel, k):  return len(set(r[:k]) & rel) / k if k else 0.
def rec_at_k(r, rel, k):   return len(set(r[:k]) & rel) / len(rel) if rel else 0.
def ap_at_k(r, rel, k):
    if not rel: return 0.
    s, h = 0., 0
    for ii, x in enumerate(r[:k]):
        if x in rel: h += 1; s += h/(ii+1)
    return s / min(len(rel), k)

def evaluate(get_fn, evals, tst, tst_tx, pop_d, nt, cat_sz, bconv, ks=K_VALUES):
    acc = {k: {m: [] for m in 'prnm'} for k in ks}
    seen = set(); ne = 0
    for uid in evals:
        ti = tst.get(uid, set())
        if not ti: continue
        try: recs = get_fn(uid, max(ks))
        except Exception: continue
        seen.update(recs); ne += 1
        for k in ks:
            acc[k]['p'].append(prec_at_k(recs, ti, k))
            acc[k]['r'].append(rec_at_k(recs,  ti, k))
            acc[k]['n'].append(ndcg_at_k(recs,  ti, k))
            acc[k]['m'].append(ap_at_k(recs,    ti, k))
    out = {'n_eval': ne}
    for k in ks:
        if not acc[k]['p']: continue
        out[f'NDCG@{k}']      = float(np.mean(acc[k]['n']))
        out[f'Precision@{k}'] = float(np.mean(acc[k]['p']))
        out[f'Recall@{k}']    = float(np.mean(acc[k]['r']))
        out[f'MAP@{k}']       = float(np.mean(acc[k]['m']))
    out['Coverage'] = len(seen) / cat_sz
    return out

def build_rp3(alpha, beta, X_csr, pop_arr):
    P_ui = skl_normalize(X_csr.astype(np.float64), norm='l1', axis=1)
    P_it = skl_normalize(X_csr.T.tocsr().astype(np.float64), norm='l1', axis=1)
    pop_beta = np.power(pop_arr + 1e-10, beta)
    W_raw = P_it @ P_ui
    W = np.asarray(W_raw.todense() if hasattr(W_raw, 'todense') else W_raw, dtype=np.float32)
    del W_raw
    np.power(W, alpha, out=W)
    W /= pop_beta[:, np.newaxis]
    np.fill_diagonal(W, 0.)
    return W

def make_get_rp3(W, X_csr, top_items_list):
    def get_fn(uid, n):
        if uid not in user2idx: return []
        ui  = user2idx[uid]
        x_u = np.asarray(X_csr.getrow(ui).todense(), dtype=np.float32).ravel()
        sc  = x_u @ W
        sc[x_u > 0] = -1.
        top = np.argpartition(sc, -n)[-n:]
        top = top[np.argsort(sc[top])[::-1]]
        return [top_items_list[ix] for ix in top]
    return get_fn

def minmax_norm(v):
    vmin, vmax = v.min(), v.max()
    r = vmax - vmin
    return (v - vmin) / r if r > 1e-12 else np.zeros_like(v)

def compute_scores_matrix(W_or_B, X_m, user_list):
    """Precomputa array (n_users, N_TOP). Permite liberar W despues."""
    scores = np.zeros((len(user_list), N_TOP), dtype=np.float32)
    for i, uid in enumerate(user_list):
        if uid not in user2idx: continue
        ui  = user2idx[uid]
        x_u = np.asarray(X_m.getrow(ui).todense(), dtype=np.float32).ravel()
        sc  = x_u @ W_or_B
        sc[x_u > 0] = -np.inf
        scores[i] = sc
    return scores

flush('Funciones base definidas.')

# ============================================================
# [3] TEMPORAL DECAY + RP3+TD BASELINE
# ============================================================
flush('\n[3] Construyendo R_td con decay=0.01...')
t0 = time.time()
train_df_td = train_df.copy()
train_df_td['days_before'] = (
    (CUTOFF_DATE - train_df_td['last_interaction_ts']).dt.total_seconds() / 86400.0
).clip(lower=0.0)
decay_w = np.exp(-TD_DECAY * train_df_td['days_before'].values)
vals_td = (train_df_td['interaction_strength'].values.astype(np.float32)
           * decay_w.astype(np.float32))
rows_td = train_df_td['visitorid'].map(user2idx).values
cols_td = train_df_td['itemid'].map(item2idx).values
R_td    = sp.csr_matrix((vals_td, (rows_td, cols_td)), shape=(n_u, n_i), dtype=np.float32)
X_top_td = R_td[:, top_items_idx].astype(np.float32).tocsr()
pop_td   = np.asarray(R_td.sum(axis=0)).ravel()[top_items_idx].astype(np.float32)
del train_df_td, decay_w, vals_td, rows_td, cols_td, R_td; gc.collect()
flush(f'  X_top_td: {X_top_td.shape}  [{time.time()-t0:.1f}s]')

flush('Construyendo RP3beta+TD (baseline)...')
t0 = time.time()
W_rp3_td   = build_rp3(RP3_ALPHA, RP3_BETA, X_top_td, pop_td)
get_rp3_td = make_get_rp3(W_rp3_td, X_top_td, top_items_global)
flush(f'  W_rp3_td: {W_rp3_td.shape}  [{time.time()-t0:.1f}s]')

flush('  Precomputando scores RP3+TD...')
t0 = time.time()
scores_val  = {}
scores_test = {}
scores_val['rp3_td']  = compute_scores_matrix(W_rp3_td, X_top_td, val_users)
scores_test['rp3_td'] = compute_scores_matrix(W_rp3_td, X_top_td, test_users_b)
flush(f'  scores rp3_td: {scores_val["rp3_td"].shape}, {scores_test["rp3_td"].shape}  [{time.time()-t0:.1f}s]')
del W_rp3_td; gc.collect()
flush('  W_rp3_td liberado (scores guardados).')

# ============================================================
# ESTRATEGIA 1 -- IPS + TD  (gamma=0.1, ya conocido de v2)
# ============================================================
flush('\n' + '='*60)
flush(f'ESTRATEGIA 1 -- IPS gamma={IPS_BEST_GAMMA} (param conocido de ejecucion anterior)')
flush('='*60)
t0 = time.time()
pop_max = pop_sub.max()
propensity = (pop_sub / (pop_max + 1e-8)) ** IPS_BEST_GAMMA
propensity = np.where(propensity < 1e-6, 1e-6, propensity)
X_ips = X_top_td.copy()
diag_ips = sp.diags(1.0 / propensity, 0, format='csr', dtype=np.float32)
X_ips = X_ips @ diag_ips
pop_ips = np.asarray(X_ips.sum(axis=0)).ravel().astype(np.float32)

flush('Construyendo RP3beta+TD+IPS...')
W_rp3_ips = build_rp3(RP3_ALPHA, RP3_BETA, X_ips, pop_ips)
flush(f'  [{time.time()-t0:.1f}s]')

flush('  Precomputando scores IPS...')
t0 = time.time()
scores_val['rp3_td_ips']  = compute_scores_matrix(W_rp3_ips, X_top_td, val_users)
scores_test['rp3_td_ips'] = compute_scores_matrix(W_rp3_ips, X_top_td, test_users_b)
flush(f'  [{time.time()-t0:.1f}s]')
del W_rp3_ips, X_ips, diag_ips, pop_ips; gc.collect()
flush('  W_rp3_ips liberado.')
# Nota: NDCG@10_test=0.028355 confirmado en v2
flush(f'##RESULT## E1_IPS ndcg_test=0.028355 smoothing={IPS_BEST_GAMMA} [de ejecucion v2]')

# ============================================================
# [5] CARGA EVENTS + ESTRATEGIA 2 -- MB+TD (weights conocidos)
# ============================================================
flush('\n[5] Cargando events.csv...')
t0 = time.time()
events_raw = pd.read_csv(RAW_DIR / 'events.csv')
events_raw.columns = [c.lower() for c in events_raw.columns]
if 'timestamp' in events_raw.columns:
    events_raw['timestamp'] = pd.to_datetime(events_raw['timestamp'], unit='ms', utc=True)
else:
    ts_col = [c for c in events_raw.columns if 'time' in c.lower()][0]
    events_raw.rename(columns={ts_col: 'timestamp'}, inplace=True)
    events_raw['timestamp'] = pd.to_datetime(events_raw['timestamp'], utc=True)
events_train = events_raw[events_raw['timestamp'] < CUTOFF_DATE].copy()
flush(f'  events_raw: {events_raw.shape}  events_train: {len(events_train):,}  [{time.time()-t0:.1f}s]')
del events_raw; gc.collect()

flush('\n' + '='*60)
flush(f'ESTRATEGIA 2 -- MB+TD w_view={MB_BEST_W_VIEW} w_cart={MB_BEST_W_CART} w_trans={MB_BEST_W_TRANS}')
flush('(pesos conocidos de optimizacion Optuna en v2, 1475s)')
flush('='*60)

ev_type_col = [c for c in events_train.columns if 'event' in c.lower()
               and 'type' in c.lower()][0] if not 'event' in events_train.columns else 'event'
# Identifica columna de tipo de evento
for col in events_train.columns:
    if events_train[col].astype(str).str.lower().isin(['view','addtocart','transaction']).mean() > 0.5:
        ev_type_col = col; break

days_col_ev = (CUTOFF_DATE - events_train['timestamp']).dt.total_seconds() / 86400.0
days_col_ev = days_col_ev.clip(lower=0.0)

mb_weight_map = {'view': MB_BEST_W_VIEW, 'addtocart': MB_BEST_W_CART, 'transaction': MB_BEST_W_TRANS}
ev_weights = events_train[ev_type_col].map(mb_weight_map).fillna(MB_BEST_W_VIEW).values
ev_decay   = np.exp(-TD_DECAY * days_col_ev.values)
ev_vals_mb = (ev_weights * ev_decay).astype(np.float32)

visitor_col = [c for c in events_train.columns if 'visitor' in c.lower()][0]
item_col    = [c for c in events_train.columns if 'item' in c.lower()][0]

ev_u = events_train[visitor_col].map(user2idx)
ev_i = events_train[item_col].map(item2idx)
valid_mask = ev_u.notna() & ev_i.notna()
ev_u = ev_u[valid_mask].values.astype(int)
ev_i = ev_i[valid_mask].values.astype(int)
ev_vals_mb = ev_vals_mb[valid_mask]

t0 = time.time()
R_mb = sp.csr_matrix((ev_vals_mb, (ev_u, ev_i)), shape=(n_u, n_i), dtype=np.float32)
X_mb_td = R_mb[:, top_items_idx].tocsr()
pop_mb  = np.asarray(R_mb.sum(axis=0)).ravel()[top_items_idx].astype(np.float32)
del R_mb, ev_u, ev_i, ev_vals_mb, ev_decay, ev_weights, days_col_ev; gc.collect()

flush('Construyendo RP3beta+MB+TD...')
W_rp3_mb = build_rp3(RP3_ALPHA, RP3_BETA, X_mb_td, pop_mb)
flush(f'  [{time.time()-t0:.1f}s]')

flush('  Precomputando scores MB+TD...')
t0 = time.time()
scores_val['rp3_mb_td']  = compute_scores_matrix(W_rp3_mb, X_top_td, val_users)
scores_test['rp3_mb_td'] = compute_scores_matrix(W_rp3_mb, X_top_td, test_users_b)
flush(f'  [{time.time()-t0:.1f}s]')
del W_rp3_mb, X_mb_td, pop_mb, events_train; gc.collect()
flush('  W_rp3_mb liberado.')
flush(f'##RESULT## E2_MB ndcg_test=0.018900 [de ejecucion v2]')

# ============================================================
# ESTRATEGIA 3 -- LightGCN  (NO ejecutado - impracticable CPU)
# ============================================================
flush('\n' + '='*60)
flush('ESTRATEGIA 3 -- LightGCN (SALTADO - impracticable en CPU)')
flush('='*60)
flush(f'  Resultado disponible de ejecucion v2:')
flush(f'  Epoch 1: BPR_loss=0.4745  NDCG@10_val={LGCN_EPOCH1_NDCG:.5f}  t={LGCN_EPOCH1_TIME:.1f}s')
flush(f'  Proyeccion: 50 epochs × {LGCN_EPOCH1_TIME:.0f}s = {50*LGCN_EPOCH1_TIME/3600:.1f}h (impracticable)')
flush(f'  LightGCN requiere GPU para datasets de esta escala (1.18M nodos, 2M aristas).')
flush(f'##RESULT## E3_LGCN ndcg_test=nan converged=False reason=cpu_too_slow epoch1_ndcg={LGCN_EPOCH1_NDCG}')
lgcn_available = False

# ============================================================
# GUARDAR SCORES A DISCO para liberar RAM antes de EASE
# (EASE inv 20K×20K necesita ~3.2GB; scores ocupan ~720MB)
# ============================================================
flush('\nGuardando scores a disco para liberar RAM antes de EASE...')
SCORE_CACHE = HERE / '_score_cache'
SCORE_CACHE.mkdir(exist_ok=True)
for nm, arr in scores_val.items():
    np.save(SCORE_CACHE / f'val_{nm}.npy', arr)
for nm, arr in scores_test.items():
    np.save(SCORE_CACHE / f'test_{nm}.npy', arr)
scores_val.clear()
scores_test.clear()
# Liberar DataFrames grandes (train_df ~140MB, test_df ~30MB, im ~170MB)
del train_df, test_df, im
gc.collect()
flush(f'  Scores guardados en {SCORE_CACHE}. DataFrames liberados. RAM liberada.')

# ============================================================
# ESTRATEGIA 4 -- EASE^R(500) + Ensemble Spearman
# ============================================================
flush('\n' + '='*60)
flush('ESTRATEGIA 4 -- EASE^R(lambda=500) + Ensemble Spearman')
flush('='*60)

flush('Construyendo EASE^R(lambda=500)...')
t0 = time.time()
G_sp     = X_top_csr.T @ X_top_csr
G_dense  = np.asarray(G_sp.todense(), dtype=np.float32)
del G_sp; gc.collect()
diag_idx  = np.arange(N_TOP)
G_reg500  = G_dense.copy(); del G_dense; gc.collect()
G_reg500[diag_idx, diag_idx] += EASE_LAM_500
# overwrite_a=True evita crear copia interna (ahorra 1.6GB RAM peak)
B_inv500  = sla.inv(G_reg500, overwrite_a=True, check_finite=False)
del G_reg500; gc.collect()
d_inv500  = np.diag(B_inv500).copy()
B_ease500 = -(B_inv500 / d_inv500[np.newaxis, :]).astype(np.float32)
np.fill_diagonal(B_ease500, 0.)
del B_inv500; gc.collect()
flush(f'  B_ease500: {B_ease500.shape}  [{time.time()-t0:.1f}s]')

flush('  Precomputando scores EASE...')
t0 = time.time()
scores_val['ease_500']  = compute_scores_matrix(B_ease500, X_top_csr, val_users)
scores_test['ease_500'] = compute_scores_matrix(B_ease500, X_top_csr, test_users_b)
flush(f'  [{time.time()-t0:.1f}s]')
del B_ease500; gc.collect()
flush('  B_ease500 liberado.')

# Recargar scores previos desde disco
flush('\nRecargando scores anteriores desde disco...')
for nm in ['rp3_td', 'rp3_td_ips', 'rp3_mb_td']:
    scores_val[nm]  = np.load(SCORE_CACHE / f'val_{nm}.npy')
    scores_test[nm] = np.load(SCORE_CACHE / f'test_{nm}.npy')
flush('  Recarga completa.')

# ============================================================
# CORRELACIONES SPEARMAN
# ============================================================
flush('\nMatriz de correlaciones Spearman entre candidatos...')
candidatos_names = ['rp3_td', 'ease_500', 'rp3_td_ips', 'rp3_mb_td']
n_models    = len(candidatos_names)
corr_matrix = np.zeros((n_models, n_models))
for ii, m1 in enumerate(candidatos_names):
    for jj, m2 in enumerate(candidatos_names):
        if ii == jj:  corr_matrix[ii, jj] = 1.0; continue
        if ii > jj:   corr_matrix[ii, jj] = corr_matrix[jj, ii]; continue
        rhos = []
        s1 = scores_val[m1]; s2 = scores_val[m2]
        for u_idx in range(len(val_users)):
            mask = np.isfinite(s1[u_idx]) & np.isfinite(s2[u_idx])
            if mask.sum() < 10: continue
            rho, _ = spearmanr(s1[u_idx][mask], s2[u_idx][mask])
            if not np.isnan(rho): rhos.append(rho)
        corr_matrix[ii, jj] = corr_matrix[jj, ii] = float(np.mean(rhos)) if rhos else 0.

import sys as _sys
df_corr = pd.DataFrame(corr_matrix, index=candidatos_names, columns=candidatos_names)
flush('\nMatriz Spearman (val_users):')
flush(df_corr.round(3).to_string())

# Selecciona trio de menor correlacion promedio
min_corr = float('inf'); best_trio = None
for trio in combinations(range(n_models), 3):
    avg_c = np.mean([corr_matrix[trio[a], trio[b]] for a, b in combinations(range(3), 2)])
    if avg_c < min_corr: min_corr = avg_c; best_trio = trio
selected = [candidatos_names[ix] for ix in best_trio]
flush(f'\nTrio seleccionado (corr_promedio={min_corr:.3f}): {selected}')

# ============================================================
# ENSEMBLE OPTUNA
# ============================================================
flush(f'\nOptuna Ensemble 40 trials sobre trio seleccionado...')
t_ens = time.time()

val_user_to_idx = {uid: idx for idx, uid in enumerate(val_users)}
sel_scores_val  = {}
for name in selected:
    scr = scores_val[name].copy()
    for u_idx in range(scr.shape[0]):
        mask = np.isfinite(scr[u_idx])
        if mask.any(): scr[u_idx][mask] = minmax_norm(scr[u_idx][mask])
        scr[u_idx][~mask] = 0.
    sel_scores_val[name] = scr

def make_ensemble_fn(w_dict, scores_dict, u2idx, top_items_list):
    def get_fn(uid, n):
        if uid not in u2idx: return []
        ui = u2idx[uid]
        sc = sum(w * scores_dict[nm][ui] for nm, w in w_dict.items() if nm in scores_dict)
        top = np.argpartition(sc, -n)[-n:]
        top = top[np.argsort(sc[top])[::-1]]
        return [top_items_list[ix] for ix in top]
    return get_fn

def objective_ens(trial):
    w1 = trial.suggest_float(f'w_{selected[0]}', 0.05, 1.0)
    w2 = trial.suggest_float(f'w_{selected[1]}', 0.05, 1.0)
    w3 = trial.suggest_float(f'w_{selected[2]}', 0.05, 1.0)
    total = w1 + w2 + w3
    w_dict = {
        selected[0]: w1/total,
        selected[1]: w2/total,
        selected[2]: w3/total,
    }
    get_fn = make_ensemble_fn(w_dict, sel_scores_val, val_user_to_idx, top_items_global)
    m = evaluate(get_fn, val_users, test_items_by_user, test_tx_by_user,
                 item_pop_dict, n_total_train, n_items_global, baseline_conv)
    return m.get('NDCG@10', 0.)

study_ens = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_ens.optimize(objective_ens, n_trials=40, show_progress_bar=False)
flush(f'Ensemble Optuna finalizado en {time.time()-t_ens:.0f}s')

best_params_ens = study_ens.best_params
total_w = sum(best_params_ens.values())
best_w_dict = {k.replace('w_', ''): v/total_w for k, v in best_params_ens.items()}
ndcg_ens_val = study_ens.best_value
flush(f'Pesos optimos ensemble: {best_w_dict}')
flush(f'NDCG@10_val ensemble = {ndcg_ens_val:.5f}')

# Evaluacion en test
flush(f'\nEvaluando Ensemble en test ({len(test_users_b)})...')
t0 = time.time()
sel_scores_test = {}
for name in selected:
    scr = scores_test[name].copy()
    for u_idx in range(scr.shape[0]):
        mask = np.isfinite(scr[u_idx])
        if mask.any(): scr[u_idx][mask] = minmax_norm(scr[u_idx][mask])
        scr[u_idx][~mask] = 0.
    sel_scores_test[name] = scr

test_user_to_idx = {uid: idx for idx, uid in enumerate(test_users_b)}
get_ens_test = make_ensemble_fn(best_w_dict, sel_scores_test, test_user_to_idx, top_items_global)
m_ens_test   = evaluate(get_ens_test, test_users_b, test_items_by_user, test_tx_by_user,
                         item_pop_dict, n_total_train, n_items_global, baseline_conv)
ndcg_ens_test = m_ens_test['NDCG@10']
d_ens = (ndcg_ens_test - NDCG_NB12B) / NDCG_NB12B * 100
flush(f'Ensemble test NDCG@10={ndcg_ens_test:.5f}  ({d_ens:+.1f}%)  [{time.time()-t0:.1f}s]')
flush(f'##RESULT## E4_ENS ndcg_test={ndcg_ens_test:.6f} ndcg_val={ndcg_ens_val:.6f} delta={d_ens:.1f}%')

# ============================================================
# GUARDAR RESULTADOS
# ============================================================
flush('\nGuardando resultados...')

results = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'baseline': {'model': 'RP3+TD (NB13-C)', 'ndcg10_test': NDCG_NB12B},
    'E1_IPS': {
        'ndcg10_test': 0.028355,
        'ndcg10_val':  0.01809,
        'best_gamma':  IPS_BEST_GAMMA,
        'delta_pct':   (0.028355 - NDCG_NB12B) / NDCG_NB12B * 100,
        'note':        'Resultado de ejecucion v2 (grid 6 gammas)'
    },
    'E2_MB': {
        'ndcg10_test': 0.018900,
        'ndcg10_val':  0.01118,
        'w_view':      MB_BEST_W_VIEW,
        'w_cart':      MB_BEST_W_CART,
        'w_trans':     MB_BEST_W_TRANS,
        'delta_pct':   (0.018900 - NDCG_NB12B) / NDCG_NB12B * 100,
        'note':        'Resultado de ejecucion v2 (Optuna 40 trials)'
    },
    'E3_LGCN': {
        'ndcg10_test': None,
        'converged':   False,
        'epoch1_ndcg_val': LGCN_EPOCH1_NDCG,
        'epoch1_time_s':   LGCN_EPOCH1_TIME,
        'projected_50ep_hours': round(50 * LGCN_EPOCH1_TIME / 3600, 1),
        'note':        f'No evaluado. Epoch 1 tomo {LGCN_EPOCH1_TIME:.0f}s en CPU. '
                       f'50 epochs = ~{50*LGCN_EPOCH1_TIME/3600:.1f}h. Requiere GPU.'
    },
    'E4_Ensemble': {
        'ndcg10_test': ndcg_ens_test,
        'ndcg10_val':  ndcg_ens_val,
        'delta_pct':   d_ens,
        'trio_selected': selected,
        'trio_corr_avg': round(min_corr, 3),
        'weights': best_w_dict,
        'spearman_matrix': df_corr.round(3).to_dict()
    },
    'target_0030_reached': ndcg_ens_test >= TARGET_NDCG
}

results_path = HERE / '_nb14v3_results.json'
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
flush(f'Guardado: {results_path}')

# CSV comparacion
rows_csv = [
    {'notebook': 'NB13-C', 'model': 'RP3+TD (baseline)',       'ndcg10_test': NDCG_NB12B,      'delta_pct': 0.0},
    {'notebook': 'NB14-E1', 'model': 'RP3+TD+IPS (γ=0.1)',     'ndcg10_test': 0.028355,         'delta_pct': results['E1_IPS']['delta_pct']},
    {'notebook': 'NB14-E2', 'model': 'RP3+MB+TD (Optuna)',      'ndcg10_test': 0.018900,         'delta_pct': results['E2_MB']['delta_pct']},
    {'notebook': 'NB14-E3', 'model': 'LightGCN+TD (epoch1)',    'ndcg10_test': None,             'delta_pct': None},
    {'notebook': 'NB14-E4', 'model': 'Ensemble Spearman',       'ndcg10_test': ndcg_ens_test,    'delta_pct': d_ens},
]
df_csv = pd.DataFrame(rows_csv)
csv_path = DATA_DIR / 'model_comparison_nb14.csv'
df_csv.to_csv(csv_path, index=False)
flush(f'Guardado: {csv_path}')

# Mostrar resumen final
flush('\n' + '='*60)
flush('RESUMEN FINAL NB14')
flush('='*60)
flush(f'  Baseline RP3+TD:          NDCG@10 = {NDCG_NB12B:.5f}')
flush(f'  E1 IPS (γ=0.1):           NDCG@10 = 0.028355  ({results["E1_IPS"]["delta_pct"]:+.1f}%)')
flush(f'  E2 MB+TD (Optuna):        NDCG@10 = 0.018900  ({results["E2_MB"]["delta_pct"]:+.1f}%)')
flush(f'  E3 LightGCN:              NO evaluado (CPU demasiado lento: {LGCN_EPOCH1_TIME:.0f}s/epoch)')
flush(f'  E4 Ensemble {selected}:')
flush(f'     NDCG@10 = {ndcg_ens_test:.5f}  ({d_ens:+.1f}%)')
flush(f'  Target 0.030:       {"ALCANZADO ✓" if ndcg_ens_test >= TARGET_NDCG else "NO alcanzado ✗"}')
flush('\n##DONE##')
