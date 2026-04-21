"""
NB15 v2 — Ensemble mejorado sobre NB14 (NDCG@10=0.04069)
Diferencia clave vs v1:
  - Evalua cada modelo INDIVIDUALMENTE para descartar iALS (señal negativa)
  - Usa el trio NB14 (rp3_td, ease_500, rp3_mb_td) como BASE garantizada
  - Agrega nuevos lambdas EASE solo si MEJORAN sobre el trio
  - Optuna sobre el conjunto completo de buenos modelos (no solo min-corr quad)
  - N_OPTUNA = 100 trials para más convergencia
  - Todos los scores ya estan en _score_cache → solo ensemble y evaluacion
"""
import sys, time, math, gc, warnings, json, os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize as skl_normalize
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
np.random.seed(42)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def flush(msg=''):
    if msg: print(msg, flush=True)
    sys.stdout.flush()

HERE        = Path(__file__).resolve().parent
ROOT        = HERE.parent
DATA_DIR    = ROOT / 'data' / 'processed'
SCORE_CACHE = HERE / '_score_cache'

CUTOFF_DATE  = pd.Timestamp('2015-08-22', tz='UTC')
RANDOM_STATE = 42
K_VALUES     = [10]
NDCG_NB12B   = 0.02859
NDCG_NB14    = 0.04069
N_OPTUNA     = 100

flush(f'ROOT: {ROOT}')
flush(f'NB14 champion: NDCG@10={NDCG_NB14}')

# ============================================================
# [1] CARGA DE DATOS (igual que NB14v3)
# ============================================================
flush('\n[1] Cargando interaction_matrix.csv...')
t0 = time.time()
im = pd.read_csv(DATA_DIR / 'interaction_matrix.csv')
im['last_interaction_ts'] = pd.to_datetime(im['last_interaction_ts'], format='ISO8601', utc=True)
flush(f'  IM: {im.shape}  [{time.time()-t0:.1f}s]')

train_mask = im['last_interaction_ts'] < CUTOFF_DATE
train_df   = im[train_mask].copy()
test_df    = im[~train_mask].copy()

warm_users = sorted(set(train_df['visitorid'].unique()) & set(test_df['visitorid'].unique()))
rng        = np.random.default_rng(RANDOM_STATE)
N_EVAL     = 3000
eval_users = rng.choice(warm_users, size=min(N_EVAL, len(warm_users)), replace=False).tolist()

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
flush(f'val_users: {len(val_users):,}  test_users_b: {len(test_users_b):,}')

# Matrices minimas para reconstruir top_items_global y user2idx
flush('\n[2] Construyendo estructuras base...')
all_train_users = sorted(train_df['visitorid'].unique())
all_train_items = sorted(train_df['itemid'].unique())
user2idx = {u: i for i, u in enumerate(all_train_users)}
item2idx = {it: i for i, it in enumerate(all_train_items)}
idx2item = {i: it for it, i in item2idx.items()}
n_u = len(all_train_users)
n_i = len(all_train_items)
EASE_TOP = 20_000

rows_r = train_df['visitorid'].map(user2idx).values
cols_r = train_df['itemid'].map(item2idx).values
vals_r = train_df['interaction_strength'].values.astype(np.float32)
R = sp.csr_matrix((vals_r, (rows_r, cols_r)), shape=(n_u, n_i), dtype=np.float32)
item_pop    = np.asarray(R.sum(axis=0)).ravel()
item_pop_dict = {idx2item[ix]: float(item_pop[ix]) for ix in range(n_i)}
n_total_train = float(R.sum())

top_items_idx    = np.argpartition(item_pop, -EASE_TOP)[-EASE_TOP:]
top_items_idx    = top_items_idx[np.argsort(item_pop[top_items_idx])[::-1]]
N_TOP            = len(top_items_idx)
top_items_global = [idx2item[ix] for ix in top_items_idx]
del R, rows_r, cols_r, vals_r; gc.collect()
flush(f'  top_items_global: {N_TOP}')

# ============================================================
# FUNCIONES
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

def minmax_norm(v):
    vmin, vmax = v.min(), v.max()
    r = vmax - vmin
    return (v - vmin) / r if r > 1e-12 else np.zeros_like(v)

def make_ensemble_fn(w_dict, scores_dict, u2idx, top_items_list):
    def get_fn(uid, n):
        if uid not in u2idx: return []
        ui = u2idx[uid]
        sc = sum(w * scores_dict[nm][ui] for nm, w in w_dict.items() if nm in scores_dict)
        top_k = np.argpartition(sc, -n)[-n:]
        top_k = top_k[np.argsort(sc[top_k])[::-1]]
        return [top_items_list[ix] for ix in top_k]
    return get_fn

# ============================================================
# [3] CARGAR TODOS LOS SCORES DESDE CACHE
# ============================================================
flush('\n[3] Cargando todos los scores desde _score_cache...')
all_cache_names = [
    'rp3_td', 'rp3_td_ips', 'rp3_mb_td',
    'ease_50', 'ease_200', 'ease_500', 'ease_1000', 'ease_3000',
    'ials'
]
scores_val  = {}
scores_test = {}

for nm in all_cache_names:
    val_f  = SCORE_CACHE / f'val_{nm}.npy'
    test_f = SCORE_CACHE / f'test_{nm}.npy'
    if val_f.exists() and test_f.exists():
        scores_val[nm]  = np.load(val_f)
        scores_test[nm] = np.load(test_f)
        flush(f'  {nm}: val={scores_val[nm].shape}  test={scores_test[nm].shape}')
    else:
        flush(f'  ADVERTENCIA: {nm} no encontrado en cache')

flush(f'  Modelos cargados: {list(scores_val.keys())}')

# ============================================================
# [4] NORMALIZAR SCORES (minmax por usuario)
# ============================================================
flush('\n[4] Normalizando scores...')
val_u2i  = {uid: idx for idx, uid in enumerate(val_users)}
test_u2i = {uid: idx for idx, uid in enumerate(test_users_b)}

def normalize_scoreset(scores_dict):
    normed = {}
    for nm, scr in scores_dict.items():
        s = scr.copy()
        for u_idx in range(s.shape[0]):
            mask = np.isfinite(s[u_idx])
            if mask.any(): s[u_idx][mask] = minmax_norm(s[u_idx][mask])
            s[u_idx][~mask] = 0.
        normed[nm] = s
    return normed

scores_val_norm  = normalize_scoreset(scores_val)
scores_test_norm = normalize_scoreset(scores_test)
flush('  Normalizacion completa.')

# ============================================================
# [5] EVALUACION INDIVIDUAL DE CADA MODELO (en val)
# ============================================================
flush('\n[5] Evaluacion individual de cada modelo (val)...')
individual_ndcg = {}
for nm, scr_norm in scores_val_norm.items():
    def make_single_fn(name, s_norm, u2idx):
        def get_fn(uid, n):
            if uid not in u2idx: return []
            ui = u2idx[uid]
            sc = s_norm[ui]
            top_k = np.argpartition(sc, -n)[-n:]
            top_k = top_k[np.argsort(sc[top_k])[::-1]]
            return [top_items_global[ix] for ix in top_k]
        return get_fn

    gfn = make_single_fn(nm, scr_norm, val_u2i)
    m = evaluate(gfn, val_users, test_items_by_user, test_tx_by_user,
                 item_pop_dict, n_total_train, n_items_global, baseline_conv)
    individual_ndcg[nm] = m.get('NDCG@10', 0.)
    flush(f'  {nm}: NDCG@10_val = {individual_ndcg[nm]:.5f}')

# Ordenar por NDCG individual
sorted_models = sorted(individual_ndcg.keys(), key=lambda x: -individual_ndcg[x])
flush(f'\n  Ranking individual: {sorted_models}')
flush(f'  Top modelos: {sorted_models[:5]}')

# Descartar modelos con NDCG < 0.001 (no aportan)
good_models = [m for m in sorted_models if individual_ndcg[m] >= 0.001]
flush(f'  Modelos elegibles (NDCG >= 0.001): {good_models}')

# ============================================================
# [6] ENSEMBLE FORZADO: Trio NB14 como base
# ============================================================
flush('\n[6] Trio NB14 base (rp3_td + ease_500 + rp3_mb_td) evaluado en val...')
nb14_trio = ['rp3_td', 'ease_500', 'rp3_mb_td']

# Verificar que el trio esta disponible
nb14_trio_available = [m for m in nb14_trio if m in scores_val_norm]
flush(f'  Trio disponible: {nb14_trio_available}')

# Optuna sobre trio base (reproducir NB14)
def objective_trio(trial):
    ws = {nm: trial.suggest_float(f'w_{nm}', 0.05, 1.0) for nm in nb14_trio_available}
    total = sum(ws.values())
    w_dict = {nm: w/total for nm, w in ws.items()}
    gfn = make_ensemble_fn(w_dict, scores_val_norm, val_u2i, top_items_global)
    m = evaluate(gfn, val_users, test_items_by_user, test_tx_by_user,
                 item_pop_dict, n_total_train, n_items_global, baseline_conv)
    return m.get('NDCG@10', 0.)

study_trio = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study_trio.optimize(objective_trio, n_trials=60)
ndcg_trio_val = study_trio.best_value
best_trio_params = study_trio.best_params
total_w = sum(best_trio_params.values())
best_trio_weights = {k.replace('w_', ''): v/total_w for k, v in best_trio_params.items()}
flush(f'  Trio NB14 val: NDCG@10={ndcg_trio_val:.5f}  weights={best_trio_weights}')

# ============================================================
# [7] GREEDY FORWARD SELECTION: agregar modelos uno a uno
# ============================================================
flush('\n[7] Greedy forward selection sobre todos los buenos modelos...')
selected_models = list(nb14_trio_available)
current_best_val = ndcg_trio_val
current_weights  = dict(best_trio_weights)

# Candidatos a agregar: buenos modelos que no estan en el trio
candidates = [m for m in good_models if m not in selected_models]
flush(f'  Candidatos a agregar: {candidates}')

for candidate in candidates:
    flush(f'\n  Probando agregar: {candidate}...')
    test_models = selected_models + [candidate]

    def objective_add(trial):
        ws = {nm: trial.suggest_float(f'w_{nm}', 0.02, 1.0) for nm in test_models}
        total = sum(ws.values())
        w_dict = {nm: w/total for nm, w in ws.items()}
        gfn = make_ensemble_fn(w_dict, scores_val_norm, val_u2i, top_items_global)
        m = evaluate(gfn, test_models, test_items_by_user, test_tx_by_user,
                     item_pop_dict, n_total_train, n_items_global, baseline_conv)
        return m.get('NDCG@10', 0.)

    # Nota: en objective_add usamos eval_users -> corrijo:
    def objective_add_c(trial):
        ws = {nm: trial.suggest_float(f'w_{nm}', 0.02, 1.0) for nm in test_models}
        total = sum(ws.values())
        w_dict = {nm: w/total for nm, w in ws.items()}
        gfn = make_ensemble_fn(w_dict, scores_val_norm, val_u2i, top_items_global)
        m = evaluate(gfn, val_users, test_items_by_user, test_tx_by_user,
                     item_pop_dict, n_total_train, n_items_global, baseline_conv)
        return m.get('NDCG@10', 0.)

    study_add = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_add.optimize(objective_add_c, n_trials=40)
    new_val = study_add.best_value

    flush(f'    {candidate}: NDCG@10_val = {new_val:.5f}  (antes={current_best_val:.5f})')

    if new_val > current_best_val + 0.0002:  # umbral minimo de mejora
        selected_models.append(candidate)
        current_best_val = new_val
        # Actualizar pesos
        best_add_p = study_add.best_params
        total_wa   = sum(best_add_p.values())
        current_weights = {k.replace('w_', ''): v/total_wa for k, v in best_add_p.items()}
        flush(f'    ACEPTADO: {candidate}  (mejora={new_val-current_best_val+0.0002:.4f})')
        flush(f'    Pesos actuales: {current_weights}')
    else:
        flush(f'    RECHAZADO: {candidate}  (no mejora suficiente)')

flush(f'\n  Seleccion final: {selected_models}  NDCG@10_val={current_best_val:.5f}')

# ============================================================
# [8] OPTUNA FINAL sobre modelos seleccionados (100 trials)
# ============================================================
flush(f'\n[8] Optuna final {N_OPTUNA} trials sobre {selected_models}...')
t_ens = time.time()

def objective_final(trial):
    ws = {nm: trial.suggest_float(f'w_{nm}', 0.02, 1.0) for nm in selected_models}
    total = sum(ws.values())
    w_dict = {nm: w/total for nm, w in ws.items()}
    gfn = make_ensemble_fn(w_dict, scores_val_norm, val_u2i, top_items_global)
    m = evaluate(gfn, val_users, test_items_by_user, test_tx_by_user,
                 item_pop_dict, n_total_train, n_items_global, baseline_conv)
    return m.get('NDCG@10', 0.)

study_final = optuna.create_study(direction='maximize',
                                   sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
# Warm start con los pesos conocidos buenos
if current_weights:
    study_final.enqueue_trial({f'w_{k}': v for k, v in current_weights.items()
                                if k in selected_models})

study_final.optimize(objective_final, n_trials=N_OPTUNA)
flush(f'Optuna final: {time.time()-t_ens:.0f}s')

best_params_f = study_final.best_params
total_wf = sum(best_params_f.values())
best_w_final = {k.replace('w_', ''): v/total_wf for k, v in best_params_f.items()}
ndcg_final_val = study_final.best_value
flush(f'Pesos finales: {best_w_final}')
flush(f'NDCG@10_val = {ndcg_final_val:.5f}')

# ============================================================
# [9] EVALUACION EN TEST
# ============================================================
flush('\n[9] Evaluacion final en test...')
t0 = time.time()
get_final_test = make_ensemble_fn(best_w_final, scores_test_norm, test_u2i, top_items_global)
m_final_test   = evaluate(get_final_test, test_users_b, test_items_by_user, test_tx_by_user,
                           item_pop_dict, n_total_train, n_items_global, baseline_conv)
ndcg_final_test = m_final_test['NDCG@10']
d_vs_nb14  = (ndcg_final_test - NDCG_NB14)  / NDCG_NB14  * 100
d_vs_nb12  = (ndcg_final_test - NDCG_NB12B) / NDCG_NB12B * 100
flush(f'Test NDCG@10 = {ndcg_final_test:.5f}  [{time.time()-t0:.1f}s]')
flush(f'  vs NB14: {d_vs_nb14:+.1f}%')
flush(f'  vs NB12 baseline: {d_vs_nb12:+.1f}%')
flush(f'##RESULT## NB15v2 NDCG@10_test={ndcg_final_test:.6f} delta_nb14={d_vs_nb14:+.1f}%')

# Elegir el mejor entre NB14 y NB15v2
if ndcg_final_test > NDCG_NB14:
    champion_ndcg  = ndcg_final_test
    champion_model = 'NB15v2'
    champion_desc  = f'Mega-Ensemble ({"+".join(selected_models)})'
    flush(f'\n  NUEVO CHAMPION: NB15v2 supera NB14 ({ndcg_final_test:.5f} > {NDCG_NB14:.5f})')
else:
    champion_ndcg  = NDCG_NB14
    champion_model = 'NB14'
    champion_desc  = 'Ensemble Spearman NB14 (rp3_td + ease_500 + rp3_mb_td)'
    flush(f'\n  NB14 sigue siendo champion ({NDCG_NB14:.5f} > {ndcg_final_test:.5f})')
    flush(f'  El analisis NB15 confirma que los modelos nuevos no superan el trio optimo de NB14')

# ============================================================
# [10] GUARDAR RESULTADOS
# ============================================================
flush('\nGuardando resultados...')

results = {
    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'method': 'greedy_forward_selection + optuna',
    'baseline_nb12': {'model': 'RP3+TD (NB13-C)', 'ndcg10_test': NDCG_NB12B},
    'baseline_nb14': {'model': 'Ensemble Spearman (NB14)', 'ndcg10_test': NDCG_NB14},
    'individual_model_ndcg_val': individual_ndcg,
    'nb14_trio_val': ndcg_trio_val,
    'greedy_selection': {
        'selected_models': selected_models,
        'final_val_ndcg': current_best_val,
    },
    'NB15v2_result': {
        'ndcg10_test':      ndcg_final_test,
        'ndcg10_val':       ndcg_final_val,
        'delta_pct_vs_nb14': d_vs_nb14,
        'delta_pct_vs_nb12': d_vs_nb12,
        'selected_models':  selected_models,
        'weights':          best_w_final,
    },
    'champion': {
        'model':      champion_model,
        'ndcg10':     champion_ndcg,
        'description': champion_desc,
    }
}

results_path = HERE / '_nb15v2_results.json'
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
flush(f'Guardado: {results_path}')

# CSV con todos los modelos
rows_csv = [
    {'notebook': 'NB13-C',  'model': 'RP3+TD (baseline)',           'ndcg10_test': NDCG_NB12B,      'delta_pct': 0.0},
    {'notebook': 'NB14-E1', 'model': 'RP3+TD+IPS (gamma=0.1)',      'ndcg10_test': 0.028355,         'delta_pct': (0.028355-NDCG_NB12B)/NDCG_NB12B*100},
    {'notebook': 'NB14-E4', 'model': 'Ensemble Spearman (NB14)',    'ndcg10_test': NDCG_NB14,        'delta_pct': (NDCG_NB14-NDCG_NB12B)/NDCG_NB12B*100},
    {'notebook': 'NB15v2',  'model': f'Mega-Ensemble ({champion_model})', 'ndcg10_test': ndcg_final_test, 'delta_pct': d_vs_nb12},
]
df_csv = pd.DataFrame(rows_csv)
csv_path = DATA_DIR / 'model_comparison_final.csv'
df_csv.to_csv(csv_path, index=False)
flush(f'Guardado CSV final: {csv_path}')

# Resumen final
flush('\n' + '='*60)
flush('RESUMEN FINAL NB15v2')
flush('='*60)
flush(f'  Baseline RP3+TD (NB12):     NDCG@10 = {NDCG_NB12B:.5f}')
flush(f'  Champion Ensemble (NB14):   NDCG@10 = {NDCG_NB14:.5f}  (+{(NDCG_NB14-NDCG_NB12B)/NDCG_NB12B*100:.1f}%)')
flush(f'  NB15v2 Mega-Ensemble:       NDCG@10 = {ndcg_final_test:.5f}  ({d_vs_nb14:+.1f}% vs NB14)')
flush(f'  Modelos seleccionados: {selected_models}')
flush(f'  CHAMPION FINAL: {champion_model} ({champion_ndcg:.5f})')
flush(f'  Individual NDCG@10 (val):')
for nm, v in sorted(individual_ndcg.items(), key=lambda x: -x[1]):
    flush(f'    {nm:20s}: {v:.5f}')
flush('\n##DONE##')
