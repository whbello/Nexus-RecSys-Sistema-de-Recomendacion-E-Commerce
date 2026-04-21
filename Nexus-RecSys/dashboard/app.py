"""
dashboard/app.py — Nexus-RecSys · Space Edition
================================================
Dashboard interactivo de recomendación de productos.
6 páginas: Centro de Mando · Demo en Vivo · Asistente IA ·
           Comparativa de Modelos · Análisis del Dataset · Métricas del Sistema
"""

import gc
import html as _html
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

_DASH_DIR = Path(__file__).resolve().parent
_ENV_FILE = _DASH_DIR.parent / ".env"
load_dotenv(_ENV_FILE, override=True)

if str(_DASH_DIR) not in sys.path:
    sys.path.insert(0, str(_DASH_DIR))

from styles import SPACE_CSS
from catalog import get_product, get_products_batch, load_catalog, catalog_available
from plot_config import SPACE_LAYOUT, COLORS, COLORWAY, apply_space_theme
from llm_engine import try_load_engine, LLMEngine

st.set_page_config(
    page_title="nexus-recsys · Space Edition",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(SPACE_CSS, unsafe_allow_html=True)


ROOT        = _DASH_DIR.parent
DATA_DIR    = ROOT / "data" / "processed"
INTERIM_DIR = ROOT / "data" / "interim"
SCORE_CACHE = ROOT / "scripts" / "_score_cache"
RESULTS_F   = ROOT / "scripts" / "_nb15v2_results.json"

# ── CONSTANTES DEL SISTEMA ────────────────────────────────────────────────────
CUTOFF_DATE      = "2015-08-22"
RANDOM_STATE     = 42
N_EVAL           = 3000
EASE_TOP         = 20_000
NDCG_CHAMPION    = 0.04310
NDCG_BASELINE    = 0.02859   # NB13-C RP3+TD
ENSEMBLE_WEIGHTS = {
    "rp3_mb_td": 0.956,
    "ease_500":  0.021,
    "rp3_td":    0.023,
}

# ── EVENTOS RETAILROCKET (precomputed) ───────────────────────────────────────
_N_VIEWS    = 2_664_312
_N_CART     = 69_332
_N_TX       = 22_457
_N_EVENTS   = _N_VIEWS + _N_CART + _N_TX
_N_USERS    = 1_407_580
_N_ITEMS    = 235_061
_SPARSITY   = 99.9994

# ── FUNCIONES DE CARGA DE DATOS ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def cargar_model_comparison() -> pd.DataFrame:
    """Carga el CSV final de comparación de modelos."""
    p = DATA_DIR / "model_comparison_final.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def cargar_nb15_results() -> dict:
    """Carga los resultados JSON del ensemble NB15v2."""
    if not RESULTS_F.exists():
        return {}
    with open(RESULTS_F, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def cargar_todos_los_modelos() -> pd.DataFrame:
    """Carga y unifica todos los CSVs de comparación de modelos (NB08–NB15 + final)."""
    BASE = NDCG_BASELINE
    frames = []

    _spec = [
        ("model_comparison_08_roi.csv",      "NB08", "Model",    "NDCG@10"),
        ("model_comparison_09_advanced.csv", "NB09", "Modelo",   "NDCG@10"),
        ("model_comparison_10_multivae.csv", "NB10", "Modelo",   "NDCG@10"),
        ("model_comparison_nb11.csv",        "NB11", "Modelo",   "NDCG@10"),
        ("model_comparison_nb13.csv",        None,   "Modelo",   "NDCG@10"),
        ("model_comparison_nb14.csv",        None,   "model",    "ndcg10_test"),
        ("model_comparison_nb15.csv",        None,   "model",    "ndcg10_test"),
        ("model_comparison_final.csv",       None,   "model",    "ndcg10_test"),
    ]

    seen_models: set = set()
    for fname, default_nb, mcol, ncol in _spec:
        p = DATA_DIR / fname
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            # Detectar columna de modelo
            _mc = mcol if mcol in df.columns else next(
                (c for c in df.columns if "model" in c.lower() or "modelo" in c.lower()), None
            )
            # Detectar columna NDCG
            _nc = ncol if ncol in df.columns else next(
                (c for c in df.columns
                 if ("ndcg" in c.lower() or "NDCG" in c)
                 and "10" in c), None
            )
            if not _mc or not _nc:
                continue
            sub = df[[_mc, _nc]].copy()
            sub.columns = ["model", "ndcg10_test"]
            sub["ndcg10_test"] = pd.to_numeric(sub["ndcg10_test"], errors="coerce")
            sub = sub.dropna(subset=["ndcg10_test"])
            # Columna notebook
            if "notebook" in df.columns:
                sub["notebook"] = df["notebook"].values
            elif "Notebook" in df.columns:
                sub["notebook"] = df["Notebook"].values
            else:
                sub["notebook"] = default_nb or fname.replace("model_comparison_", "").replace(".csv", "").upper()
            sub["delta_pct"] = (sub["ndcg10_test"] - BASE) / BASE * 100
            # Evitar duplicados exactos de modelo
            sub = sub[~sub["model"].isin(seen_models)]
            seen_models.update(sub["model"].tolist())
            frames.append(sub[["notebook", "model", "ndcg10_test", "delta_pct"]])
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=["notebook", "model", "ndcg10_test", "delta_pct"])
    return pd.concat(frames, ignore_index=True)


@st.cache_resource(show_spinner="⏳ Cargando motor de recomendación...")
def cargar_ensemble():
    """
    Carga la matriz de interacciones, reconstruye el split de evaluación,
    y combina los score-matrices del cache para el ensemble NB15v2.
    Devuelve (ensemble_scores, user2row, top_items_global, train_items_by_user).
    """
    import scipy.sparse as sp

    cutoff_ts = pd.Timestamp(CUTOFF_DATE, tz="UTC")

    # Cargar interaction matrix
    im = pd.read_csv(DATA_DIR / "interaction_matrix.csv")
    im["last_interaction_ts"] = pd.to_datetime(
        im["last_interaction_ts"], format="ISO8601", utc=True
    )

    train_mask = im["last_interaction_ts"] < cutoff_ts
    train_df   = im[train_mask].copy()
    test_df    = im[~train_mask].copy()

    warm_users = sorted(
        set(train_df["visitorid"].unique()) & set(test_df["visitorid"].unique())
    )

    rng        = np.random.default_rng(RANDOM_STATE)
    eval_users = rng.choice(
        warm_users, size=min(N_EVAL, len(warm_users)), replace=False
    ).tolist()

    train_items_by_user = train_df.groupby("visitorid")["itemid"].apply(set).to_dict()
    test_items_by_user  = test_df.groupby("visitorid")["itemid"].apply(set).to_dict()

    # Split val / test (15% val, 85% test) — mismo algoritmo que NB15v2
    activity_groups = np.array([
        0 if len(train_items_by_user.get(u, set())) == 1
        else (1 if len(train_items_by_user.get(u, set())) <= 4 else 2)
        for u in eval_users
    ])
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
    test_users_b = eval_arr[~val_mask].tolist()

    # user_id → row index en score matrices
    user2row = {uid: i for i, uid in enumerate(test_users_b)}

    # Construir top_items_global (top EASE_TOP ítems por popularidad)
    all_train_users = sorted(train_df["visitorid"].unique())
    all_train_items = sorted(train_df["itemid"].unique())
    u2idx = {u: i for i, u in enumerate(all_train_users)}
    i2idx = {it: i for i, it in enumerate(all_train_items)}
    idx2item = {i: it for it, i in i2idx.items()}
    n_u = len(all_train_users)
    n_i = len(all_train_items)

    rows_r = train_df["visitorid"].map(u2idx).values
    cols_r = train_df["itemid"].map(i2idx).values
    vals_r = train_df["interaction_strength"].values.astype(np.float32)
    R = sp.csr_matrix((vals_r, (rows_r, cols_r)), shape=(n_u, n_i), dtype=np.float32)
    item_pop       = np.asarray(R.sum(axis=0)).ravel()
    top_items_idx  = np.argpartition(item_pop, -EASE_TOP)[-EASE_TOP:]
    top_items_idx  = top_items_idx[np.argsort(item_pop[top_items_idx])[::-1]]
    top_items_global = [idx2item[ix] for ix in top_items_idx]

    del R, rows_r, cols_r, vals_r, train_df, test_df, im
    gc.collect()

    # Cargar score matrices del cache (solo modelos del ensemble NB15v2)
    scores_dict: dict = {}
    _score_map = {
        "rp3_mb_td": SCORE_CACHE / "test_rp3_mb_td.npy",
        "ease_500":  SCORE_CACHE / "test_ease_500.npy",
        "rp3_td":    SCORE_CACHE / "test_rp3_td.npy",
    }
    for name, path in _score_map.items():
        if path.exists():
            scores_dict[name] = np.load(str(path), mmap_mode="r")

    # Combinar con pesos del ensemble
    if scores_dict:
        # shape (n_test_users, EASE_TOP)
        first_key  = next(iter(scores_dict))
        n_rows     = scores_dict[first_key].shape[0]
        n_cols     = scores_dict[first_key].shape[1]
        ensemble_s = np.zeros((n_rows, n_cols), dtype=np.float32)
        for name, w in ENSEMBLE_WEIGHTS.items():
            if name in scores_dict:
                ensemble_s += w * scores_dict[name]
        del scores_dict
        gc.collect()
    else:
        ensemble_s = None

    return (ensemble_s, user2row, top_items_global, train_items_by_user,
            test_items_by_user, test_users_b)


def get_recs_live(user_id: int, top_n: int = 10) -> list:
    """
    Genera recomendaciones en vivo para un usuario warm usando el ensemble NB15v2.
    Devuelve lista de dicts con item_id y score.
    """
    ensemble_s, user2row, top_items_global, train_items, _, _ = cargar_ensemble()

    if ensemble_s is None or user_id not in user2row:
        return []

    row_idx = user2row[user_id]
    scores  = ensemble_s[row_idx].copy()

    # Filtrar ítems ya vistos en entrenamiento
    seen = train_items.get(user_id, set())
    for item in seen:
        try:
            col = top_items_global.index(item)  # O(n) — tolerable para demo
            scores[col] = -1.0
        except ValueError:
            pass

    top_k_idx = np.argpartition(scores, -top_n)[-top_n:]
    top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]

    return [
        {"item_id": top_items_global[i], "score": float(scores[i])}
        for i in top_k_idx
        if scores[i] > 0
    ]


def highlight_ganador(row: pd.Series) -> list:
    """Resalta la fila del modelo ganador (mayor NDCG@10)."""
    return ["background-color: #1a5c2e; font-weight:bold" for _ in row]


@st.cache_resource(show_spinner=False)
def get_demo_users() -> list:
    """
    Retorna una muestra de ~45 usuarios warm válidos para la demo,
    estratificada por nivel de actividad (high/mid/low).
    """
    _, user2row, _, train_items, _, _ = cargar_ensemble()
    all_users = sorted(user2row.keys())
    rng = np.random.default_rng(2024)
    high = [u for u in all_users if len(train_items.get(u, set())) > 10]
    mid  = [u for u in all_users if 3 <= len(train_items.get(u, set())) <= 10]
    low  = [u for u in all_users if len(train_items.get(u, set())) < 3]
    sample: list = []
    for grp, k in [(high, 20), (mid, 15), (low, 10)]:
        if grp:
            chosen = rng.choice(grp, size=min(k, len(grp)), replace=False).tolist()
            sample.extend(int(c) for c in chosen)
    return sorted(sample)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-family:'Orbitron',monospace; font-size:1.6rem;
                    font-weight:900; color:#00c2ff;
                    text-shadow: 0 0 20px rgba(0,194,255,0.5);">
            🚀 NEXUS
        </div>
        <div style="font-size:0.9rem; color:#a8c8e8; letter-spacing:2px; margin-top:4px;">
            DATA CO. · RECSYS v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="space-divider">', unsafe_allow_html=True)

    pagina = st.radio(
        "nav",
        options=[
            "🌌 Centro de Mando",
            "🔭 Demo en Vivo",
            "🤖 Asistente IA",
            "📊 Comparativa de Modelos",
            "🪐 Análisis del Dataset",
            "⭐ Métricas del Sistema",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<hr class="space-divider">', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(100,255,218,0.08); border:1px solid rgba(100,255,218,0.25);
                border-radius:8px; padding:14px; text-align:center;">
        <div style="color:#64ffda; font-size:1rem; font-family:'Orbitron',monospace;
                    letter-spacing:1px;">
            ● SISTEMA ACTIVO
        </div>
        <div style="color:#ffffff; font-size:1rem; margin-top:6px; font-weight:600;">
            Mega-Ensemble NB15v2
        </div>
        <div style="color:#00c2ff; font-size:1.3rem; font-weight:900;
                    font-family:'Orbitron',monospace; margin-top:8px;">
            NDCG@10 = 0.0431
        </div>
        <div style="color:#64ffda; font-size:1rem; margin-top:4px; font-weight:600;">
            +50.8% vs baseline
        </div>
        <hr style="border:none;height:1px;background:rgba(0,194,255,0.2);margin:10px 0;">
        <div style="color:#ccd6f6; font-size:0.95rem; line-height:1.9;">
            RetailRocket · 2.75M eventos<br>
            1.4M usuarios · 235K ítems<br>
            Sparsity 99.9994%
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — CENTRO DE MANDO
# ═══════════════════════════════════════════════════════════════════════════════
if "Centro de Mando" in pagina:
    st.markdown("""
    <div style="padding:20px 0 10px;">
        <div style="font-family:'Orbitron',monospace; font-size:1.8rem; font-weight:900;
                    color:#ffffff; text-shadow: 0 0 30px rgba(0,194,255,0.4);">
            🌌 CENTRO DE MANDO
        </div>
        <div style="color:#ccd6f6; font-size:1rem; margin-top:4px;">
            Nexus Data Co. · Sistema de Recomendación v2.0 · RetailRocket Dataset
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    kc1, kc2, kc3, kc4 = st.columns(4)
    for col, val, label, sub in [
        (kc1, "0.0431",  "NDCG@10 CAMPEÓN",   "Mega-Ensemble NB15v2"),
        (kc2, "+50.8%",  "MEJORA VS BASELINE", "Sobre RP3+TD (NB13-C)"),
        (kc3, "3 modelos","ENSEMBLE FINAL",     "RP3+MB+TD · EASE · RP3+TD"),
        (kc4, "3,000",   "USUARIOS EVALUADOS", "Warm users · split temporal"),
    ]:
        col.markdown(f"""
        <div style="background:rgba(10,20,60,0.7);border:1px solid rgba(0,194,255,0.25);
                    border-radius:10px;padding:16px;text-align:center;">
            <div style="font-family:'Orbitron',monospace;font-size:1.8rem;font-weight:900;
                        color:#00c2ff;">{val}</div>
            <div style="color:#64ffda;font-size:0.95rem;font-weight:700;
                        letter-spacing:1px;margin-top:6px;">{label}</div>
            <div style="color:#ccd6f6;font-size:0.9rem;margin-top:4px;">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="space-divider">', unsafe_allow_html=True)

    # ── Gráfico: Evolución NDCG@10 por Notebook ──────────────────────────────
    col_l, col_r = st.columns([3, 2])
    with col_l:
        _evo_data = pd.DataFrame({
            "Notebook": ["NB08\nSVD+TD+IPS", "NB09\nRP3beta",  "NB11\nEns RP3+EASE",
                         "NB13-C\nRP3+TD",   "NB14-E4\nEns Spearman", "NB15v2\nMega-Ensemble"],
            "NDCG@10":  [0.00934,            0.02576,           0.02603,
                         0.02859,            0.04069,            0.04310],
            "Tipo":     ["Individual",       "Individual",      "Ensemble",
                         "Individual",       "Ensemble",        "Ensemble"],
        })
        _colores_evo = {"Individual": "#a8c8e8", "Ensemble": "#00c2ff"}
        _evo_data["color"] = _evo_data["Tipo"].map(_colores_evo)
        fig_ev = go.Figure(go.Bar(
            x=_evo_data["Notebook"],
            y=_evo_data["NDCG@10"],
            marker_color=_evo_data["color"].tolist(),
            text=[f"{v:.4f}" for v in _evo_data["NDCG@10"]],
            textposition="outside",
            textfont=dict(color="#ccd6f6", size=10),
        ))
        fig_ev.add_hline(
            y=NDCG_BASELINE,
            line_dash="dash",
            line_color="#ff6b6b",
            annotation_text=f"Baseline NB13-C = {NDCG_BASELINE:.4f}",
            annotation_font_color="#ff6b6b",
        )
        apply_space_theme(fig_ev, height=380, title="Evolución NDCG@10 — NB08 → NB15")
        fig_ev.update_layout(showlegend=False)
        st.plotly_chart(fig_ev, use_container_width=True, config={"responsive": True, "displayModeBar": False})

    with col_r:
        # ── Gráfico: Composición del Ensemble ─────────────────────────────────
        _ew_names  = ["RP3+MB+TD", "RP3+TD", "EASE^R-500"]
        _ew_vals   = [0.956,        0.023,     0.021]
        fig_proto = go.Figure(go.Bar(
            y=_ew_names,
            x=_ew_vals,
            orientation="h",
            marker_color=[COLORWAY[0], COLORWAY[1], COLORWAY[2]],
            text=[f"{v:.1%}" for v in _ew_vals],
            textposition="inside",
            textfont=dict(color="#050a1a", size=12, family="Orbitron,monospace"),
        ))
        apply_space_theme(fig_proto, height=220, title="Pesos del Ensemble Campeón")
        st.plotly_chart(fig_proto, use_container_width=True, config={"responsive": True, "displayModeBar": False})

        # ── Gráfico: Distribución de tipos de evento ───────────────────────────
        fig_pie = go.Figure(go.Pie(
            labels=["Views", "Add to Cart", "Transacciones"],
            values=[_N_VIEWS, _N_CART, _N_TX],
            hole=0.55,
            marker=dict(
                colors=[COLORWAY[0], COLORWAY[2], COLORWAY[1]],
                line=dict(color="rgba(0,0,0,0)", width=0),
            ),
            textfont=dict(color="#ccd6f6", size=11),
        ))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccd6f6"),
            legend=dict(bgcolor="rgba(5,10,30,0.8)", font=dict(color="#ccd6f6")),
            margin=dict(t=40, b=10, l=10, r=10),
            height=220,
            title=dict(text="Tipos de Evento", font=dict(family="Orbitron,monospace",
                       color="#ffffff", size=13), x=0.02),
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"responsive": True, "displayModeBar": False})

    # ── Gráfico: Funnel de conversión ─────────────────────────────────────────
    st.markdown('<hr class="space-divider">', unsafe_allow_html=True)
    _funnel_labels  = ["Visualizaciones (Views)",
                       "Carrito (Add to Cart)",
                       "Compras (Transactions)"]
    _funnel_vals    = [_N_VIEWS, _N_CART, _N_TX]
    fig_funnel = go.Figure(go.Funnel(
        y=_funnel_labels,
        x=_funnel_vals,
        textinfo="value+percent previous",
        textfont=dict(color="#ffffff", size=12),
        marker=dict(
            color=[COLORWAY[0], COLORWAY[2], COLORWAY[1]],
            line=dict(color=["rgba(0,194,255,0.5)"]*3, width=2),
        ),
        connector=dict(line=dict(color="rgba(0,194,255,0.3)", width=2)),
    ))
    apply_space_theme(fig_funnel, height=300, title="Funnel de Conversión del E-Commerce RetailRocket")
    st.plotly_chart(fig_funnel, use_container_width=True, config={"responsive": True, "displayModeBar": False})

    # Info extra
    st.markdown("""
    <div style="background:rgba(0,20,60,0.5);border:1px solid rgba(0,194,255,0.2);
                border-radius:8px;padding:16px 20px;margin-top:8px;">
        <div style="color:#64ffda;font-size:0.8rem;font-family:'Orbitron',monospace;
                    letter-spacing:1px;margin-bottom:8px;">● CONCLUSIÓN TÉCNICA</div>
        <div style="color:#ccd6f6;font-size:0.9rem;line-height:1.7;">
            El <strong style="color:#00c2ff;">Mega-Ensemble NB15v2</strong> alcanza
            <strong style="color:#64ffda;">NDCG@10 = 0.0431</strong>, superando en +50.8% al
            mejor modelo individual (RP3+TD, NB13-C). El éxito del ensemble se debe a la
            <em>baja correlación de Spearman (ρ=0.216)</em> entre los modelos componentes,
            que capturan aspectos distintos del comportamiento del usuario a pesar de la
            extrema esparsidad del dataset (99.9994%).
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — DEMO EN VIVO
# ═══════════════════════════════════════════════════════════════════════════════
elif "Demo en Vivo" in pagina:
    st.markdown("""
    <div style="padding:20px 0 10px;">
        <div style="font-family:'Orbitron',monospace; font-size:1.8rem; font-weight:900;
                    color:#ffffff;">🔭 DEMO EN VIVO</div>
        <div style="color:#ccd6f6; font-size:1rem; margin-top:4px;">
            Motor de recomendación en tiempo real · Ensemble NB15v2
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_ctrl, col_res = st.columns([1, 2])

    with col_ctrl:
        st.markdown("""
        <div style="background:rgba(10,20,60,0.7);border:1px solid rgba(0,194,255,0.25);
                    border-radius:10px;padding:16px;">
            <div style="color:#64ffda;font-size:0.9rem;font-family:'Orbitron',monospace;
                        letter-spacing:1px;margin-bottom:12px;">● SELECTOR DE USUARIO</div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        _demo_users = get_demo_users()
        _, _, _, _train_items_demo, _, _ = cargar_ensemble()

        uid_input = st.selectbox(
            "Seleccionar Usuario",
            options=_demo_users,
            format_func=lambda x: f"Usuario #{x:,}  ({len(_train_items_demo.get(x, set()))} interacciones)",
            help="Usuarios warm con historial confirmado en el set de evaluación.",
        )
        top_n_input = st.slider("N° de recomendaciones", min_value=3, max_value=15, value=8)

        run_btn = st.button(
            "🚀 Generar Recomendaciones",
            use_container_width=True,
            type="primary",
        )

        st.markdown("""
        <div style="background:rgba(0,194,255,0.07);border:1px solid rgba(0,194,255,0.2);
                    border-radius:8px;padding:14px;margin-top:12px;">
            <div style="color:#ccd6f6;font-size:1rem;line-height:1.8;">
                <strong style="color:#00c2ff;">ℹ ~2,551 usuarios warm</strong> disponibles.<br>
                El sistema filtra ítems ya vistos<br>
                y aplica pesos del ensemble:<br>
                <span style="color:#64ffda;font-weight:700;">RP3+MB+TD(95.6%) · EASE(2.1%) · RP3+TD(2.3%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_res:
        if run_btn or uid_input:
            with st.spinner("⚡ Calculando recomendaciones..."):
                recs_raw = get_recs_live(int(uid_input), top_n=top_n_input)

            if not recs_raw:
                st.warning(
                    f"⚠ Usuario #{uid_input} no tiene recomendaciones disponibles. "
                    "Selecciona otro usuario del menú."
                )
            else:
                # Enriquecer con catálogo
                recs_enriq = []
                for rank, r in enumerate(recs_raw, 1):
                    prod = get_product(r["item_id"])
                    recs_enriq.append({
                        "rank":        rank,
                        "item_id":     r["item_id"],
                        "score":       r["score"],
                        "name":        prod.get("name",        f"Ref. #{r['item_id']}"),
                        "category":    prod.get("category",    "Sin categoría"),
                        "subcategory": prod.get("subcategory", "General"),
                        "description": prod.get("description", ""),
                        "price":       prod.get("price"),
                        "emoji":       prod.get("emoji",       "📦"),
                        "option":      prod.get("option",      "A"),
                    })

                st.markdown(f"""
                <div style="color:#64ffda;font-size:1rem;font-family:'Orbitron',monospace;
                            letter-spacing:1px;margin-bottom:10px;font-weight:700;">
                    ● {len(recs_enriq)} RECOMENDACIONES · USUARIO #{uid_input}
                </div>
                """, unsafe_allow_html=True)

                for rec in recs_enriq:
                    score_pct   = int(min(rec["score"] * 100, 100))
                    badge_color = "#00c2ff" if rec.get("option") == "A" else "#64ffda"
                    badge_label = "● A" if rec.get("option") == "A" else "● B"
                    precio_html = (
                        f'<div style="color:#ffb84d;font-weight:700;font-size:0.9rem;">'
                        f'${rec["price"]:.0f}</div>'
                    ) if rec.get("price") else ""
                    _name   = _html.escape(str(rec.get("name",   f"Ref. #{rec['item_id']}")))
                    _cat    = _html.escape(str(rec.get("category", "")))
                    _subcat = _html.escape(str(rec.get("subcategory", "")))
                    _desc   = _html.escape(str(rec.get("description", ""))[:100])
                    _emoji  = rec.get("emoji", "📦")
                    st.html(f"""
                    <div style="background:linear-gradient(135deg,rgba(10,20,60,0.92),rgba(20,30,80,0.92));
                                border:1px solid rgba(0,194,255,0.28);border-radius:10px;
                                padding:16px 20px;margin:6px 0;font-family:'Exo 2',sans-serif;">
                      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div style="display:flex;align-items:flex-start;gap:10px;flex:1;">
                          <span style="font-family:monospace;font-size:1.15rem;font-weight:900;
                                       color:#00c2ff;min-width:30px;">#{rec['rank']}</span>
                          <div style="flex:1;">
                            <div style="font-weight:700;color:#ffffff;font-size:1.05rem;
                                        margin-bottom:4px;">{_emoji} {_name}</div>
                            <div style="color:#a8c8e8;font-size:0.95rem;">{_cat} · {_subcat}</div>
                            <div style="color:#8ab8d8;font-size:0.9rem;margin-top:3px;
                                        font-style:italic;">{_desc}</div>
                          </div>
                        </div>
                        <div style="text-align:right;min-width:80px;padding-left:12px;">
                          {precio_html}
                          <div style="font-size:1rem;color:#64ffda;font-weight:700;margin-top:4px;">{rec['score']:.3f}</div>
                          <div style="margin-top:6px;">
                            <span style="background:rgba(0,194,255,0.15);color:{badge_color};
                                         border:1px solid {badge_color};border-radius:4px;
                                         padding:3px 10px;font-size:0.9rem;">{badge_label}</span>
                          </div>
                        </div>
                      </div>
                      <div style="height:3px;background:rgba(0,194,255,0.12);border-radius:2px;margin-top:10px;">
                        <div style="width:{score_pct}%;height:100%;
                                    background:linear-gradient(90deg,#2b5be0,#00c2ff);border-radius:2px;"></div>
                      </div>
                    </div>
                    """)
        else:
            st.info("👆 Ingresa un ID de usuario y pulsa **Generar Recomendaciones**.", icon="🔍")


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — ASISTENTE IA
# ═══════════════════════════════════════════════════════════════════════════════
elif "Asistente IA" in pagina:
    st.markdown("""
    <div style="padding:20px 0 10px;">
        <div style="font-family:'Orbitron',monospace; font-size:1.8rem; font-weight:900;
                    color:#ffffff;">🤖 ASISTENTE IA</div>
        <div style="color:#ccd6f6; font-size:1rem; margin-top:4px;">
            NEXUS AI · Explicaciones LLM + Consultor de Métricas
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Cargar motor LLM
    @st.cache_resource(show_spinner=False)
    def get_llm():
        engine, _err = try_load_engine()
        return engine

    llm: LLMEngine | None = get_llm()

    tab1, tab2 = st.tabs(["🤖 Recomendaciones + Explicación IA", "💬 Consultor de Métricas"])

    # ── TAB 1: Recomendaciones + LLM Explicación ─────────────────────────────
    with tab1:
        col_rec, col_e = st.columns([3, 2])

        with col_rec:
            st.markdown("""
            <div style="color:#64ffda;font-size:0.9rem;font-family:'Orbitron',monospace;
                        letter-spacing:1px;margin-bottom:10px;">● GENERAR RECOMENDACIONES</div>
            """, unsafe_allow_html=True)

            _demo_users_t1 = get_demo_users()
            _, _, _, _train_items_t1, _, _ = cargar_ensemble()
            uid_t1 = st.selectbox(
                "Seleccionar Usuario",
                options=_demo_users_t1,
                format_func=lambda x: f"Usuario #{x:,}  ({len(_train_items_t1.get(x, set()))} interacciones)",
                key="uid_t1",
                help="Usuarios warm con historial en el set de evaluación.",
            )
            top_t1   = st.slider("N° recomendaciones", 3, 10, 6, key="top_t1")
            run_t1   = st.button("🚀 Recomendar + Explicar", use_container_width=True,
                                  type="primary", key="run_t1")

        with col_e:
            st.markdown("""
            <div style="background:rgba(0,194,255,0.06);border:1px solid rgba(0,194,255,0.2);
                        border-radius:8px;padding:12px;margin-top:8px;">
                <div style="color:#64ffda;font-size:0.95rem;font-family:'Orbitron',monospace;
                            letter-spacing:1px;margin-bottom:8px;font-weight:700;">● NEXUS AI</div>
                <div style="color:#ccd6f6;font-size:0.95rem;line-height:1.7;">
                    Genera las recomendaciones y<br>
                    NEXUS AI explicará por qué<br>
                    estos productos fueron elegidos<br>
                    para este usuario específico.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="space-divider">', unsafe_allow_html=True)

        if run_t1:
            with st.spinner("⚡ Calculando recomendaciones..."):
                recs_raw_t1 = get_recs_live(int(uid_t1), top_n=top_t1)

            if not recs_raw_t1:
                st.warning(f"⚠ Usuario #{uid_t1} no está en el set de evaluación.")
            else:
                recs_t1 = []
                for rank, r in enumerate(recs_raw_t1, 1):
                    prod = get_product(r["item_id"])
                    recs_t1.append({
                        "rank":     rank,
                        "item_id":  r["item_id"],
                        "score":    r["score"],
                        "name":     prod.get("name",     f"Ref. #{r['item_id']}"),
                        "category": prod.get("category", "Sin categoría"),
                        "emoji":    prod.get("emoji",    "📦"),
                    })

                col_cards, col_expl = st.columns([3, 2])

                with col_cards:
                    st.markdown(f"""
                    <div style="color:#64ffda;font-size:1rem;font-family:'Orbitron',monospace;
                                letter-spacing:1px;margin-bottom:8px;font-weight:700;">
                        ● {len(recs_t1)} RECOMENDACIONES · USUARIO #{uid_t1}
                    </div>
                    """, unsafe_allow_html=True)

                    for rec in recs_t1[:6]:
                        score_pct = int(min(rec["score"] * 100, 100))
                        _name_t1  = _html.escape(str(rec.get("name", f"Ref. #{rec['item_id']}")))
                        _cat_t1   = _html.escape(str(rec.get("category", "")))
                        _emoji_t1 = rec.get("emoji", "📦")
                        st.html(f"""
                        <div style="background:linear-gradient(135deg,rgba(10,20,60,0.92),rgba(20,30,80,0.92));
                                    border:1px solid rgba(0,194,255,0.25);border-radius:8px;
                                    padding:12px 16px;margin:5px 0;font-family:'Exo 2',sans-serif;">
                          <div style="display:flex;gap:10px;align-items:center;">
                            <span style="font-family:monospace;font-size:1.1rem;font-weight:900;
                                         color:#00c2ff;min-width:28px;">#{rec['rank']}</span>
                            <div style="flex:1;">
                              <div style="font-weight:700;color:#ffffff;font-size:1rem;">
                                {_emoji_t1} {_name_t1}
                              </div>
                              <div style="color:#a8c8e8;font-size:0.92rem;">{_cat_t1}</div>
                            </div>
                            <div style="color:#64ffda;font-size:0.82rem;font-weight:700;">
                              {rec['score']:.3f}
                            </div>
                          </div>
                          <div style="height:3px;background:rgba(0,194,255,0.1);border-radius:2px;margin-top:8px;">
                            <div style="width:{score_pct}%;height:100%;
                                        background:linear-gradient(90deg,#2b5be0,#00c2ff);border-radius:2px;"></div>
                          </div>
                        </div>
                        """)

                with col_expl:
                    st.markdown("""
                    <div style="color:#64ffda;font-size:0.95rem;font-family:'Orbitron',monospace;
                                letter-spacing:1px;margin-bottom:8px;font-weight:700;">● NEXUS AI EXPLICA</div>
                    """, unsafe_allow_html=True)

                    if llm is not None:
                        with st.spinner("🤖 NEXUS AI pensando..."):
                            _, _, _, train_items, _, _ = cargar_ensemble()
                            hist_ids = list(train_items.get(int(uid_t1), set()))[:5]
                            hist_enriched = [
                                {
                                    "name":     get_product(iid).get("name", f"#{iid}"),
                                    "category": get_product(iid).get("category", "?"),
                                    "event":    "interacción",
                                }
                                for iid in hist_ids
                            ]
                            user_profile = {
                                "user_type":        "usuario warm",
                                "n_interactions":   len(train_items.get(int(uid_t1), set())),
                                "top_categories":   ", ".join(set(
                                    get_product(iid).get("category", "?")
                                    for iid in hist_ids[:3]
                                )),
                            }
                            recs_for_llm = [
                                {"name": r["name"], "category": r["category"], "score": r["score"]}
                                for r in recs_t1[:5]
                            ]
                            explicacion = llm.explain_recommendations(
                                user_id=int(uid_t1),
                                user_history=hist_enriched,
                                recommendations=recs_for_llm,
                                user_profile=user_profile,
                            )
                        st.markdown(f"""
                        <div style="background:rgba(0,194,255,0.06);border:1px solid rgba(0,194,255,0.2);
                                    border-radius:8px;padding:14px;font-family:'Exo 2',sans-serif;
                                    color:#ccd6f6;font-size:0.85rem;line-height:1.7;">
                            {_html.escape(explicacion)}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background:rgba(255,107,107,0.08);border:1px solid rgba(255,107,107,0.3);
                                    border-radius:8px;padding:12px;color:#ff6b6b;font-size:0.82rem;">
                            ⚠ NEXUS AI no disponible.<br>
                            Configurá <code>GROQ_API_KEY</code> en el archivo <code>.env</code>.
                        </div>
                        """, unsafe_allow_html=True)

    # ── TAB 2: Consultor de Métricas ──────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div style="color:#64ffda;font-size:1rem;font-family:'Orbitron',monospace;
                    letter-spacing:1px;margin-bottom:12px;font-weight:700;">● CONSULTOR DE MÉTRICAS TÉCNICAS</div>
        """, unsafe_allow_html=True)

        # Quick questions
        quick_qs = [
            "¿Por qué el NDCG@10 es 0.0431 y no más alto?",
            "¿Cómo funciona el ensemble de 3 modelos?",
            "¿Por qué falló el deep learning (SASRec, NCF)?",
            "¿Qué es el cold-start y cómo lo manejan?",
            "¿Por qué usan split temporal y no aleatorio?",
            "¿Qué es el Temporal Decay en RP3beta?",
            "¿Cómo se eligió NDCG y no otra métrica?",
            "¿Qué haría el sistema con datos de stock y precio?",
        ]

        q_sel = st.selectbox(
            "Preguntas frecuentes",
            options=["Escribe tu propia pregunta → "] + quick_qs,
            key="consult_q",
        )

        custom_q = ""
        if q_sel.startswith("Escribe"):
            custom_q = st.text_input(
                "Tu pregunta",
                placeholder="Ej: ¿Por qué no usaron BERT4Rec?",
                key="custom_consult",
            )
        else:
            custom_q = q_sel

        ask_btn = st.button("🤖 Consultar a NEXUS AI", type="primary", key="ask_metrics")

        # Historial de chat
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if ask_btn and custom_q:
            if llm is not None:
                with st.spinner("🤖 NEXUS AI pensando..."):
                    respuesta = llm.answer_metrics_question(custom_q)
                st.session_state.chat_history.append({
                    "q": custom_q, "a": respuesta, "ts": time.strftime("%H:%M:%S")
                })
            else:
                st.error("⚠ NEXUS AI no disponible. Configurá GROQ_API_KEY en .env.")

        for msg in reversed(st.session_state.chat_history):
            _q = _html.escape(msg["q"])
            _a = _html.escape(msg["a"])
            _t = _html.escape(msg["ts"])
            st.html(f"""
            <div style="margin:10px 0;font-family:'Exo 2',sans-serif;">
                <div style="background:rgba(0,194,255,0.1);border:1px solid rgba(0,194,255,0.25);
                            border-radius:8px 8px 2px 8px;padding:10px 14px;margin-bottom:4px;">
                    <span style="color:#a8c8e8;font-size:0.88rem;">🧑 Tú · {_t}</span>
                    <div style="color:#ffffff;font-size:1rem;margin-top:4px;">{_q}</div>
                </div>
                <div style="background:rgba(100,255,218,0.07);border:1px solid rgba(100,255,218,0.2);
                            border-radius:2px 8px 8px 8px;padding:10px 14px;">
                    <span style="color:#64ffda;font-size:0.88rem;font-family:'Orbitron',monospace;font-weight:700;">
                        🤖 NEXUS AI
                    </span>
                    <div style="color:#ffffff;font-size:1rem;margin-top:4px;line-height:1.7;">{_a}</div>
                </div>
            </div>
            """)

        if st.session_state.chat_history:
            if st.button("🗑 Limpiar historial", key="clear_chat"):
                st.session_state.chat_history = []
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — COMPARATIVA DE MODELOS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Comparativa" in pagina:
    st.markdown("""
    <div style="padding:20px 0 10px;">
        <div style="font-family:'Orbitron',monospace; font-size:1.8rem; font-weight:900;
                    color:#ffffff;">📊 COMPARATIVA DE MODELOS</div>
        <div style="color:#ccd6f6; font-size:1rem; margin-top:4px;">
            Evolución de NB08 a NB15 · Protocolos de evaluación estandarizados
        </div>
    </div>
    """, unsafe_allow_html=True)

    df_comp_all = cargar_todos_los_modelos()

    if df_comp_all.empty:
        st.error("No se encontraron archivos de comparación de modelos.")
    else:
        # Filtros
        _notebooks_disp = sorted(df_comp_all["notebook"].dropna().unique())
        nbs_sel = st.multiselect(
            "📂 Filtrar por Notebook",
            options=_notebooks_disp,
            default=_notebooks_disp,
            key="nb_filter",
        )

        df_filtrado = df_comp_all[df_comp_all["notebook"].isin(nbs_sel)].copy() if nbs_sel else df_comp_all.copy()
        df_filtrado = df_filtrado.dropna(subset=["ndcg10_test"]).reset_index(drop=True)

        # Sólo modelos con NDCG@10 razonable para comparar (excluir SASRec leave-one-out)
        df_std = df_filtrado[df_filtrado["ndcg10_test"] < 0.5].copy()

        col_chart, col_info = st.columns([3, 1])

        with col_chart:
            if not df_std.empty:
                _best_ndcg_idx = int(df_std["ndcg10_test"].idxmax())
                _colors_comp   = [
                    "#00c2ff" if i == _best_ndcg_idx else COLORWAY[i % len(COLORWAY)]
                    for i in range(len(df_std))
                ]
                fig_comp = go.Figure(go.Bar(
                    x=df_std["model"],
                    y=df_std["ndcg10_test"],
                    marker_color=_colors_comp,
                    text=[f"{v:.4f}" for v in df_std["ndcg10_test"]],
                    textposition="outside",
                    textfont=dict(color="#ccd6f6", size=9),
                    hovertemplate="<b>%{x}</b><br>NDCG@10: %{y:.4f}<extra></extra>",
                ))
                fig_comp.add_hline(
                    y=NDCG_BASELINE,
                    line_dash="dash",
                    line_color="#ff6b6b",
                    annotation_text=f"Baseline = {NDCG_BASELINE:.4f}",
                    annotation_font_color="#ff6b6b",
                )
                apply_space_theme(fig_comp, height=420, title="NDCG@10 por Modelo (Protocolo Estándar)")
                fig_comp.update_layout(
                    xaxis=dict(tickangle=-35, tickfont=dict(size=12, color="#ccd6f6")),
                    showlegend=False,
                )
                st.plotly_chart(fig_comp, use_container_width=True,
                                config={"responsive": True, "displayModeBar": False})
            else:
                st.info("No hay modelos con protocolo estándar en la selección actual.")

        with col_info:
            _champ = df_std.loc[df_std["ndcg10_test"].idxmax()] if not df_std.empty else None
            if _champ is not None:
                st.markdown(f"""
                <div style="background:rgba(0,194,255,0.08);border:1px solid rgba(0,194,255,0.3);
                            border-radius:8px;padding:14px;text-align:center;">
                    <div style="color:#64ffda;font-size:1rem;font-family:'Orbitron',monospace;font-weight:700;">
                        ● MODELO GANADOR
                    </div>
                    <div style="color:#ffffff;font-size:1rem;margin-top:6px;font-weight:700;">
                        {_html.escape(str(_champ['model']))}
                    </div>
                    <div style="color:#00c2ff;font-size:1.2rem;font-weight:900;
                                font-family:'Orbitron',monospace;margin-top:6px;">
                        {_champ['ndcg10_test']:.4f}
                    </div>
                    <div style="color:#64ffda;font-size:0.9rem;margin-top:4px;font-weight:600;">
                        Notebook: {_html.escape(str(_champ.get('notebook','N/A')))}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<hr class="space-divider">', unsafe_allow_html=True)
        st.markdown("#### Tabla completa")
        cols_show = [c for c in ["notebook", "model", "ndcg10_test", "delta_pct"]
                     if c in df_filtrado.columns]
        _df_show = df_filtrado[cols_show].reset_index(drop=True)
        if not _df_show.empty:
            _best_idx = int(_df_show["ndcg10_test"].idxmax()) if "ndcg10_test" in _df_show.columns else -1
            _col_labels = {"notebook": "Notebook", "model": "Modelo",
                           "ndcg10_test": "NDCG@10", "delta_pct": "Δ% vs NB13-C"}
            _thead = "".join(
                f'<th style="padding:8px 14px;border-bottom:1px solid rgba(0,194,255,0.35);'
                f'color:#00c2ff;font-family:monospace;font-size:0.82rem;'
                f'text-align:left;text-transform:uppercase;letter-spacing:1px;">'
                f'{_col_labels.get(c, c)}</th>'
                for c in cols_show
            )
            _rows_html = ""
            for idx, row in _df_show.iterrows():
                _win    = (idx == _best_idx)
                _row_bg = "rgba(0,194,255,0.07)" if _win else "transparent"
                _cells  = ""
                for c in cols_show:
                    val  = row.get(c, "")
                    base = "padding:8px 14px;font-family:'Exo 2',sans-serif;"
                    if c == "ndcg10_test":
                        try:
                            _cells += (
                                f'<td style="{base}color:#64ffda;'
                                f'font-weight:{"800" if _win else "400"};font-family:monospace;">'
                                f'{float(val):.4f}</td>'
                            )
                        except Exception:
                            _cells += f'<td style="{base}color:#ccd6f6;">{_html.escape(str(val))}</td>'
                    elif c == "delta_pct":
                        try:
                            pct   = float(val)
                            _pc   = "#64ffda" if pct >= 0 else "#ff6b6b"
                            _sign = "+" if pct >= 0 else ""
                            _cells += (
                                f'<td style="{base}color:{_pc};font-family:monospace;">'
                                f'{_sign}{pct:.1f}%</td>'
                            )
                        except Exception:
                            _cells += f'<td style="{base}color:#ccd6f6;">{_html.escape(str(val))}</td>'
                    else:
                        _txt_color = "#64ffda" if _win else "#ccd6f6"
                        _cells += f'<td style="{base}color:{_txt_color};">{_html.escape(str(val))}</td>'
                _rows_html += (
                    f'<tr style="background:{_row_bg};'
                    f'border-bottom:1px solid rgba(100,130,200,0.12);">{_cells}</tr>'
                )
            st.html(
                f'<div style="overflow-x:auto;border-radius:8px;'
                f'border:1px solid rgba(0,194,255,0.22);margin-top:8px;">'
                f'<table style="width:100%;border-collapse:collapse;font-size:0.9rem;">'
                f'<thead style="background:rgba(0,20,60,0.65);"><tr>{_thead}</tr></thead>'
                f'<tbody>{_rows_html}</tbody></table></div>'
            )
        else:
            st.info("No hay datos para mostrar con los filtros aplicados.")


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 5 — ANÁLISIS DEL DATASET
# ═══════════════════════════════════════════════════════════════════════════════
elif "Análisis del Dataset" in pagina:
    st.markdown("""
    <div style="padding:20px 0 10px;">
        <div style="font-family:'Orbitron',monospace; font-size:1.8rem; font-weight:900; color:#ffffff;">
            🪐 ANÁLISIS DEL DATASET
        </div>
        <div style="color:#ccd6f6; font-size:1rem; margin-top:4px;">
            RetailRocket E-Commerce · Mayo–Septiembre 2015
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1d, c2d, c3d = st.columns(3)
    for col, val, label, delta in [
        (c1d, "2.75M",  "EVENTOS TOTALES", "view · cart · buy"),
        (c2d, "1.4M",   "USUARIOS ÚNICOS", "1,407,580 visitor IDs"),
        (c3d, "235K",   "ÍTEMS ÚNICOS",    "235,061 productos"),
    ]:
        col.markdown(f"""
        <div style="background:rgba(10,20,60,0.7);border:1px solid rgba(0,194,255,0.25);
                    border-radius:10px;padding:16px;text-align:center;">
            <div style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:900;
                        color:#00c2ff;">{val}</div>
            <div style="color:#64ffda;font-size:0.85rem;font-weight:600;
                        letter-spacing:1px;margin-top:4px;">{label}</div>
            <div style="color:#ccd6f6;font-size:0.82rem;margin-top:3px;">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="space-divider">', unsafe_allow_html=True)

    # ── Gráfico: Distribución de tipos de eventos ─────────────────────────────
    col_ev1, col_ev2 = st.columns(2)

    with col_ev1:
        _etiquetas = ["Views", "Add to Cart", "Transacciones"]
        _valores   = [_N_VIEWS, _N_CART, _N_TX]
        fig_ev = go.Figure(go.Bar(
            x=_etiquetas,
            y=_valores,
            marker_color=[COLORWAY[0], COLORWAY[2], COLORWAY[1]],
            text=[f"{v:,}" for v in _valores],
            textposition="outside",
            textfont=dict(color="#ccd6f6", size=11),
        ))
        apply_space_theme(fig_ev, height=320, title="Eventos por Tipo de Interacción")
        fig_ev.update_layout(showlegend=False)
        st.plotly_chart(fig_ev, use_container_width=True,
                        config={"responsive": True, "displayModeBar": False})

    with col_ev2:
        # Tasa de conversión del funnel
        _cr_cart = _N_CART  / _N_VIEWS * 100
        _cr_tx   = _N_TX    / _N_CART  * 100
        fig_pl = go.Figure()
        fig_pl.add_trace(go.Bar(
            x=["View→Cart", "Cart→Compra"],
            y=[_cr_cart, _cr_tx],
            marker_color=[COLORWAY[2], COLORWAY[1]],
            text=[f"{_cr_cart:.2f}%", f"{_cr_tx:.1f}%"],
            textposition="outside",
            textfont=dict(color="#ccd6f6"),
        ))
        apply_space_theme(fig_pl, height=320, title="Tasa de Conversión (%)")
        fig_pl.update_layout(showlegend=False)
        st.plotly_chart(fig_pl, use_container_width=True,
                        config={"responsive": True, "displayModeBar": False})

    st.markdown('<hr class="space-divider">', unsafe_allow_html=True)

    # ── Gráfico: Distribución de actividad de usuarios ────────────────────────
    @st.cache_data(show_spinner=False)
    def _cargar_funnel_metrics():
        p = INTERIM_DIR / "cp03_funnel_metrics.parquet"
        if not p.exists():
            return None
        return pd.read_parquet(p, columns=["n_views", "n_addtocarts", "n_transactions"])

    col_usr, col_cats = st.columns(2)

    with col_usr:
        df_f = _cargar_funnel_metrics()
        if df_f is not None:
            _n_events_user = df_f["n_views"] + df_f["n_addtocarts"] + df_f["n_transactions"]
            _bins = [1, 2, 3, 5, 10, 20, 50, 100, 500, 999999]
            _labs = ["1", "2", "3-4", "5-9", "10-19", "20-49", "50-99", "100-499", "500+"]
            _cnt  = pd.cut(
                _n_events_user[_n_events_user >= 1],
                bins=[0] + _bins,
                labels=["0"] + _labs,
            ).value_counts().sort_index()
            # Quitar el bin "0"
            _cnt = _cnt[_cnt.index != "0"]
            fig_usr = go.Figure(go.Bar(
                x=_cnt.index.tolist(),
                y=_cnt.values.tolist(),
                marker_color=COLORWAY[:len(_cnt)],
                text=[f"{v:,}" for v in _cnt.values],
                textposition="outside",
                textfont=dict(color="#ccd6f6", size=9),
            ))
            apply_space_theme(fig_usr, height=320, title="Distribución de Actividad de Usuarios (eventos)")
        else:
            # Fallback: datos aproximados conocidos del dataset
            _act_labels = ["1 evento", "2-4", "5-9", "10-24", "25-49", "50+"]
            _act_vals   = [950000, 280000, 110000, 50000, 12000, 5580]
            fig_usr = go.Figure(go.Bar(
                x=_act_labels, y=_act_vals,
                marker_color=COLORWAY[:len(_act_labels)],
                text=[f"{v:,}" for v in _act_vals],
                textposition="outside",
                textfont=dict(color="#ccd6f6", size=10),
            ))
            apply_space_theme(fig_usr, height=320, title="Distribución de Actividad de Usuarios")
        fig_usr.update_layout(showlegend=False)
        st.plotly_chart(fig_usr, use_container_width=True,
                        config={"responsive": True, "displayModeBar": False})

    with col_cats:
        # Top categorías desde item_features.csv
        @st.cache_data(show_spinner=False)
        def _top_categorias():
            p = DATA_DIR / "item_features.csv"
            if not p.exists():
                return pd.DataFrame()
            df = pd.read_csv(p, usecols=["itemid", "root_category", "n_views_item"])
            return (
                df.groupby("root_category")
                .agg(n_items=("itemid", "count"), n_views=("n_views_item", "sum"))
                .reset_index()
                .sort_values("n_items", ascending=False)
                .head(12)
            )

        df_cats = _top_categorias()
        if not df_cats.empty:
            fig_p4 = go.Figure(go.Bar(
                y=[str(int(c)) if pd.notna(c) else "N/A" for c in df_cats["root_category"]],
                x=df_cats["n_items"].tolist(),
                orientation="h",
                marker_color=COLORWAY[:len(df_cats)],
                text=[f"{v:,}" for v in df_cats["n_items"]],
                textposition="outside",
                textfont=dict(color="#ccd6f6", size=9),
            ))
            apply_space_theme(fig_p4, height=320, title="Top Categorías por N° de Ítems")
        else:
            fig_p4 = go.Figure()
            apply_space_theme(fig_p4, height=320, title="Top Categorías")
        fig_p4.update_layout(showlegend=False)
        st.plotly_chart(fig_p4, use_container_width=True,
                        config={"responsive": True, "displayModeBar": False})

    st.markdown('<hr class="space-divider">', unsafe_allow_html=True)

    # ── Tabla de estadísticas clave ────────────────────────────────────────────
    df_stats = pd.DataFrame({
        "Métrica": [
            "Total eventos",
            "Usuarios únicos",
            "Ítems únicos",
            "Sparsidad",
            "Ratio view→cart",
            "Ratio cart→compra",
            "Período temporal",
            "Split train/test",
        ],
        "Valor": [
            f"{_N_EVENTS:,}",
            f"{_N_USERS:,}",
            f"{_N_ITEMS:,}",
            f"{_SPARSITY:.4f}%",
            f"{_N_CART/_N_VIEWS*100:.2f}%",
            f"{_N_TX/_N_CART*100:.1f}%",
            "May–Sep 2015",
            f"<{CUTOFF_DATE} / ≥{CUTOFF_DATE}",
        ],
        "Detalle": [
            "view + addtocart + transaction",
            "Visitor IDs con al menos 1 evento",
            "Items con al menos 1 interacción",
            "Datos faltantes = 99.9994% de la matriz",
            "Solo 2.60% de views generan un carrito",
            "32.4% de carritos resultan en compra",
            "Datos: RetailRocket E-Commerce",
            f"Train: 82.22% / Test: 17.78%",
        ],
    })

    _stat_rows = ""
    for _, srow in df_stats.iterrows():
        _m = _html.escape(str(srow["Métrica"]))
        _v = _html.escape(str(srow["Valor"]))
        _d = _html.escape(str(srow["Detalle"]))
        _stat_rows += (
            f'<tr style="border-bottom:1px solid rgba(100,130,200,0.15);">'
            f'<td style="padding:8px 14px;color:#64ffda;font-family:monospace;">{_m}</td>'
            f'<td style="padding:8px 14px;color:#00c2ff;font-weight:700;font-family:monospace;">{_v}</td>'
            f'<td style="padding:8px 14px;color:#ccd6f6;font-size:0.95rem;">{_d}</td>'
            f'</tr>'
        )
    st.html(
        f'<div style="overflow-x:auto;border-radius:8px;border:1px solid rgba(0,194,255,0.22);margin-top:8px;">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead style="background:rgba(0,20,60,0.65);">'
        f'<tr>'
        f'<th style="padding:8px 14px;color:#00c2ff;font-family:monospace;text-align:left;">Métrica</th>'
        f'<th style="padding:8px 14px;color:#00c2ff;font-family:monospace;text-align:left;">Valor</th>'
        f'<th style="padding:8px 14px;color:#00c2ff;font-family:monospace;text-align:left;">Detalle</th>'
        f'</tr></thead>'
        f'<tbody>{_stat_rows}</tbody></table></div>'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 6 — MÉTRICAS DEL SISTEMA
# ═══════════════════════════════════════════════════════════════════════════════
elif "Métricas del Sistema" in pagina:
    st.markdown("""
    <div style="padding:20px 0 10px;">
        <div style="font-family:'Orbitron',monospace; font-size:1.8rem; font-weight:900;
                    color:#ffffff;">⭐ MÉTRICAS DEL SISTEMA</div>
        <div style="color:#ccd6f6; font-size:1rem; margin-top:4px;">
            Evaluación técnica completa · Protocolo warm · N_EVAL=3,000
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs principales
    km1, km2, km3, km4 = st.columns(4)
    for col, val, label, sub in [
        (km1, "0.0431",  "NDCG@10",          "NB15v2 · conjunto de test"),
        (km2, "0.0068",  "PRECISION@10",      "NB15v2 evaluación"),
        (km3, "0.0415",  "RECALL@10",         "NB15v2 evaluación"),
        (km4, "99.9994%","SPARSIDAD",          "Extrema · 1 ítem/usuario mediana"),
    ]:
        col.markdown(f"""
        <div style="background:rgba(10,20,60,0.7);border:1px solid rgba(0,194,255,0.25);
                    border-radius:10px;padding:16px;text-align:center;">
            <div style="font-family:'Orbitron',monospace;font-size:1.3rem;font-weight:900;
                        color:#00c2ff;">{val}</div>
            <div style="color:#64ffda;font-size:0.85rem;font-weight:600;
                        letter-spacing:1px;margin-top:4px;">{label}</div>
            <div style="color:#ccd6f6;font-size:0.82rem;margin-top:3px;">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="space-divider">', unsafe_allow_html=True)

    col_g, col_m = st.columns([1, 2])

    with col_g:
        # ── Gauge de NDCG@10 ──────────────────────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=NDCG_CHAMPION * 100,
            number=dict(
                suffix="%",
                font=dict(
                    family="Orbitron,monospace",
                    color="#00c2ff",
                    size=28,
                ),
            ),
            delta=dict(
                reference=NDCG_BASELINE * 100,
                valueformat=".2f",
                suffix="%",
                increasing=dict(color="#64ffda"),
            ),
            gauge=dict(
                axis=dict(range=[0, 10], tickcolor="#ccd6f6",
                          tickfont=dict(color="#ccd6f6", size=12)),
                bar=dict(color="#00c2ff", thickness=0.25),
                bgcolor="rgba(10,20,60,0.5)",
                borderwidth=1,
                bordercolor="rgba(0,194,255,0.3)",
                steps=[
                    dict(range=[0, 2.86], color="rgba(255,107,107,0.2)"),
                    dict(range=[2.86, 4.07], color="rgba(255,184,77,0.2)"),
                    dict(range=[4.07, 4.31], color="rgba(100,255,218,0.2)"),
                    dict(range=[4.31, 10], color="rgba(0,194,255,0.05)"),
                ],
                threshold=dict(
                    line=dict(color="#64ffda", width=3),
                    thickness=0.75,
                    value=NDCG_CHAMPION * 100,
                ),
            ),
            title=dict(
                text="NDCG@10 Campeón (%)",
                font=dict(family="Orbitron,monospace", color="#ffffff", size=12),
            ),
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccd6f6"),
            height=280,
            margin=dict(t=40, b=10, l=20, r=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True,
                        config={"responsive": True, "displayModeBar": False})

    with col_m:
        # ── Comparativa de métricas: modelos clave ────────────────────────────
        _modelos_met = ["NB13-C\nRP3+TD", "NB14-E4\nEns Spearman", "NB15v2\nMega-Ensemble"]
        _met_ndcg    = [0.02859,  0.04069,  0.04310]
        _met_prec    = [0.00682,  0.00890,  0.00682]
        _met_rec     = [0.04141,  0.0543,   0.04150]

        fig_met = go.Figure()
        fig_met.add_trace(go.Bar(
            name="NDCG@10",
            x=_modelos_met, y=_met_ndcg,
            marker_color=COLORWAY[0],
            text=[f"{v:.4f}" for v in _met_ndcg],
            textposition="outside",
            textfont=dict(color="#ccd6f6", size=11),
        ))
        fig_met.add_trace(go.Bar(
            name="Precision@10",
            x=_modelos_met, y=_met_prec,
            marker_color=COLORWAY[2],
            text=[f"{v:.4f}" for v in _met_prec],
            textposition="outside",
            textfont=dict(color="#ccd6f6", size=11),
        ))
        fig_met.add_trace(go.Bar(
            name="Recall@10",
            x=_modelos_met, y=_met_rec,
            marker_color=COLORWAY[1],
            text=[f"{v:.4f}" for v in _met_rec],
            textposition="outside",
            textfont=dict(color="#ccd6f6", size=11),
        ))
        apply_space_theme(fig_met, height=280, title="Métricas de Evaluación — Modelos Clave")
        fig_met.update_layout(barmode="group")
        st.plotly_chart(fig_met, use_container_width=True,
                        config={"responsive": True, "displayModeBar": False})

    st.markdown('<hr class="space-divider">', unsafe_allow_html=True)

    col_corr, col_tl = st.columns(2)

    with col_corr:
        # ── Correlación entre modelos del ensemble ────────────────────────────
        _corr_labels = ["RP3+MB+TD", "EASE^R-500", "RP3+TD"]
        _corr_matrix = np.array([
            [1.000, 0.216, 0.782],
            [0.216, 1.000, 0.198],
            [0.782, 0.198, 1.000],
        ])
        fig_corr = go.Figure(go.Heatmap(
            z=_corr_matrix,
            x=_corr_labels,
            y=_corr_labels,
            colorscale=[
                [0.0,  "rgba(5,10,26,1)"],
                [0.3,  "rgba(43,91,224,0.6)"],
                [0.7,  "rgba(0,194,255,0.6)"],
                [1.0,  "rgba(100,255,218,0.9)"],
            ],
            zmin=0, zmax=1,
            text=np.round(_corr_matrix, 3),
            texttemplate="%{text}",
            textfont=dict(color="#ffffff", size=12),
            showscale=True,
            colorbar=dict(
                tickfont=dict(color="#ccd6f6", size=12),
                title=dict(text="ρ", font=dict(color="#ccd6f6")),
            ),
        ))
        apply_space_theme(fig_corr, height=300, title="Correlación Spearman entre Modelos")
        st.plotly_chart(fig_corr, use_container_width=True,
                        config={"responsive": True, "displayModeBar": False})

    with col_tl:
        # ── Timeline de mejoras de NDCG@10 ────────────────────────────────────
        _tl_nb   = ["NB07", "NB08", "NB09", "NB10", "NB11", "NB13-C", "NB14", "NB15v2"]
        _tl_ndcg = [0.00809, 0.00934, 0.02576, 0.02545, 0.02603, 0.02859, 0.04069, 0.04310]
        _tl_desc = [
            "SVD (k=50)", "SVD+TD+IPS", "RP3beta", "Mult-VAE",
            "Ens RP3+EASE", "RP3+TD (decay=0.01)",
            "Ensemble Spearman", "Mega-Ensemble NB15v2",
        ]

        fig_tl = go.Figure()
        fig_tl.add_trace(go.Scatter(
            x=_tl_nb, y=_tl_ndcg,
            mode="lines+markers+text",
            line=dict(color=COLORWAY[0], width=2),
            marker=dict(
                color=[COLORWAY[0] if nd < NDCG_CHAMPION else "#64ffda" for nd in _tl_ndcg],
                size=[8 if nd < NDCG_CHAMPION else 14 for nd in _tl_ndcg],
                line=dict(color="rgba(0,194,255,0.5)", width=1),
            ),
            text=_tl_nb,
            textposition="top center",
            textfont=dict(color="#ccd6f6", size=12),
            hovertext=_tl_desc,
            hovertemplate="<b>%{x}</b><br>NDCG@10: %{y:.4f}<br>%{hovertext}<extra></extra>",
        ))
        fig_tl.add_hline(
            y=NDCG_BASELINE,
            line_dash="dash",
            line_color="#ff6b6b",
            annotation_text="Baseline",
            annotation_font_color="#ff6b6b",
        )
        apply_space_theme(fig_tl, height=300, title="Evolución NDCG@10 por Notebook")
        fig_tl.update_layout(showlegend=False)
        st.plotly_chart(fig_tl, use_container_width=True,
                        config={"responsive": True, "displayModeBar": False})

    st.markdown('<hr class="space-divider">', unsafe_allow_html=True)

    # ── Tabla resumen de resultados del NB15v2 ────────────────────────────────
    nb15_res = cargar_nb15_results()
    if nb15_res:
        _champ_res = nb15_res.get("NB15v2_result", {})
        _sel_mods  = _champ_res.get("selected_models", ["rp3_mb_td", "ease_500", "rp3_td"])
        st.markdown("""
        <div style="color:#64ffda;font-size:1rem;font-family:'Orbitron',monospace;
                    letter-spacing:1px;margin-bottom:12px;font-weight:700;">● RESULTADO OFICIAL NB15v2</div>
        """, unsafe_allow_html=True)
        _res_cols = st.columns(4)
        for rc, rval, rlbl in [
            (_res_cols[0], f"{float(_champ_res.get('ndcg10_test', NDCG_CHAMPION)):.4f}", "NDCG@10 Test"),
            (_res_cols[1], f"{float(_champ_res.get('ndcg10_val',  0.0217)):.4f}",        "NDCG@10 Val"),
            (_res_cols[2], f"+{float(_champ_res.get('delta_pct_vs_nb12', 50.76)):.1f}%", "Δ% vs NB13-C"),
            (_res_cols[3], f"+{float(_champ_res.get('delta_pct_vs_nb14', 5.93)):.1f}%",  "Δ% vs NB14"),
        ]:
            rc.markdown(f"""
            <div style="background:rgba(10,20,60,0.7);border:1px solid rgba(0,194,255,0.2);
                        border-radius:8px;padding:12px;text-align:center;">
                <div style="font-family:'Orbitron',monospace;font-size:1.3rem;font-weight:900;
                            color:#00c2ff;">{rval}</div>
                <div style="color:#ccd6f6;font-size:0.92rem;margin-top:6px;font-weight:600;">{rlbl}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:rgba(0,20,60,0.5);border:1px solid rgba(100,255,218,0.2);
                    border-radius:8px;padding:14px;margin-top:12px;">
            <div style="color:#64ffda;font-size:1rem;font-family:'Orbitron',monospace;
                        letter-spacing:1px;margin-bottom:8px;font-weight:700;">● MODELOS SELECCIONADOS</div>
            <div style="color:#ffffff;font-size:1rem;line-height:1.7;font-weight:600;">
                {" + ".join([_html.escape(str(m)) for m in _sel_mods])}
            </div>
            <div style="color:#a8c8e8;font-size:0.95rem;margin-top:6px;">
                Método: Greedy Forward Selection + Optuna (100 trials)
            </div>
        </div>
        """, unsafe_allow_html=True)
