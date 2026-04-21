"""
styles.py — Estilos globales del dashboard nexus-recsys
=========================================================
Temática: universo espacial (deep space, nebulosas, estrellas fugaces)
Fuentes:  Orbitron (títulos), Exo 2 (cuerpo)
Paleta:   Azul profundo · Cyan neón (#00c2ff) · Turquesa (#64ffda)
"""

SPACE_CSS = """
<style>
/* ── FUENTES ─────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

/* ── OCULTAR CHROME DE STREAMLIT (barra blanca superior) ──────────────────── */
/* Mantener altura natural: el botón de expandir sidebar (stExpandSidebarButton) */
/* vive dentro del header — height:0 lo haría inaccesible */
header[data-testid="stHeader"] {
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    border-bottom: none !important;
}
[data-testid="stDecoration"]    { display: none !important; }
#MainMenu                       { visibility: hidden !important; }
footer                          { visibility: hidden !important; }
/* Reducir padding top que Streamlit agrega por el header */
.block-container                { padding-top: 1.5rem !important; font-size: 1.05rem !important; }

/* ── stToolbar: ocultar visualmente PERO NO con display:none ─────────────────
   CRÍTICO: stExpandSidebarButton vive DENTRO de stToolbar (código fuente
   Streamlit 1.56 confirmado). display:none en stToolbar mata el botón expand.
   Solución: visibility:hidden en hijos del toolbar, excepto stExpandSidebarButton. */
[data-testid="stToolbar"] { background: transparent !important; }
[data-testid="stToolbar"] * { visibility: hidden !important; }
/* Solo el wrapper y el button visible — NO el span (*). El span queda hidden por la regla anterior. */
[data-testid="stExpandSidebarButton"] { visibility: visible !important; }
[data-testid="stExpandSidebarButton"] button { visibility: visible !important; }

/* ── FONDO ESPACIAL ──────────────────────────────────────────────────────── */
.stApp {
    background: radial-gradient(ellipse at 20% 50%, #0d1b4b 0%, #050a1a 50%, #000000 100%);
    font-family: 'Exo 2', sans-serif;
}

/* Estrellas estáticas */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        radial-gradient(1px 1px at 10% 15%, rgba(255,255,255,0.8) 0%, transparent 100%),
        radial-gradient(1px 1px at 30% 45%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 50% 70%, rgba(255,255,255,0.9) 0%, transparent 100%),
        radial-gradient(1px 1px at 70% 20%, rgba(255,255,255,0.7) 0%, transparent 100%),
        radial-gradient(1px 1px at 85% 55%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 20% 80%, rgba(255,255,255,0.8) 0%, transparent 100%),
        radial-gradient(1px 1px at 60% 35%, rgba(200,220,255,0.7) 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 75%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(2px 2px at 40% 60%, rgba(100,180,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 75% 40%, rgba(255,255,255,0.9) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* ── ANIMACIONES ─────────────────────────────────────────────────────────── */
@keyframes shooting-star {
    0%   { transform: translateX(0) translateY(0) rotate(-45deg); opacity: 1; width: 0; }
    70%  { opacity: 1; }
    100% { transform: translateX(-600px) translateY(600px) rotate(-45deg); opacity: 0; width: 150px; }
}
@keyframes shooting-star-2 {
    0%   { transform: translateX(0) translateY(0) rotate(-35deg); opacity: 0.8; width: 0; }
    80%  { opacity: 0.8; }
    100% { transform: translateX(-400px) translateY(400px) rotate(-35deg); opacity: 0; width: 100px; }
}
@keyframes nebula-pulse {
    0%, 100% { opacity: 0.15; }
    50%       { opacity: 0.25; }
}
@keyframes float-up {
    from { transform: translateY(20px); opacity: 0; }
    to   { transform: translateY(0);    opacity: 1; }
}
@keyframes glow-pulse {
    0%, 100% { box-shadow: 0 0 20px rgba(0, 194, 255, 0.3); }
    50%       { box-shadow: 0 0 40px rgba(0, 194, 255, 0.6), 0 0 60px rgba(0, 194, 255, 0.2); }
}
@keyframes twinkle {
    0%, 100% { opacity: 0.3; transform: scale(0.8); }
    50%       { opacity: 1;   transform: scale(1.2); }
}

/* Estrellas fugaces */
.shooting-star-1 {
    position: fixed; top: 15%; right: 20%;
    width: 0; height: 2px;
    background: linear-gradient(to right, transparent, #ffffff, #00c2ff);
    border-radius: 50px;
    animation: shooting-star 4s ease-in-out 2s infinite;
    z-index: 1; pointer-events: none;
}
.shooting-star-2 {
    position: fixed; top: 35%; right: 60%;
    width: 0; height: 1.5px;
    background: linear-gradient(to right, transparent, #ffffff, #64ffda);
    border-radius: 50px;
    animation: shooting-star-2 6s ease-in-out 5s infinite;
    z-index: 1; pointer-events: none;
}
.shooting-star-3 {
    position: fixed; top: 5%; right: 40%;
    width: 0; height: 1px;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.9));
    border-radius: 50px;
    animation: shooting-star 8s ease-in-out 8s infinite;
    z-index: 1; pointer-events: none;
}

/* Nebulosas de fondo */
.nebula-bg {
    position: fixed; top: -20%; right: -10%;
    width: 600px; height: 600px;
    background: radial-gradient(ellipse, rgba(43,91,224,0.12) 0%, rgba(100,255,218,0.05) 40%, transparent 70%);
    border-radius: 50%;
    animation: nebula-pulse 8s ease-in-out infinite;
    pointer-events: none; z-index: 0;
}
.nebula-bg-2 {
    position: fixed; bottom: -10%; left: -5%;
    width: 400px; height: 400px;
    background: radial-gradient(ellipse, rgba(100,0,200,0.08) 0%, rgba(0,194,255,0.04) 50%, transparent 70%);
    border-radius: 50%;
    animation: nebula-pulse 12s ease-in-out 4s infinite;
    pointer-events: none; z-index: 0;
}

/* ── SIDEBAR ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050a1a 0%, #0a1540 50%, #050a1a 100%) !important;
    border-right: 1px solid rgba(0,194,255,0.35) !important;
    z-index: 999 !important;
}
[data-testid="stSidebar"] * { color: #ccd6f6 !important; }
[data-testid="stSidebarNav"]  { z-index: 999 !important; }
[data-testid="stSidebarContent"] { z-index: 999 !important; }

/* ── BOTÓN COLAPSAR sidebar: data-testid="stSidebarCollapseButton" ───────────
   font-size:0 en el span oculta el texto (confirmado funcionando). */
[data-testid="stSidebarCollapseButton"] button span { font-size: 0 !important; line-height: 0 !important; }
[data-testid="stSidebarCollapseButton"] button {
    position: relative !important; min-width: 2rem !important; min-height: 2rem !important;
    background: transparent !important; border: none !important; cursor: pointer !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    overflow: visible !important;
}
[data-testid="stSidebarCollapseButton"] button::after {
    content: "" !important; display: block !important;
    width: 0 !important; height: 0 !important;
    border-top: 6px solid transparent !important;
    border-bottom: 6px solid transparent !important;
    border-right: 9px solid rgba(0,194,255,0.85) !important;
    pointer-events: none !important;
}

/* ── BOTÓN EXPANDIR sidebar: data-testid="stExpandSidebarButton" ──────────
   CRÍTICO: el testid está en el <button> directamente, no en un wrapper.
   Por eso hay que apuntar a [data-testid="stExpandSidebarButton"] directo. */
[data-testid="stExpandSidebarButton"] span { font-size: 0 !important; line-height: 0 !important; }
[data-testid="stExpandSidebarButton"] {
    position: relative !important; min-width: 2rem !important; min-height: 2rem !important;
    background-color: transparent !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 9 12'%3E%3Cpolygon points='0,0 9,6 0,12' fill='%2300c2ff'/%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 9px 12px !important;
    border: none !important;
    cursor: pointer !important;
    visibility: visible !important;
}

[data-testid="stSidebar"]::before {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        radial-gradient(1px 1px at 20% 20%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 60% 50%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 80% 80%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 40% 70%, rgba(100,180,255,0.4) 0%, transparent 100%);
    pointer-events: none;
}

/* ── MÉTRICAS / KPI CARDS ────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, rgba(10,20,60,0.9) 0%, rgba(5,10,30,0.95) 100%);
    border: 1px solid rgba(0,194,255,0.3);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    animation: glow-pulse 3s ease-in-out infinite, float-up 0.6s ease-out;
    backdrop-filter: blur(10px);
    transition: transform 0.3s ease, border-color 0.3s ease;
    cursor: default;
}
.metric-card:hover {
    transform: translateY(-4px) scale(1.02);
    border-color: rgba(0,194,255,0.7);
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 2.8rem; font-weight: 900;
    color: #00c2ff;
    text-shadow: 0 0 20px rgba(0,194,255,0.5);
    margin: 8px 0;
}
.metric-label {
    font-family: 'Exo 2', sans-serif;
    font-size: 1.1rem; color: #ccd6f6;
    letter-spacing: 1px; text-transform: uppercase;
}
.metric-delta { font-size: 1.1rem; color: #64ffda; font-weight: 700; }

/* ── TÍTULOS ─────────────────────────────────────────────────────────────── */
h1 {
    font-family: 'Orbitron', monospace !important;
    color: #ffffff !important;
    font-size: 2.2rem !important; font-weight: 900 !important;
    text-shadow: 0 0 30px rgba(0,194,255,0.4);
}
h2 {
    font-family: 'Orbitron', monospace !important;
    color: #ffffff !important;
    font-size: 1.5rem !important; font-weight: 700 !important;
    text-shadow: 0 0 20px rgba(0,194,255,0.3);
}
h3 {
    font-family: 'Orbitron', monospace !important;
    color: #ccd6f6 !important;
    font-size: 1.3rem !important; font-weight: 600 !important;
}
p, li, td, th, label, span {
    font-family: 'Exo 2', sans-serif !important;
    color: #ccd6f6 !important;
    font-size: 1.05rem !important;
}

/* ── CARDS DE RECOMENDACIÓN ──────────────────────────────────────────────── */
.rec-card {
    background: linear-gradient(135deg, rgba(10,20,60,0.85), rgba(20,30,80,0.85));
    border: 1px solid rgba(0,194,255,0.25);
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    transition: all 0.3s ease;
    animation: float-up 0.4s ease-out;
    backdrop-filter: blur(8px);
}
.rec-card:hover {
    border-color: rgba(0,194,255,0.6);
    transform: translateX(6px);
    background: linear-gradient(135deg, rgba(15,30,80,0.9), rgba(25,40,100,0.9));
}
.rec-rank {
    font-family: 'Orbitron', monospace;
    font-size: 1.65rem; font-weight: 900; color: #00c2ff;
    min-width: 40px; display: inline-block;
}
.rec-name { font-size: 1.2rem; font-weight: 700; color: #ffffff; }
.rec-category { font-size: 1rem; color: #a8c8e8; margin-top: 2px; }
.rec-price { font-size: 1.1rem; color: #64ffda; font-weight: 700; }
.rec-score-bar {
    height: 4px;
    background: linear-gradient(90deg, #00c2ff, #64ffda);
    border-radius: 2px;
    transition: width 0.8s ease;
    margin-top: 8px;
}
.rec-desc { font-size: 0.95rem; color: #a8c8e8; font-style: italic; margin-top: 4px; }

/* ── LLM CHAT INTERFACE ──────────────────────────────────────────────────── */
.llm-container {
    background: linear-gradient(135deg, rgba(5,10,30,0.95), rgba(10,20,60,0.95));
    border: 1px solid rgba(100,255,218,0.3);
    border-radius: 16px; padding: 24px;
    animation: glow-pulse 4s ease-in-out infinite;
    backdrop-filter: blur(12px);
}
.chat-message-user {
    background: linear-gradient(135deg, rgba(43,91,224,0.3), rgba(43,91,224,0.1));
    border: 1px solid rgba(43,91,224,0.4);
    border-radius: 12px 12px 0 12px;
    padding: 14px 18px; margin: 8px 0 8px 40px;
    color: #ccd6f6; font-family: 'Exo 2', sans-serif;
}
.chat-message-ai {
    background: linear-gradient(135deg, rgba(0,194,255,0.1), rgba(100,255,218,0.05));
    border: 1px solid rgba(0,194,255,0.25);
    border-radius: 12px 12px 12px 0;
    padding: 14px 18px; margin: 8px 40px 8px 0;
    color: #e6f1ff; font-family: 'Exo 2', sans-serif;
    line-height: 1.7;
}

/* ── BOTONES ─────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #0a3d7a, #1a5fa8) !important;
    border: 1px solid rgba(0,194,255,0.5) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    font-size: 1rem !important;
    letter-spacing: 1px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1a5fa8, #00c2ff) !important;
    border-color: rgba(0,194,255,0.9) !important;
    box-shadow: 0 0 20px rgba(0,194,255,0.4) !important;
    transform: translateY(-2px) !important;
}

/* ── INPUTS ──────────────────────────────────────────────────────────────── */
/* Contenedor baseweb (number input, text input) */
div[data-baseweb="input"] {
    background: rgba(10,20,60,0.9) !important;
    border: 1px solid rgba(0,194,255,0.35) !important;
    border-radius: 8px !important;
}
/* Forzar fondo oscuro en el div que envuelve el baseweb input — elimina el blanco del tema */
div[data-baseweb="input"] > div,
div[data-baseweb="input"] > div > div {
    background: rgba(10,20,60,0.9) !important;
    background-color: rgba(10,20,60,0.9) !important;
}
div[data-baseweb="input"]:focus-within {
    border-color: rgba(0,194,255,0.8) !important;
    box-shadow: 0 0 12px rgba(0,194,255,0.2) !important;
}
div[data-baseweb="input"] input {
    background: transparent !important;
    color: #e8eaf6 !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 1rem !important;
}
/* Número: botones +/- */
div[data-baseweb="input"] button {
    background: rgba(0,194,255,0.12) !important;
    color: #00c2ff !important;
    border: none !important;
}
.stTextInput > div > input, .stTextArea textarea {
    background: rgba(10,20,60,0.8) !important;
    border: 1px solid rgba(0,194,255,0.3) !important;
    border-radius: 8px !important;
    color: #e8eaf6 !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 1rem !important;
}
/* Contenedor externo del text_input — el div que tiene el fondo blanco */
.stTextInput > div,
.stTextInput > div > div {
    background: rgba(10,20,60,0.85) !important;
    border-radius: 8px !important;
}
/* Raíz del widget en Streamlit 1.56 — evita fondo blanco del tema */
[data-testid="stTextInputRootElement"],
[data-testid="stTextInputRootElement"] > div,
[data-testid="InputInstructions"],
section[data-testid="stMain"] div[data-baseweb="input"],
section[data-testid="stMain"] div[data-baseweb="input"] > div {
    background: rgba(10,20,60,0.85) !important;
    border: 1px solid rgba(0,194,255,0.3) !important;
    border-radius: 8px !important;
}
section[data-testid="stMain"] div[data-baseweb="input"] input,
[data-testid="stTextInputRootElement"] input {
    background: transparent !important;
    color: #e8eaf6 !important;
    font-size: 1rem !important;
}
/* TextArea también */
.stTextArea textarea,
section[data-testid="stMain"] textarea {
    background: rgba(10,20,60,0.85) !important;
    border: 1px solid rgba(0,194,255,0.3) !important;
    color: #e8eaf6 !important;
    border-radius: 8px !important;
}
.stTextInput > div > input:focus, .stTextArea textarea:focus {
    border-color: rgba(0,194,255,0.8) !important;
    box-shadow: 0 0 15px rgba(0,194,255,0.2) !important;
}
/* Selectbox: capa exterior baseweb */
div[data-baseweb="select"] > div:first-child {
    background: rgba(10,20,60,0.9) !important;
    border: 1px solid rgba(0,194,255,0.35) !important;
    border-radius: 8px !important;
    color: #e8eaf6 !important;
}
div[data-baseweb="select"] [data-baseweb="control"] {
    background: rgba(10,20,60,0.9) !important;
    border: 1px solid rgba(0,194,255,0.35) !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] svg {
    color: #e8eaf6 !important;
    fill: #00c2ff !important;
}
[data-testid="stSelectbox"] > div > div[data-baseweb="select"] > div {
    background-color: rgba(10,20,60,0.9) !important;
    border: 1px solid rgba(0,194,255,0.35) !important;
    border-radius: 8px !important;
    color: #e8eaf6 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #e8eaf6 !important;
}
/* Number Input específico */
[data-testid="stNumberInput"] > label + div,
[data-testid="stNumberInput"] > div > div {
    background: rgba(10,20,60,0.9) !important;
    border-radius: 8px !important;
}
[data-testid="stNumberInput"] input {
    background: rgba(10,20,60,0.9) !important;
    border: 1px solid rgba(0,194,255,0.3) !important;
    color: #e8eaf6 !important;
    font-size: 1rem !important;
}

/* ── DIVIDER ─────────────────────────────────────────────────────────────── */
.space-divider {
    border: none; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,194,255,0.4), transparent);
    margin: 24px 0;
}

/* ── LABELS DE WIDGETS (Streamlit) — legibilidad en demos ────────────────── */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
.stSlider label,
.stSelectbox label,
.stNumberInput label,
.stTextInput label,
.stTextArea label {
    font-family: 'Exo 2', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #ccd6f6 !important;
}
/* Valor del slider */
[data-testid="stSlider"] [data-testid="stThumbValue"],
[data-testid="stSlider"] span {
    font-size: 0.95rem !important;
    color: #00c2ff !important;
}
/* Texto de ayuda (help tooltip) */
[data-testid="stTooltipHoverTarget"] {
    color: #64ffda !important;
}

/* ── STATUS BADGES ───────────────────────────────────────────────────────── */
.badge-warm {
    background: rgba(100,255,218,0.15); border: 1px solid #64ffda;
    color: #64ffda; border-radius: 20px; padding: 3px 12px; font-size: 0.8rem;
    font-family: 'Exo 2', sans-serif; display: inline-block;
}
.badge-cold {
    background: rgba(255,107,107,0.15); border: 1px solid #ff6b6b;
    color: #ff6b6b; border-radius: 20px; padding: 3px 12px; font-size: 0.8rem;
    font-family: 'Exo 2', sans-serif; display: inline-block;
}
.badge-new {
    background: rgba(255,184,77,0.15); border: 1px solid #ffb84d;
    color: #ffb84d; border-radius: 20px; padding: 3px 12px; font-size: 0.8rem;
    font-family: 'Exo 2', sans-serif; display: inline-block;
}
.badge-option-a {
    background: rgba(0,194,255,0.1); border: 1px solid rgba(0,194,255,0.4);
    color: #00c2ff; border-radius: 20px; padding: 2px 8px; font-size: 0.72rem;
    font-family: 'Exo 2', sans-serif; display: inline-block;
}
.badge-option-b {
    background: rgba(136,146,176,0.15); border: 1px solid rgba(180,196,220,0.4);
    color: #ccd6f6; border-radius: 20px; padding: 2px 8px; font-size: 0.85rem;
    font-family: 'Exo 2', sans-serif; display: inline-block;
}

/* ── SCROLLBAR ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #050a1a; }
::-webkit-scrollbar-thumb { background: rgba(0,194,255,0.4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,194,255,0.7); }

/* ── TABS ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(10,20,60,0.5) !important;
    border-radius: 8px; border-bottom: 1px solid rgba(0,194,255,0.2);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Exo 2', sans-serif !important;
    color: #b0c8e8 !important; font-weight: 700 !important;
    font-size: 1rem !important;
}
.stTabs [aria-selected="true"] {
    color: #00c2ff !important;
    border-bottom: 2px solid #00c2ff !important;
    background: transparent !important;
}

/* ── EXPANDER ────────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(10,20,60,0.6) !important;
    border: 1px solid rgba(0,194,255,0.2) !important;
    border-radius: 8px !important;
    font-family: 'Exo 2', sans-serif !important;
}

/* ── PROGRESS BAR ────────────────────────────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, #00c2ff, #64ffda) !important;
    border-radius: 4px !important;
}

/* ── DATAFRAMES ──────────────────────────────────────────────────────────── */
.dataframe {
    background: rgba(5,10,30,0.9) !important;
    border: 1px solid rgba(0,194,255,0.2) !important;
    border-radius: 8px !important;
    font-family: 'Exo 2', sans-serif !important;
    color: #ccd6f6 !important;
}
/* Contenedor externo del componente stDataFrame */
[data-testid="stDataFrame"] > div {
    background: rgba(8,15,45,0.85) !important;
    border: 1px solid rgba(0,194,255,0.2) !important;
    border-radius: 8px !important;
}

/* ── PLOTLY CHARTS — min-width para evitar translate(NaN) con width=0 ─────── */
[data-testid="stPlotlyChart"] {
    min-width: 100px !important;
    min-height: 100px !important;
}
[data-testid="stPlotlyChart"] > div {
    min-width: 100px !important;
}

/* ── MULTISELECT DARK THEME ──────────────────────────────────────────────── */
[data-testid="stMultiSelect"] > div > div[data-baseweb="select"] > div {
    background-color: rgba(10,20,60,0.85) !important;
    border: 1px solid rgba(0,194,255,0.35) !important;
    border-radius: 8px !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background-color: rgba(0,194,255,0.18) !important;
    border: 1px solid rgba(0,194,255,0.4) !important;
    border-radius: 4px !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span {
    color: #00c2ff !important;
}
/* Dropdown popup */
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] ul,
[data-baseweb="popover"] > div,
[data-baseweb="popover"] > div > div,
div[data-baseweb="menu"] {
    background: rgba(5,12,40,0.99) !important;
    border: 1px solid rgba(0,194,255,0.3) !important;
    border-radius: 8px !important;
}
[data-baseweb="menu"] li,
[data-baseweb="option"] {
    color: #e8eaf6 !important;
    background: transparent !important;
    font-size: 0.95rem !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="option"]:hover,
[data-baseweb="option"][aria-selected="true"] {
    background: rgba(0,194,255,0.18) !important;
    color: #00c2ff !important;
}
/* Texto dentro del multiselect input */
[data-testid="stMultiSelect"] input {
    color: #e8eaf6 !important;
    background: transparent !important;
}

/* ── SELECTBOX (dropdown) DARK THEME ────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div[data-baseweb="select"] > div {
    background-color: rgba(10,20,60,0.85) !important;
    border: 1px solid rgba(0,194,255,0.35) !important;
    border-radius: 8px !important;
    color: #ccd6f6 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] span {
    color: #ccd6f6 !important;
}

/* ── NUMBER INPUT DARK THEME ─────────────────────────────────────────────── */
[data-testid="stNumberInput"] input {
    background: rgba(10,20,60,0.85) !important;
    border: 1px solid rgba(0,194,255,0.3) !important;
    color: #ccd6f6 !important;
}
</style>

<!-- Elementos HTML para efectos espaciales -->
<div class="shooting-star-1"></div>
<div class="shooting-star-2"></div>
<div class="shooting-star-3"></div>
<div class="nebula-bg"></div>
<div class="nebula-bg-2"></div>
"""
