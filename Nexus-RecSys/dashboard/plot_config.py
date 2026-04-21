"""
plot_config.py — Configuración Plotly para temática espacial
=============================================================
Importar SPACE_LAYOUT y aplicarlo a todos los go.Figure del dashboard.
"""

# Paleta de colores del sistema
COLORS = {
    "primary":    "#00c2ff",   # cyan neón
    "secondary":  "#64ffda",   # turquesa
    "accent":     "#ffb84d",   # naranja dorado
    "danger":     "#ff6b6b",   # rojo
    "purple":     "#a855f7",   # púrpura
    "blue":       "#2b5be0",   # azul
    "gray":       "#8892b0",   # gris azulado
    "white":      "#ccd6f6",   # blanco apagado
    "bg_dark":    "#050a1a",   # fondo oscuro
    "bg_card":    "rgba(10,20,60,0.85)",
}

# Secuencia de colores para gráficos con múltiples series
COLORWAY = [
    "#00c2ff",  # cyan
    "#64ffda",  # turquesa
    "#ffb84d",  # naranja
    "#ff6b6b",  # rojo
    "#2b5be0",  # azul
    "#a855f7",  # púrpura
    "#f9d71c",  # amarillo
    "#4ade80",  # verde
]

# Layout base para todos los gráficos
# NOTA: NO incluir title_font aquí — en Plotly 6.x genera title.text="undefined"
SPACE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,20,60,0.3)",
    font=dict(
        family="Exo 2, sans-serif",
        color="#ccd6f6",
        size=15,
    ),
    colorway=COLORWAY,
    xaxis=dict(
        gridcolor="rgba(0,194,255,0.1)",
        zerolinecolor="rgba(0,194,255,0.2)",
        linecolor="rgba(0,194,255,0.2)",
        tickfont=dict(color="#ccd6f6", size=13),
        title_font=dict(color="#ffffff", size=14),
    ),
    yaxis=dict(
        gridcolor="rgba(0,194,255,0.1)",
        zerolinecolor="rgba(0,194,255,0.2)",
        linecolor="rgba(0,194,255,0.2)",
        tickfont=dict(color="#ccd6f6", size=13),
        title_font=dict(color="#ffffff", size=14),
    ),
    hoverlabel=dict(
        bgcolor="rgba(10,20,60,0.95)",
        bordercolor="rgba(0,194,255,0.5)",
        font=dict(family="Exo 2", color="#ffffff", size=14),
    ),
    legend=dict(
        bgcolor="rgba(5,10,30,0.8)",
        bordercolor="rgba(0,194,255,0.2)",
        borderwidth=1,
        font=dict(color="#ccd6f6", size=13),
    ),
    margin=dict(t=50, b=40, l=50, r=20),
)


def apply_space_theme(fig, height: int = 400, title: str = "") -> None:
    """Aplica la temática espacial a una figura Plotly in-place."""
    layout_update = dict(**SPACE_LAYOUT, height=height, autosize=True)
    if title:
        layout_update["title"] = dict(
            text=title,
            font=dict(family="Orbitron, monospace", color="#ffffff", size=16),
            x=0.02,
        )
    else:
        # Asegurar que no haya título residual con text=undefined
        layout_update["title"] = dict(text="")
    fig.update_layout(**layout_update)


def bar_color_gradient(n: int) -> list:
    """Genera una lista de n colores del gradiente neón para barras."""
    return [COLORWAY[i % len(COLORWAY)] for i in range(n)]
