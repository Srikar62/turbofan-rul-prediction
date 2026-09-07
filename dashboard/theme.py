"""
Centralized dark-theme styling for the Turbofan RUL Dashboard.
Aerospace-inspired palette with glassmorphic card styles.
"""
import plotly.graph_objects as go
import plotly.io as pio

# ── Colour Palette ──────────────────────────────────────────────
COLORS = {
    "bg":           "#0b0e14",
    "surface":      "#141820",
    "surface_alt":  "#1a1f2e",
    "card":         "#161b26",
    "border":       "#252d3d",
    "text":         "#e2e8f0",
    "text_muted":   "#8892a4",
    "primary":      "#3b82f6",
    "primary_glow": "rgba(59,130,246,0.15)",
    "accent":       "#f59e0b",
    "accent_glow":  "rgba(245,158,11,0.15)",
    "success":      "#10b981",
    "danger":       "#ef4444",
    "info":         "#06b6d4",
    "chart_blue":   "#3b82f6",
    "chart_orange": "#f59e0b",
    "chart_green":  "#10b981",
    "chart_red":    "#ef4444",
    "chart_cyan":   "#06b6d4",
    "chart_purple": "#8b5cf6",
    "ci_fill":      "rgba(59,130,246,0.12)",
}

FD_COLORS = {
    1: "#3b82f6",
    2: "#f59e0b",
    3: "#10b981",
    4: "#8b5cf6",
}

EXPERT_COLORS = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444"]

# ── Plotly Template ─────────────────────────────────────────────
PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(family="Inter, Segoe UI, sans-serif", color=COLORS["text"], size=12),
        title=dict(font=dict(size=16, color=COLORS["text"])),
        xaxis=dict(
            gridcolor=COLORS["border"],
            zerolinecolor=COLORS["border"],
            linecolor=COLORS["border"],
            tickfont=dict(color=COLORS["text_muted"]),
        ),
        yaxis=dict(
            gridcolor=COLORS["border"],
            zerolinecolor=COLORS["border"],
            linecolor=COLORS["border"],
            tickfont=dict(color=COLORS["text_muted"]),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=COLORS["border"],
            font=dict(color=COLORS["text_muted"], size=11),
        ),
        colorway=[
            COLORS["chart_blue"], COLORS["chart_orange"],
            COLORS["chart_green"], COLORS["chart_red"],
            COLORS["chart_cyan"], COLORS["chart_purple"],
        ],
        margin=dict(l=50, r=30, t=50, b=40),
    )
)
pio.templates["turbofan_dark"] = PLOTLY_TEMPLATE
pio.templates.default = "turbofan_dark"

# ── CSS Styles ──────────────────────────────────────────────────
BODY_STYLE = {
    "backgroundColor": COLORS["bg"],
    "color": COLORS["text"],
    "fontFamily": "'Inter', 'Segoe UI', sans-serif",
    "minHeight": "100vh",
    "padding": "0",
    "margin": "0",
}

HEADER_STYLE = {
    "background": f"linear-gradient(135deg, {COLORS['surface']} 0%, {COLORS['surface_alt']} 100%)",
    "borderBottom": f"1px solid {COLORS['border']}",
    "padding": "20px 32px",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "space-between",
}

HEADER_TITLE_STYLE = {
    "fontSize": "22px",
    "fontWeight": "700",
    "color": COLORS["text"],
    "margin": "0",
    "letterSpacing": "-0.3px",
}

HEADER_SUBTITLE_STYLE = {
    "fontSize": "13px",
    "color": COLORS["text_muted"],
    "margin": "4px 0 0 0",
    "fontWeight": "400",
}

CONTENT_STYLE = {
    "padding": "24px 32px",
    "maxWidth": "1440px",
    "margin": "0 auto",
}

KPI_CARD_STYLE = {
    "background": f"linear-gradient(145deg, {COLORS['card']} 0%, {COLORS['surface_alt']} 100%)",
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "12px",
    "padding": "20px 24px",
    "textAlign": "center",
    "flex": "1",
    "minWidth": "180px",
    "position": "relative",
    "overflow": "hidden",
}

KPI_VALUE_STYLE = {
    "fontSize": "28px",
    "fontWeight": "700",
    "margin": "8px 0 4px 0",
    "letterSpacing": "-0.5px",
}

KPI_LABEL_STYLE = {
    "fontSize": "12px",
    "fontWeight": "600",
    "textTransform": "uppercase",
    "letterSpacing": "0.8px",
    "color": COLORS["text_muted"],
    "margin": "0",
}

KPI_SUB_STYLE = {
    "fontSize": "12px",
    "color": COLORS["text_muted"],
    "margin": "0",
}

KPI_ROW_STYLE = {
    "display": "flex",
    "gap": "16px",
    "marginBottom": "24px",
    "flexWrap": "wrap",
}

CHART_CARD_STYLE = {
    "background": COLORS["surface"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "12px",
    "padding": "20px",
    "marginBottom": "20px",
}

CHART_ROW_STYLE = {
    "display": "flex",
    "gap": "20px",
    "marginBottom": "20px",
    "flexWrap": "wrap",
}

CHART_HALF_STYLE = {
    "flex": "1",
    "minWidth": "400px",
}

TAB_STYLE = {
    "backgroundColor": COLORS["surface"],
    "color": COLORS["text_muted"],
    "border": "none",
    "borderBottom": f"2px solid transparent",
    "padding": "12px 20px",
    "fontWeight": "500",
    "fontSize": "13px",
    "letterSpacing": "0.3px",
}

TAB_SELECTED_STYLE = {
    "backgroundColor": COLORS["surface"],
    "color": COLORS["primary"],
    "border": "none",
    "borderBottom": f"2px solid {COLORS['primary']}",
    "padding": "12px 20px",
    "fontWeight": "600",
    "fontSize": "13px",
    "letterSpacing": "0.3px",
}

TABLE_STYLE = {
    "backgroundColor": COLORS["surface"],
    "color": COLORS["text"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "8px",
    "overflow": "hidden",
}

TABLE_HEADER_STYLE = {
    "backgroundColor": COLORS["surface_alt"],
    "color": COLORS["text"],
    "fontWeight": "600",
    "fontSize": "12px",
    "textTransform": "uppercase",
    "letterSpacing": "0.5px",
    "border": f"1px solid {COLORS['border']}",
    "padding": "12px 16px",
}

TABLE_CELL_STYLE = {
    "backgroundColor": COLORS["surface"],
    "color": COLORS["text"],
    "border": f"1px solid {COLORS['border']}",
    "padding": "10px 16px",
    "fontSize": "13px",
}

REFRESH_BTN_STYLE = {
    "backgroundColor": COLORS["primary"],
    "color": "#fff",
    "border": "none",
    "borderRadius": "8px",
    "padding": "10px 20px",
    "fontWeight": "600",
    "fontSize": "13px",
    "cursor": "pointer",
    "letterSpacing": "0.3px",
}

DROPDOWN_STYLE = {
    "backgroundColor": COLORS["surface_alt"],
    "color": COLORS["text"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "8px",
}

SECTION_TITLE_STYLE = {
    "fontSize": "16px",
    "fontWeight": "600",
    "color": COLORS["text"],
    "margin": "0 0 16px 0",
    "letterSpacing": "-0.2px",
}
