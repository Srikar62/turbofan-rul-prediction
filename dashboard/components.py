"""
Reusable Plotly figure builders for the Turbofan RUL Dashboard.
Each function returns a plotly.graph_objects.Figure or a Dash component.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import html, dash_table

from .theme import (
    COLORS, FD_COLORS, EXPERT_COLORS,
    KPI_CARD_STYLE, KPI_VALUE_STYLE, KPI_LABEL_STYLE, KPI_SUB_STYLE,
    TABLE_HEADER_STYLE, TABLE_CELL_STYLE, CHART_CARD_STYLE,
    SECTION_TITLE_STYLE,
)


# ═══════════════════════════════════════════════════════════════
#  KPI Cards
# ═══════════════════════════════════════════════════════════════

def create_kpi_card(label, value, subtitle="", color=COLORS["primary"], icon=""):
    """Build a glassmorphic KPI metric card."""
    glow = color.replace(")", ",0.10)").replace("rgb", "rgba") if "rgb" in color else color
    style = {
        **KPI_CARD_STYLE,
        "borderTop": f"3px solid {color}",
        "boxShadow": f"0 4px 24px {glow}" if "rgba" not in color else f"0 4px 24px rgba(0,0,0,0.3)",
    }
    return html.Div([
        html.P(label, style=KPI_LABEL_STYLE),
        html.H2(f"{icon} {value}" if icon else str(value),
                style={**KPI_VALUE_STYLE, "color": color}),
        html.P(subtitle, style=KPI_SUB_STYLE),
    ], style=style)


# ═══════════════════════════════════════════════════════════════
#  Overview Tab
# ═══════════════════════════════════════════════════════════════

def create_metrics_comparison_bars(results):
    """Grouped bar chart comparing RMSE and Score across all FD datasets."""
    fd_ids = sorted(results['datasets'].keys(), key=int)
    labels = [f"FD00{fd}" for fd in fd_ids]
    rmse_vals = [results['datasets'][fd]['metrics']['rmse'] for fd in fd_ids]
    score_vals = [results['datasets'][fd]['metrics']['score'] for fd in fd_ids]
    mae_vals = [results['datasets'][fd]['metrics']['mae'] for fd in fd_ids]
    colors = [FD_COLORS[int(fd)] for fd in fd_ids]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("RMSE (cycles)", "Score", "MAE (cycles)"),
        horizontal_spacing=0.08,
    )

    for i, (vals, name) in enumerate([
        (rmse_vals, "RMSE"), (score_vals, "Score"), (mae_vals, "MAE")
    ]):
        fig.add_trace(go.Bar(
            x=labels, y=vals, name=name,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.2f}" for v in vals],
            textposition='outside',
            textfont=dict(size=11, color=COLORS['text_muted']),
            showlegend=False,
        ), row=1, col=i + 1)

    fig.update_layout(
        height=350,
        margin=dict(t=50, b=30),
        bargap=0.35,
    )
    fig.update_yaxes(showgrid=True, gridcolor=COLORS['border'])
    return fig


def hex_to_rgba(hex_code, alpha=0.15):
    """Convert hex color string like #3b82f6 to rgba(59,130,246,0.15)."""
    h = hex_code.lstrip('#')
    if len(h) == 6:
        r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"
    return hex_code


def create_radar_chart(results):
    """Spider/radar chart comparing all metrics across FD datasets."""
    fd_ids = sorted(results['datasets'].keys(), key=int)
    categories = ['RMSE', 'MAE', 'Score', 'Uncertainty']

    fig = go.Figure()
    for fd in fd_ids:
        m = results['datasets'][fd]['metrics']
        # Normalize values for radar (lower is better for all except we invert)
        vals = [m['rmse'], m['mae'], m['score'] / 100, m['mean_uncertainty']]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],  # close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=hex_to_rgba(FD_COLORS[int(fd)], 0.15),
            line=dict(color=FD_COLORS[int(fd)], width=2),
            name=f"FD00{fd}",
            opacity=0.85,
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=COLORS['surface'],
            radialaxis=dict(
                visible=True, gridcolor=COLORS['border'],
                linecolor=COLORS['border'],
                tickfont=dict(color=COLORS['text_muted'], size=10),
            ),
            angularaxis=dict(
                gridcolor=COLORS['border'], linecolor=COLORS['border'],
                tickfont=dict(color=COLORS['text_muted'], size=11),
            ),
        ),
        height=400,
        margin=dict(t=30, b=30, l=60, r=60),
        legend=dict(x=1.05, y=1),
    )
    return fig


def create_summary_table(results):
    """Styled DataTable summarising metrics for all datasets."""
    fd_ids = sorted(results['datasets'].keys(), key=int)
    rows = []
    for fd in fd_ids:
        m = results['datasets'][fd]['metrics']
        info = results.get('dataset_info', {}).get(fd, {})
        rows.append({
            'Dataset': f'FD00{fd}',
            'RMSE': f"{m['rmse']:.2f}",
            'MAE': f"{m['mae']:.2f}",
            'Score': f"{m['score']:.1f}",
            'Uncertainty': f"{m['mean_uncertainty']:.1f} cyc",
            'Engines': f"{info.get('train_engines', '?')}/{info.get('test_engines', '?')}",
            'Sensors': info.get('sensor_count', '?'),
            'Window': info.get('window_size', '?'),
        })

    columns = [{'name': c, 'id': c} for c in rows[0].keys()]

    return dash_table.DataTable(
        data=rows,
        columns=columns,
        style_header=TABLE_HEADER_STYLE,
        style_cell=TABLE_CELL_STYLE,
        style_data_conditional=[
            {'if': {'row_index': 'odd'},
             'backgroundColor': COLORS['surface_alt']},
        ],
        style_table={'overflowX': 'auto', 'borderRadius': '8px'},
        style_as_list_view=True,
    )


def create_dataset_info_cards(results):
    """Small info cards showing dataset properties."""
    fd_ids = sorted(results.get('dataset_info', {}).keys(), key=int)
    cards = []
    for fd in fd_ids:
        info = results['dataset_info'][fd]
        card = html.Div([
            html.H4(f"FD00{fd}", style={
                "color": FD_COLORS[int(fd)], "margin": "0 0 8px 0",
                "fontSize": "15px", "fontWeight": "700"}),
            html.P(f"Train: {info['train_engines']} engines · {info['train_samples']:,} windows",
                   style={"margin": "2px 0", "fontSize": "12px", "color": COLORS['text_muted']}),
            html.P(f"Test: {info['test_engines']} engines · {info['test_samples']} windows",
                   style={"margin": "2px 0", "fontSize": "12px", "color": COLORS['text_muted']}),
            html.P(f"Sensors: {info['sensor_count']} · Window: {info['window_size']} · Cap: {info['rul_max']}",
                   style={"margin": "2px 0", "fontSize": "12px", "color": COLORS['text_muted']}),
            html.P(f"Conditions: {info['n_conditions']}",
                   style={"margin": "2px 0", "fontSize": "12px", "color": COLORS['text_muted']}),
        ], style={
            **KPI_CARD_STYLE,
            "borderLeft": f"3px solid {FD_COLORS[int(fd)]}",
            "borderTop": "none",
            "textAlign": "left",
            "padding": "16px 20px",
        })
        cards.append(card)
    return cards


# ═══════════════════════════════════════════════════════════════
#  Per-Dataset Tab
# ═══════════════════════════════════════════════════════════════

def create_prediction_plot(results, fd_id):
    """Scatter plot: Predicted vs True RUL with 90% CI band."""
    fd = str(fd_id)
    preds = results['datasets'][fd]['predictions']
    pred = np.array(preds['predicted'])
    true = np.array(preds['true'])
    ci_lo = np.array(preds['ci_lo'])
    ci_hi = np.array(preds['ci_hi'])

    n = min(400, len(true))
    x = np.arange(n)
    color = FD_COLORS[fd_id]

    fig = go.Figure()

    # CI band
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([ci_hi[:n], ci_lo[:n][::-1]]),
        fill='toself', fillcolor=COLORS['ci_fill'],
        line=dict(width=0), name='90% CI', hoverinfo='skip',
    ))

    # True RUL
    fig.add_trace(go.Scatter(
        x=x, y=true[:n], mode='lines',
        line=dict(color=COLORS['text'], width=1.5),
        name='True RUL',
    ))

    # Predicted RUL
    fig.add_trace(go.Scatter(
        x=x, y=pred[:n], mode='lines',
        line=dict(color=color, width=1.5),
        name='Predicted RUL',
    ))

    m = results['datasets'][fd]['metrics']
    fig.update_layout(
        title=f"FD00{fd_id} — Predictions vs Truth  |  RMSE={m['rmse']:.2f}  Score={m['score']:.0f}",
        xaxis_title="Sample", yaxis_title="RUL (cycles)",
        height=400, yaxis=dict(rangemode='tozero'),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
    )
    return fig


def create_residual_plots(results, fd_id):
    """Error distribution histogram + residuals vs true RUL."""
    fd = str(fd_id)
    preds = results['datasets'][fd]['predictions']
    pred = np.array(preds['predicted'])
    true = np.array(preds['true'])
    errors = pred - true
    color = FD_COLORS[fd_id]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Error Distribution", "Residual vs True RUL"),
        horizontal_spacing=0.1,
    )

    # Histogram
    fig.add_trace(go.Histogram(
        x=errors, nbinsx=40,
        marker=dict(color=color, line=dict(color=COLORS['surface'], width=1)),
        opacity=0.85, name='Error',
    ), row=1, col=1)
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS['danger'],
                  annotation_text="0", row=1, col=1)

    # Residual scatter
    fig.add_trace(go.Scatter(
        x=true, y=errors, mode='markers',
        marker=dict(color=color, size=4, opacity=0.5),
        name='Residuals',
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS['danger'], row=1, col=2)

    fig.update_layout(
        height=350, showlegend=False,
        margin=dict(t=50, b=30),
    )
    fig.update_xaxes(title_text="Error (cycles)", row=1, col=1)
    fig.update_xaxes(title_text="True RUL (cycles)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Error (cycles)", row=1, col=2)
    return fig


def create_uncertainty_plot(results, fd_id):
    """Predicted RUL with uncertainty error bars, colored by confidence."""
    fd = str(fd_id)
    preds = results['datasets'][fd]['predictions']
    pred = np.array(preds['predicted'])
    std = np.array(preds['std'])
    true = np.array(preds['true'])

    n = len(pred)
    x = np.arange(n)

    # Color by std — low std = green, high std = red
    std_norm = (std - std.min()) / (std.max() - std.min() + 1e-8)
    marker_colors = [
        f"rgb({int(50 + 200*s)}, {int(180 - 140*s)}, {int(80 + 40*(1-s))})"
        for s in std_norm
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=true, mode='lines',
        line=dict(color=COLORS['text_muted'], width=1, dash='dot'),
        name='True RUL',
    ))

    fig.add_trace(go.Scatter(
        x=x, y=pred, mode='markers',
        error_y=dict(type='data', array=(std * 1.645).tolist(), visible=True,
                     color='rgba(255,255,255,0.15)', thickness=1),
        marker=dict(color=marker_colors, size=5),
        name='Predicted ± 90% CI',
    ))

    fig.update_layout(
        title=f"FD00{fd_id} — Uncertainty Analysis  |  Mean σ = {std.mean():.1f} cycles",
        xaxis_title="Engine (test sample)", yaxis_title="RUL (cycles)",
        height=380, yaxis=dict(rangemode='tozero'),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
    )
    return fig


def create_pred_vs_true_scatter(results, fd_id):
    """45-degree scatter: predicted vs true with perfect-line reference."""
    fd = str(fd_id)
    preds = results['datasets'][fd]['predictions']
    pred = np.array(preds['predicted'])
    true = np.array(preds['true'])
    color = FD_COLORS[fd_id]

    fig = go.Figure()

    max_val = max(pred.max(), true.max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val], mode='lines',
        line=dict(color=COLORS['text_muted'], width=1, dash='dash'),
        name='Perfect', showlegend=True,
    ))

    fig.add_trace(go.Scatter(
        x=true, y=pred, mode='markers',
        marker=dict(color=color, size=6, opacity=0.6,
                    line=dict(color=COLORS['surface'], width=0.5)),
        name=f'FD00{fd_id}',
    ))

    fig.update_layout(
        title=f"FD00{fd_id} — Predicted vs True RUL",
        xaxis_title="True RUL (cycles)", yaxis_title="Predicted RUL (cycles)",
        height=380,
        xaxis=dict(range=[0, max_val]),
        yaxis=dict(range=[0, max_val]),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  Expert Utilisation Tab
# ═══════════════════════════════════════════════════════════════

def create_expert_heatmap(results):
    """Heatmap of MoE expert gate weights across datasets."""
    expert_data = results.get('expert_gates', {})
    fd_ids = sorted(expert_data.keys(), key=int)
    n_experts = len(expert_data.get(fd_ids[0], [])) if fd_ids else 4

    z = [expert_data[fd] for fd in fd_ids]
    x_labels = [f"Expert {i+1}" for i in range(n_experts)]
    y_labels = [f"FD00{fd}" for fd in fd_ids]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x_labels, y=y_labels,
        colorscale=[
            [0, COLORS['surface_alt']],
            [0.5, COLORS['primary']],
            [1, COLORS['accent']],
        ],
        text=[[f"{v:.3f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=13, color=COLORS['text']),
        hovertemplate="Dataset: %{y}<br>Expert: %{x}<br>Weight: %{z:.4f}<extra></extra>",
        colorbar=dict(title="Weight", tickfont=dict(color=COLORS['text_muted'])),
    ))

    fig.update_layout(
        title="MoE Expert Gate Weights",
        height=300,
        margin=dict(t=50, b=30, l=80),
        xaxis=dict(side='top'),
    )
    return fig


def create_expert_bars(results):
    """Grouped bar chart of expert utilisation per dataset."""
    expert_data = results.get('expert_gates', {})
    fd_ids = sorted(expert_data.keys(), key=int)
    n_experts = len(expert_data.get(fd_ids[0], [])) if fd_ids else 4

    fig = go.Figure()
    for i in range(n_experts):
        vals = [expert_data[fd][i] for fd in fd_ids]
        fig.add_trace(go.Bar(
            x=[f"FD00{fd}" for fd in fd_ids],
            y=vals, name=f"Expert {i+1}",
            marker_color=EXPERT_COLORS[i % len(EXPERT_COLORS)],
            text=[f"{v:.3f}" for v in vals],
            textposition='outside',
            textfont=dict(size=10, color=COLORS['text_muted']),
        ))

    fig.update_layout(
        title="Expert Utilisation by Dataset",
        barmode='group', bargap=0.2, bargroupgap=0.05,
        height=380,
        yaxis_title="Gate Weight",
        legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
    )
    return fig


def create_expert_entropy(results):
    """Entropy of expert gate weights — higher = more balanced."""
    expert_data = results.get('expert_gates', {})
    fd_ids = sorted(expert_data.keys(), key=int)

    entropies = []
    n_experts = 4
    max_entropy = np.log(n_experts)

    for fd in fd_ids:
        gates = np.array(expert_data[fd]).clip(1e-8)
        entropy = -(gates * np.log(gates)).sum()
        entropies.append(entropy / max_entropy * 100)  # as percentage of max

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"FD00{fd}" for fd in fd_ids],
        y=entropies,
        marker=dict(
            color=entropies,
            colorscale=[[0, COLORS['danger']], [0.5, COLORS['accent']], [1, COLORS['success']]],
            cmin=0, cmax=100,
            line=dict(width=0),
        ),
        text=[f"{e:.1f}%" for e in entropies],
        textposition='outside',
        textfont=dict(size=12, color=COLORS['text']),
    ))

    fig.add_hline(y=100, line_dash="dash", line_color=COLORS['success'],
                  annotation_text="Perfect Balance (100%)",
                  annotation_font=dict(color=COLORS['success'], size=11))

    fig.update_layout(
        title="Expert Balance (Entropy as % of Maximum)",
        yaxis_title="Balance %", yaxis=dict(range=[0, 115]),
        height=350, showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  EDA Tab
# ═══════════════════════════════════════════════════════════════

def create_eda_image_section(fd_id):
    """Return Dash components displaying the pre-generated EDA images."""
    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eda_path = os.path.join(base, f'eda_fd00{fd_id}.png')
    deg_path = os.path.join(base, f'eda_degradation_fd00{fd_id}.png')

    elements = []
    if os.path.exists(eda_path):
        elements.append(html.Div([
            html.H4(f"FD00{fd_id} — EDA Overview", style=SECTION_TITLE_STYLE),
            html.Img(src=f'/assets/eda_fd00{fd_id}.png',
                     style={'width': '100%', 'borderRadius': '8px',
                            'border': f"1px solid {COLORS['border']}"}),
        ], style={**CHART_CARD_STYLE, 'marginBottom': '16px'}))

    if os.path.exists(deg_path):
        elements.append(html.Div([
            html.H4(f"FD00{fd_id} — Sensor Degradation Trajectories",
                     style=SECTION_TITLE_STYLE),
            html.Img(src=f'/assets/eda_degradation_fd00{fd_id}.png',
                     style={'width': '100%', 'borderRadius': '8px',
                            'border': f"1px solid {COLORS['border']}"}),
        ], style=CHART_CARD_STYLE))

    if not elements:
        elements.append(html.P(
            f"No EDA images found for FD00{fd_id}. Run EDA.py first.",
            style={'color': COLORS['text_muted'], 'textAlign': 'center', 'padding': '40px'},
        ))

    return elements
