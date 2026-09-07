"""
Main Dash application for Turbofan RUL Prediction Dashboard.
Run:  python -m dashboard.app
"""
import os
import sys
import shutil

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly.graph_objects as go

from .theme import (
    COLORS, FD_COLORS,
    BODY_STYLE, HEADER_STYLE, HEADER_TITLE_STYLE, HEADER_SUBTITLE_STYLE,
    CONTENT_STYLE, KPI_ROW_STYLE, CHART_CARD_STYLE, CHART_ROW_STYLE,
    CHART_HALF_STYLE, TAB_STYLE, TAB_SELECTED_STYLE, REFRESH_BTN_STYLE,
    SECTION_TITLE_STYLE, DROPDOWN_STYLE,
)
from .components import (
    create_kpi_card,
    create_metrics_comparison_bars,
    create_radar_chart,
    create_summary_table,
    create_dataset_info_cards,
    create_prediction_plot,
    create_residual_plots,
    create_uncertainty_plot,
    create_pred_vs_true_scatter,
    create_expert_heatmap,
    create_expert_bars,
    create_expert_entropy,
    create_eda_image_section,
)
from .data_loader import load_results, refresh_results


# ── Copy EDA images to Dash assets folder ───────────────────────
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

for fd_id in range(1, 5):
    for pattern in [f'eda_fd00{fd_id}.png', f'eda_degradation_fd00{fd_id}.png']:
        src = os.path.join(PROJECT_ROOT, pattern)
        dst = os.path.join(ASSETS_DIR, pattern)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

# Also copy the results image if available
for img in ['rul_results.png']:
    src = os.path.join(PROJECT_ROOT, img)
    dst = os.path.join(ASSETS_DIR, img)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)

# ── Load initial data ──────────────────────────────────────────
print('[Dashboard] Initializing...')
RESULTS = load_results()


# ── Build Dash App ─────────────────────────────────────────────
app = Dash(
    __name__,
    title='Turbofan RUL Dashboard',
    update_title='Loading...',
    assets_folder=ASSETS_DIR,
    suppress_callback_exceptions=True,
)
server = app.server  # For deployment (gunicorn / waitress)


def _avg_metric(key):
    fd_ids = list(RESULTS['datasets'].keys())
    return sum(RESULTS['datasets'][fd]['metrics'][key] for fd in fd_ids) / len(fd_ids)


# ═══════════════════════════════════════════════════════════════
#  Layout
# ═══════════════════════════════════════════════════════════════

def build_header():
    return html.Div([
        html.Div([
            html.H1("🛩️ Turbofan RUL Prediction Dashboard",
                     style=HEADER_TITLE_STYLE),
            html.P("CNN-BiLSTM-3DAttn + MoE(4) + MC Dropout  •  C-MAPSS Benchmark",
                    style=HEADER_SUBTITLE_STYLE),
        ]),
    ], style=HEADER_STYLE)


def build_kpi_row():
    return html.Div([
        create_kpi_card("Avg RMSE", f"{_avg_metric('rmse'):.2f}",
                        "cycles", COLORS['primary'], "📉"),
        create_kpi_card("Avg Score", f"{_avg_metric('score'):.0f}",
                        "C-MAPSS asymmetric", COLORS['accent'], "🎯"),
        create_kpi_card("Avg MAE", f"{_avg_metric('mae'):.2f}",
                        "cycles", COLORS['success'], "📏"),
        create_kpi_card("Avg Uncertainty", f"{_avg_metric('mean_uncertainty'):.1f}",
                        "σ cycles", COLORS['info'], "🔮"),
    ], style=KPI_ROW_STYLE)


def build_overview_tab():
    return html.Div([
        html.H3("Performance Summary", style=SECTION_TITLE_STYLE),

        html.Div(
            create_summary_table(RESULTS),
            style={**CHART_CARD_STYLE, 'padding': '16px'},
        ),

        html.Div([
            html.H3("Metrics Comparison", style=SECTION_TITLE_STYLE),
            dcc.Graph(figure=create_metrics_comparison_bars(RESULTS),
                      config={'displayModeBar': False}),
        ], style=CHART_CARD_STYLE),

        html.Div(style=CHART_ROW_STYLE, children=[
            html.Div([
                html.H3("Multi-Metric Radar", style=SECTION_TITLE_STYLE),
                dcc.Graph(figure=create_radar_chart(RESULTS),
                          config={'displayModeBar': False}),
            ], style={**CHART_CARD_STYLE, **CHART_HALF_STYLE}),

            html.Div([
                html.H3("Dataset Properties", style=SECTION_TITLE_STYLE),
                html.Div(create_dataset_info_cards(RESULTS),
                         style={'display': 'flex', 'flexDirection': 'column',
                                'gap': '12px'}),
            ], style={**CHART_CARD_STYLE, **CHART_HALF_STYLE}),
        ]),
    ])


def build_fd_tab(fd_id):
    m = RESULTS['datasets'][str(fd_id)]['metrics']
    info = RESULTS.get('dataset_info', {}).get(str(fd_id), {})
    hp = RESULTS['datasets'][str(fd_id)].get('hyperparameters', {})

    return html.Div([
        # Per-dataset KPIs
        html.Div([
            create_kpi_card("RMSE", f"{m['rmse']:.2f}", "cycles",
                            FD_COLORS[fd_id]),
            create_kpi_card("Score", f"{m['score']:.0f}", "C-MAPSS",
                            FD_COLORS[fd_id]),
            create_kpi_card("MAE", f"{m['mae']:.2f}", "cycles",
                            FD_COLORS[fd_id]),
            create_kpi_card("Uncertainty", f"{m['mean_uncertainty']:.1f}",
                            "σ cycles", FD_COLORS[fd_id]),
            create_kpi_card("Config",
                            f"W={info.get('window_size','?')} E={hp.get('epochs','?')}",
                            f"lr={hp.get('lr','?')} dr={hp.get('dropout','?')}",
                            COLORS['text_muted']),
        ], style=KPI_ROW_STYLE),

        # Prediction plot
        html.Div([
            dcc.Graph(figure=create_prediction_plot(RESULTS, fd_id),
                      config={'displayModeBar': True,
                              'modeBarButtonsToRemove': ['lasso2d', 'select2d']}),
        ], style=CHART_CARD_STYLE),

        # Side-by-side: scatter + uncertainty
        html.Div(style=CHART_ROW_STYLE, children=[
            html.Div([
                dcc.Graph(figure=create_pred_vs_true_scatter(RESULTS, fd_id),
                          config={'displayModeBar': False}),
            ], style={**CHART_CARD_STYLE, **CHART_HALF_STYLE}),
            html.Div([
                dcc.Graph(figure=create_uncertainty_plot(RESULTS, fd_id),
                          config={'displayModeBar': False}),
            ], style={**CHART_CARD_STYLE, **CHART_HALF_STYLE}),
        ]),

        # Residuals
        html.Div([
            dcc.Graph(figure=create_residual_plots(RESULTS, fd_id),
                      config={'displayModeBar': False}),
        ], style=CHART_CARD_STYLE),
    ])


def build_expert_tab():
    return html.Div([
        html.H3("Mixture of Experts — Gate Utilisation", style=SECTION_TITLE_STYLE),

        html.Div([
            dcc.Graph(figure=create_expert_heatmap(RESULTS),
                      config={'displayModeBar': False}),
        ], style=CHART_CARD_STYLE),

        html.Div(style=CHART_ROW_STYLE, children=[
            html.Div([
                dcc.Graph(figure=create_expert_bars(RESULTS),
                          config={'displayModeBar': False}),
            ], style={**CHART_CARD_STYLE, **CHART_HALF_STYLE}),
            html.Div([
                dcc.Graph(figure=create_expert_entropy(RESULTS),
                          config={'displayModeBar': False}),
            ], style={**CHART_CARD_STYLE, **CHART_HALF_STYLE}),
        ]),

        html.Div([
            html.P([
                html.Strong("How to read: "),
                "Each expert in the MoE head contributes to the final RUL prediction "
                "weighted by a learned gate function. Balanced utilisation (all gates ≈ 0.25) "
                "means no expert collapse — the model leverages all 4 specialist sub-networks. "
                "The entropy bar shows balance as a % of perfect uniformity.",
            ], style={'color': COLORS['text_muted'], 'fontSize': '13px',
                      'lineHeight': '1.6', 'maxWidth': '800px'}),
        ], style={**CHART_CARD_STYLE, 'padding': '20px'}),
    ])


def build_eda_tab():
    return html.Div([
        html.H3("Exploratory Data Analysis", style=SECTION_TITLE_STYLE),

        html.Div([
            html.Label("Select Dataset:", style={
                'color': COLORS['text_muted'], 'fontSize': '13px',
                'marginRight': '12px', 'fontWeight': '500'}),
            dcc.Dropdown(
                id='eda-fd-selector',
                options=[{'label': f'FD00{i}', 'value': i} for i in range(1, 5)],
                value=1,
                clearable=False,
                style={**DROPDOWN_STYLE, 'width': '140px'},
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '20px'}),

        html.Div(id='eda-content'),
    ])


# ── Full Layout ────────────────────────────────────────────────
app.layout = html.Div([
    # Hidden store for refresh signal
    dcc.Store(id='results-store', data=0),
    html.Div(id='notification-area'),

    build_header(),

    html.Div([
        build_kpi_row(),

        dcc.Tabs(
            id='main-tabs',
            value='overview',
            children=[
                dcc.Tab(label='Overview', value='overview',
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label='FD001', value='fd1',
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label='FD002', value='fd2',
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label='FD003', value='fd3',
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label='FD004', value='fd4',
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label='Experts', value='experts',
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label='EDA', value='eda',
                        style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
            ],
            style={
                'borderBottom': f"1px solid {COLORS['border']}",
                'marginBottom': '24px',
            },
            colors={
                'border': COLORS['border'],
                'primary': COLORS['primary'],
                'background': COLORS['surface'],
            },
        ),

        html.Div(id='tab-content'),

    ], style=CONTENT_STYLE),

    # Footer
    html.Div([
        html.P("Turbofan RUL Prediction  •  CNN-BiLSTM-3DAttn + MoE(4)  •  C-MAPSS NASA Benchmark",
               style={'textAlign': 'center', 'color': COLORS['text_muted'],
                      'fontSize': '11px', 'padding': '16px 0',
                      'borderTop': f"1px solid {COLORS['border']}",
                      'margin': '40px 32px 0 32px'}),
    ]),
], style=BODY_STYLE)


# ═══════════════════════════════════════════════════════════════
#  Callbacks
# ═══════════════════════════════════════════════════════════════

@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value'),
)
def render_tab(tab):
    if tab == 'overview':
        return build_overview_tab()
    elif tab in ('fd1', 'fd2', 'fd3', 'fd4'):
        fd_id = int(tab[-1])
        return build_fd_tab(fd_id)
    elif tab == 'experts':
        return build_expert_tab()
    elif tab == 'eda':
        return build_eda_tab()
    return html.P("Select a tab.", style={'color': COLORS['text_muted']})


@app.callback(
    Output('eda-content', 'children'),
    Input('eda-fd-selector', 'value'),
)
def render_eda(fd_id):
    if fd_id is None:
        fd_id = 1
    return create_eda_image_section(fd_id)


@app.callback(
    Output('notification-area', 'children'),
    Input('refresh-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def handle_refresh(n_clicks):
    global RESULTS
    try:
        RESULTS = refresh_results()
        return html.Div(
            "✓ Data refreshed successfully. Reload tabs to see updates.",
            style={
                'backgroundColor': COLORS['success'],
                'color': '#fff', 'padding': '10px 24px',
                'textAlign': 'center', 'fontSize': '13px', 'fontWeight': '500',
            },
        )
    except Exception as e:
        return html.Div(
            f"✗ Refresh failed: {str(e)}",
            style={
                'backgroundColor': COLORS['danger'],
                'color': '#fff', 'padding': '10px 24px',
                'textAlign': 'center', 'fontSize': '13px', 'fontWeight': '500',
            },
        )


# ═══════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    print('\n' + '=' * 60)
    print('  [AERO] Turbofan RUL Prediction Dashboard')
    print('  http://127.0.0.1:8050')
    print('=' * 60 + '\n')
    app.run(debug=False, host='0.0.0.0', port=8050)
