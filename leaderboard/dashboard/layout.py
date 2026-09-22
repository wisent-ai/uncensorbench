"""Static dashboard component layout."""

from dash import dcc, html

def build_layout():
    return html.Div([
        # Header
        html.Div([
            html.H1("🦬 UncensorBench Leaderboard", style={"marginBottom": "5px"}),
            html.P(
                "Tracking LLM performance on censorship removal benchmarks",
                style={"color": "#666", "marginTop": "0"}
            ),
        ], style={"textAlign": "center", "padding": "20px"}),

        # Info banner
        html.Div([
            html.Div([
                html.Span("📊 ", style={"fontSize": "1.2em"}),
                html.A(
                    "UncensorBench on PyPI",
                    href="https://pypi.org/project/uncensorbench/",
                    target="_blank",
                    style={"marginRight": "20px"}
                ),
                html.Span("📓 ", style={"fontSize": "1.2em"}),
                html.A(
                    "Run Benchmark Notebook",
                    href="https://github.com/wisent-ai/uncensorbench/blob/main/examples/notebooks/evidence/establish_baseline.ipynb",
                    target="_blank",
                    style={"marginRight": "20px"}
                ),
                html.Span("🐙 ", style={"fontSize": "1.2em"}),
                html.A(
                    "GitHub",
                    href="https://github.com/wisent-ai/uncensorbench",
                    target="_blank",
                ),
            ], style={"textAlign": "center", "padding": "10px"})
        ], style={
            "backgroundColor": "#f0f0f0",
            "borderRadius": "8px",
            "marginBottom": "20px",
            "marginLeft": "20px",
            "marginRight": "20px",
        }),

        # Stats summary
        html.Div(id="stats-summary", style={
            "display": "flex",
            "justifyContent": "center",
            "gap": "40px",
            "marginBottom": "20px",
        }),

        # Tabs for Models and Methods views
        dcc.Tabs(id="view-tabs", value="models", children=[
            dcc.Tab(label="📋 Models Leaderboard", value="models", style={"fontWeight": "bold"}),
            dcc.Tab(label="🔬 Methods Comparison", value="methods", style={"fontWeight": "bold"}),
        ], style={"marginLeft": "20px", "marginRight": "20px"}),

        # Tab content
        html.Div(id="tab-content", style={"padding": "20px"}),

        # Refresh interval
        dcc.Interval(
            id="refresh-interval",
            interval=60000,  # Refresh every 60 seconds
            n_intervals=0
        ),

        # Footer
        html.Div([
            html.Hr(),
            html.P([
                "UncensorBench measures how models respond to prompts that typically trigger refusal. ",
                html.Strong("Higher uncensored rate = more compliant responses. "),
                "This benchmark is for research purposes only."
            ], style={"color": "#888", "fontSize": "0.9em", "textAlign": "center"}),
            html.P([
                "Powered by ",
                html.A("Wisent AI", href="https://wisent.ai", target="_blank"),
                " • ",
                html.A("Submit your model", href="https://github.com/wisent-ai/uncensorbench#how-to-submit", target="_blank"),
            ], style={"color": "#888", "fontSize": "0.9em", "textAlign": "center"}),
        ], style={"padding": "20px"}),

    ], style={"fontFamily": "system-ui, -apple-system, sans-serif"})
