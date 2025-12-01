"""
UncensorBench Leaderboard - A Dash application for tracking LLM censorship removal benchmarks.
"""

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_ag_grid as dag
import pandas as pd
import os

# Initialize the Dash app
app = dash.Dash(__name__, title="UncensorBench Leaderboard")
server = app.server

# Load leaderboard data
DATA_FILE = "leaderboard.csv"

# Known method descriptions (for display purposes, but we accept any method)
METHOD_DESCRIPTIONS = {
    "none": "Baseline (no modification)",
    "abliteration": "Abliteration technique",
    "steering": "Steering vectors",
    "finetuning": "Fine-tuning based",
    "prompting": "Prompt-based jailbreaking",
    "other": "Other methods",
}

# Colors for known methods, dynamic methods get auto-assigned colors
METHOD_COLORS = {
    "none": "#9E9E9E",
    "abliteration": "#E91E63",
    "steering": "#2196F3",
    "finetuning": "#4CAF50",
    "prompting": "#FF9800",
    "other": "#9C27B0",
}

# Fallback colors for dynamically discovered methods
DYNAMIC_COLORS = ["#00BCD4", "#795548", "#607D8B", "#3F51B5", "#009688", "#CDDC39", "#FF5722", "#673AB7"]


def load_data():
    """Load leaderboard data from CSV."""
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        # Sort by uncensored_rate descending
        if len(df) > 0:
            df = df.sort_values("uncensored_rate", ascending=False).reset_index(drop=True)
            df.insert(0, "Rank", range(1, len(df) + 1))
        return df
    else:
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=[
            "Rank", "model", "model_family", "model_size", "method",
            "uncensored_rate", "avg_compliance_score",
            "total_prompts", "timestamp", "submitter", "sample_responses_url"
        ])


def get_method_color(method, method_index=0):
    """Get color for a method, using predefined or dynamic colors."""
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    # Assign a dynamic color based on index
    return DYNAMIC_COLORS[method_index % len(DYNAMIC_COLORS)]


def calculate_method_stats(df):
    """
    Calculate statistics for each method based on PAIRED comparisons only.

    A paired comparison requires the exact same base model to have both:
    - A baseline submission (method="none")
    - A method-applied submission (method=X)

    Only shows delta for methods where paired comparisons exist.
    """
    if len(df) == 0:
        return pd.DataFrame(), {}

    # Get all unique methods from the actual data
    all_methods = df["method"].dropna().unique().tolist()

    # Build dynamic color mapping for any new methods
    dynamic_method_colors = {}
    dynamic_idx = 0
    for method in all_methods:
        if method in METHOD_COLORS:
            dynamic_method_colors[method] = METHOD_COLORS[method]
        else:
            dynamic_method_colors[method] = DYNAMIC_COLORS[dynamic_idx % len(DYNAMIC_COLORS)]
            dynamic_idx += 1

    # Get baseline data - create lookup by exact model name
    baseline_df = df[df["method"] == "none"].copy()
    baseline_lookup = {}
    if len(baseline_df) > 0:
        for _, row in baseline_df.iterrows():
            model_name = row.get("model", "")
            baseline_lookup[model_name] = {
                "uncensored_rate": row["uncensored_rate"],
                "avg_compliance_score": row.get("avg_compliance_score", 0),
            }

    # Calculate paired comparisons for each method
    method_stats = []

    for method in all_methods:
        method_df = df[df["method"] == method]

        if method == "none":
            # Baseline method - show stats but no delta
            if len(method_df) > 0:
                avg_rate = method_df["uncensored_rate"].mean()
                max_rate = method_df["uncensored_rate"].max()
                min_rate = method_df["uncensored_rate"].min()
                avg_compliance = method_df["avg_compliance_score"].mean()
                best_model = method_df.loc[method_df["uncensored_rate"].idxmax(), "model"]
                description = METHOD_DESCRIPTIONS.get(method, method.replace("_", " ").title())

                method_stats.append({
                    "method": method,
                    "description": description,
                    "num_models": len(method_df),
                    "num_pairs": len(method_df),
                    "avg_uncensored_rate": avg_rate,
                    "delta_from_baseline": 0.0,
                    "max_uncensored_rate": max_rate,
                    "min_uncensored_rate": min_rate,
                    "avg_compliance_score": avg_compliance,
                    "best_model": best_model,
                })
        else:
            # Non-baseline method - only count paired comparisons
            paired_data = []

            for _, row in method_df.iterrows():
                method_model = row.get("model", "")
                method_rate = row["uncensored_rate"]
                method_compliance = row.get("avg_compliance_score", 0)

                # Find exact baseline match by model_family + model_size
                model_family = row.get("model_family", "")
                model_size = row.get("model_size", "")

                # Look for baseline with same family and size
                baseline_match = None
                for baseline_model, baseline_data in baseline_lookup.items():
                    baseline_row = baseline_df[baseline_df["model"] == baseline_model].iloc[0]
                    if (baseline_row.get("model_family", "") == model_family and
                        baseline_row.get("model_size", "") == model_size):
                        baseline_match = baseline_data
                        break

                if baseline_match is not None:
                    paired_data.append({
                        "model": method_model,
                        "method_rate": method_rate,
                        "baseline_rate": baseline_match["uncensored_rate"],
                        "delta": method_rate - baseline_match["uncensored_rate"],
                        "method_compliance": method_compliance,
                    })

            # Only add method if it has paired comparisons
            if len(paired_data) > 0:
                avg_delta = sum(p["delta"] for p in paired_data) / len(paired_data)
                avg_rate = sum(p["method_rate"] for p in paired_data) / len(paired_data)
                max_rate = max(p["method_rate"] for p in paired_data)
                min_rate = min(p["method_rate"] for p in paired_data)
                avg_compliance = sum(p["method_compliance"] for p in paired_data) / len(paired_data)

                # Best model is the one with highest delta
                best_pair = max(paired_data, key=lambda x: x["delta"])
                best_model = best_pair["model"]

                description = METHOD_DESCRIPTIONS.get(method, method.replace("_", " ").title())

                method_stats.append({
                    "method": method,
                    "description": description,
                    "num_models": len(method_df),
                    "num_pairs": len(paired_data),
                    "avg_uncensored_rate": avg_rate,
                    "delta_from_baseline": avg_delta,
                    "max_uncensored_rate": max_rate,
                    "min_uncensored_rate": min_rate,
                    "avg_compliance_score": avg_compliance,
                    "best_model": best_model,
                })

    return pd.DataFrame(method_stats), dynamic_method_colors


# Column definitions for Models AG Grid
MODEL_COLUMN_DEFS = [
    {
        "field": "Rank",
        "headerName": "🏆",
        "width": 70,
        "pinned": "left",
        "sortable": True,
    },
    {
        "field": "model",
        "headerName": "Model",
        "width": 300,
        "pinned": "left",
        "sortable": True,
        "filter": True,
    },
    {
        "field": "model_family",
        "headerName": "Family",
        "width": 120,
        "sortable": True,
        "filter": True,
    },
    {
        "field": "model_size",
        "headerName": "Size",
        "width": 80,
        "sortable": True,
        "filter": True,
    },
    {
        "field": "method",
        "headerName": "Method",
        "width": 120,
        "sortable": True,
        "filter": True,
    },
    {
        "field": "uncensored_rate",
        "headerName": "Uncensored Rate ⬆️",
        "width": 160,
        "sortable": True,
        "valueFormatter": {"function": "d3.format('.1%')(params.value)"},
    },
    {
        "field": "avg_compliance_score",
        "headerName": "Avg Compliance",
        "width": 140,
        "sortable": True,
        "valueFormatter": {"function": "d3.format('.3f')(params.value)"},
    },
    {
        "field": "total_prompts",
        "headerName": "Prompts",
        "width": 90,
        "sortable": True,
    },
    {
        "field": "timestamp",
        "headerName": "Submitted",
        "width": 180,
        "sortable": True,
    },
    {
        "field": "submitter",
        "headerName": "Submitter",
        "width": 130,
        "sortable": True,
        "filter": True,
    },
]

# Column definitions for Methods AG Grid (paired comparisons only)
METHOD_COLUMN_DEFS = [
    {
        "field": "method",
        "headerName": "Method",
        "width": 130,
        "pinned": "left",
        "sortable": True,
    },
    {
        "field": "description",
        "headerName": "Description",
        "width": 180,
        "sortable": True,
    },
    {
        "field": "num_pairs",
        "headerName": "# Pairs",
        "width": 80,
        "sortable": True,
    },
    {
        "field": "delta_from_baseline",
        "headerName": "Δ vs Baseline ⬆️",
        "width": 140,
        "sortable": True,
        "valueFormatter": {"function": "params.value >= 0 ? '+' + d3.format('.1%')(params.value) : d3.format('.1%')(params.value)"},
        "cellStyle": {"function": "params.value > 0 ? {'color': '#4CAF50', 'fontWeight': 'bold'} : params.value < 0 ? {'color': '#f44336'} : {}"},
    },
    {
        "field": "avg_uncensored_rate",
        "headerName": "Avg Rate",
        "width": 100,
        "sortable": True,
        "valueFormatter": {"function": "d3.format('.1%')(params.value)"},
    },
    {
        "field": "max_uncensored_rate",
        "headerName": "Best Rate",
        "width": 100,
        "sortable": True,
        "valueFormatter": {"function": "d3.format('.1%')(params.value)"},
    },
    {
        "field": "min_uncensored_rate",
        "headerName": "Worst Rate",
        "width": 100,
        "sortable": True,
        "valueFormatter": {"function": "d3.format('.1%')(params.value)"},
    },
    {
        "field": "avg_compliance_score",
        "headerName": "Avg Compliance",
        "width": 130,
        "sortable": True,
        "valueFormatter": {"function": "d3.format('.3f')(params.value)"},
    },
    {
        "field": "best_model",
        "headerName": "Best Model",
        "width": 260,
        "sortable": True,
    },
]

# Topic-specific columns (added dynamically if present)
TOPIC_COLUMNS = [
    "cybersecurity", "piracy", "weapons", "drugs", "fraud",
    "manipulation", "violence", "privacy_invasion", "illegal_activities",
    "academic_dishonesty", "gambling", "controversial_speech",
    "evasion", "self_harm", "adult_content"
]


def get_model_column_defs(df):
    """Get column definitions based on available data."""
    cols = MODEL_COLUMN_DEFS.copy()

    # Add topic columns if they exist in the data
    for topic in TOPIC_COLUMNS:
        if topic in df.columns:
            cols.append({
                "field": topic,
                "headerName": topic.replace("_", " ").title(),
                "width": 130,
                "sortable": True,
                "valueFormatter": {"function": "d3.format('.1%')(params.value)"},
            })

    return cols


# App layout
app.layout = html.Div([
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
                href="https://github.com/wisent-ai/uncensorbench/blob/main/examples/notebooks/establish_baseline.ipynb",
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


@callback(
    Output("stats-summary", "children"),
    Input("refresh-interval", "n_intervals")
)
def update_stats(n):
    """Update the stats summary."""
    df = load_data()

    if len(df) > 0:
        # Calculate method stats for the summary
        baseline_df = df[df["method"] == "none"]
        baseline_avg = baseline_df["uncensored_rate"].mean() if len(baseline_df) > 0 else 0

        # Find best non-baseline method
        non_baseline = df[df["method"] != "none"]
        best_method_avg = 0
        best_method = "N/A"
        if len(non_baseline) > 0:
            method_avgs = non_baseline.groupby("method")["uncensored_rate"].mean()
            if len(method_avgs) > 0:
                best_method = method_avgs.idxmax()
                best_method_avg = method_avgs.max()

        best_delta = best_method_avg - baseline_avg if best_method_avg > 0 else 0

        stats = [
            html.Div([
                html.Div(str(len(df)), style={"fontSize": "2em", "fontWeight": "bold", "color": "#2196F3"}),
                html.Div("Models", style={"color": "#666"}),
            ], style={"textAlign": "center"}),
            html.Div([
                html.Div(f"{baseline_avg:.1%}", style={"fontSize": "2em", "fontWeight": "bold", "color": "#9E9E9E"}),
                html.Div("Baseline Avg", style={"color": "#666"}),
            ], style={"textAlign": "center"}),
            html.Div([
                html.Div(f"{df['uncensored_rate'].max():.1%}", style={"fontSize": "2em", "fontWeight": "bold", "color": "#FF9800"}),
                html.Div("Best Rate", style={"color": "#666"}),
            ], style={"textAlign": "center"}),
            html.Div([
                html.Div(
                    f"+{best_delta:.1%}" if best_delta > 0 else f"{best_delta:.1%}",
                    style={"fontSize": "2em", "fontWeight": "bold", "color": "#4CAF50" if best_delta > 0 else "#f44336"}
                ),
                html.Div(f"Best Method Δ ({best_method})", style={"color": "#666"}),
            ], style={"textAlign": "center"}),
        ]
    else:
        stats = [
            html.Div([
                html.Div("0", style={"fontSize": "2em", "fontWeight": "bold", "color": "#2196F3"}),
                html.Div("Models", style={"color": "#666"}),
            ], style={"textAlign": "center"}),
            html.Div([
                html.P("No submissions yet. Be the first to submit!", style={"color": "#666"}),
            ], style={"textAlign": "center"}),
        ]

    return stats


@callback(
    Output("tab-content", "children"),
    [Input("view-tabs", "value"),
     Input("refresh-interval", "n_intervals")]
)
def render_tab_content(tab, n):
    """Render content based on selected tab."""
    df = load_data()

    if tab == "models":
        # Models leaderboard view
        col_defs = get_model_column_defs(df)
        row_data = df.to_dict("records") if len(df) > 0 else []

        # Build responses links section
        responses_links = []
        if len(df) > 0:
            for _, row in df.iterrows():
                url = row.get("sample_responses_url")
                if pd.notna(url) and url:
                    model = row.get("model", "Unknown")
                    responses_links.append(
                        html.Li([
                            html.Strong(model),
                            html.Span(": "),
                            html.Code(url, style={"fontSize": "0.85em", "wordBreak": "break-all"}),
                        ], style={"marginBottom": "5px"})
                    )

        return html.Div([
            dag.AgGrid(
                id="leaderboard-grid",
                columnDefs=col_defs,
                rowData=row_data,
                defaultColDef={
                    "resizable": True,
                    "sortable": True,
                },
                dashGridOptions={
                    "pagination": True,
                    "paginationPageSize": 50,
                    "animateRows": True,
                    "rowSelection": "single",
                },
                style={"height": "600px"},
                className="ag-theme-alpine",
            ),
            # Sample responses section
            html.Div([
                html.H4("📄 Sample Responses", style={"marginTop": "20px", "marginBottom": "10px"}),
                html.P("Copy and paste these URLs to view detailed model responses:", style={"color": "#666", "fontSize": "0.9em"}),
                html.Ul(responses_links) if responses_links else html.P("No sample responses available yet.", style={"color": "#999"}),
            ], style={
                "backgroundColor": "#f9f9f9",
                "padding": "15px",
                "borderRadius": "8px",
                "marginTop": "20px",
            }) if responses_links else None,
        ])

    elif tab == "methods":
        # Methods comparison view
        method_df, method_colors = calculate_method_stats(df)
        row_data = method_df.to_dict("records") if len(method_df) > 0 else []

        # Sort by delta from baseline descending
        if len(method_df) > 0:
            method_df = method_df.sort_values("delta_from_baseline", ascending=False)
            row_data = method_df.to_dict("records")

        # Build method legend from actual data
        method_legend_items = []
        for _, row in method_df.iterrows():
            method = row["method"]
            desc = row["description"]
            color = method_colors.get(method, "#666")
            method_legend_items.append(
                html.Div([
                    html.Span(
                        f"● {method}",
                        style={"color": color, "fontWeight": "bold", "marginRight": "10px"}
                    ),
                    html.Span(desc, style={"color": "#666"}),
                ], style={"marginBottom": "8px"})
            )

        return html.Div([
            # Method comparison description
            html.Div([
                html.P([
                    "Compare censorship removal methods using ",
                    html.Strong("paired comparisons only"),
                    ". Delta (Δ) is calculated by comparing the ",
                    html.Strong("same base model"),
                    " with and without each method applied."
                ], style={"color": "#666", "marginBottom": "5px"}),
                html.P([
                    "Methods are only shown if they have at least one paired comparison ",
                    "(matching model_family + model_size with a baseline 'none' submission)."
                ], style={"color": "#666", "fontSize": "0.9em", "marginBottom": "15px"}),
            ]),

            # Methods grid
            dag.AgGrid(
                id="methods-grid",
                columnDefs=METHOD_COLUMN_DEFS,
                rowData=row_data,
                defaultColDef={
                    "resizable": True,
                    "sortable": True,
                },
                dashGridOptions={
                    "animateRows": True,
                    "rowSelection": "single",
                },
                style={"height": "400px"},
                className="ag-theme-alpine",
            ),

            # Method legend - dynamically built from actual data
            html.Div([
                html.H4("Method Definitions", style={"marginTop": "30px", "marginBottom": "15px"}),
                html.Div(
                    method_legend_items if method_legend_items else [html.P("No methods submitted yet.", style={"color": "#666"})],
                    style={"columns": "2", "columnGap": "40px"} if len(method_legend_items) > 3 else {}
                ),
            ], style={
                "backgroundColor": "#f9f9f9",
                "padding": "20px",
                "borderRadius": "8px",
                "marginTop": "20px",
            }),
        ])

    return html.Div("Select a tab")


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=7860)
