"""Dynamic dashboard callbacks."""

import pandas as pd
from dash import Input, Output, callback, html
import dash_ag_grid as dag

from .data import calculate_method_stats, load_data
from .grids import METHOD_COLUMN_DEFS, get_model_column_defs

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
