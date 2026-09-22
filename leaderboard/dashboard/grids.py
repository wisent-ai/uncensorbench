"""Grid definitions for dashboard records."""

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
