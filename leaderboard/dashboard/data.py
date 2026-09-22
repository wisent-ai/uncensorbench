"""Leaderboard CSV and paired-method analysis."""

import os

import pandas as pd

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
