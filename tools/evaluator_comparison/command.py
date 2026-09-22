"""CLI command for evaluator comparisons."""

import argparse

from .analysis import HAS_ML_DEPS, generate_report, run_evaluators
from .metrics import load_labels

def main():
    parser = argparse.ArgumentParser(description="Compare evaluators against human labels")
    parser.add_argument("--labels", type=str, required=True, help="Path to human labels JSON file")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model for LLM-based evaluators")
    parser.add_argument("--no-model", action="store_true", help="Skip LLM-based evaluators (KeywordEvaluator only)")
    parser.add_argument("--output", type=str, default="evaluator_comparison_report.txt", help="Output report path")

    args = parser.parse_args()

    if not HAS_ML_DEPS and not args.no_model:
        print("Warning: ML dependencies not available. Running with --no-model")
        args.no_model = True

    labels = load_labels(args.labels)
    print(f"Loaded {len(labels)} human labels")

    # Filter to only include binary labels (exclude 'partial' for now, or treat as refusal)
    # For simplicity, we treat 'partial' as not_refusal (the model did provide some info)
    for item in labels:
        if item.get('human_label') == 'partial':
            item['is_refusal'] = False

    results, has_semantic, has_llm_evals = run_evaluators(
        labels,
        model_name=args.model if not args.no_model else None,
        use_model=not args.no_model
    )

    generate_report(labels, results, has_semantic, has_llm_evals, args.output)
