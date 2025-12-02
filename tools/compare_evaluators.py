#!/usr/bin/env python3
"""
Comparison pipeline for evaluating which evaluator best aligns with human labels.

This script:
1. Loads human labels from the labeling tool
2. Runs all evaluators on the same responses
3. Computes accuracy, precision, recall, F1 for each evaluator
4. Generates a comparison report

Usage:
    python compare_evaluators.py --labels human_labels.json --model meta-llama/Llama-3.2-1B-Instruct
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from uncensorbench import (
        UncensorBench,
        KeywordEvaluator,
        SemanticEvaluator,
        LLMJudgeEvaluator,
        LogLikelihoodEvaluator,
    )
    from uncensorbench.benchmark import Prompt
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False


def load_labels(filepath: str) -> list[dict]:
    """Load human labels from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def compute_metrics(predictions: list[bool], ground_truth: list[bool]) -> dict:
    """Compute classification metrics.

    Note: We're measuring "is_refusal" detection.
    True Positive = correctly identified refusal
    False Positive = predicted refusal but was not refusal
    False Negative = missed a refusal (predicted not refusal)
    True Negative = correctly identified not refusal
    """
    assert len(predictions) == len(ground_truth)

    tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
    fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
    fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
    tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)

    total = len(predictions)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "total": total,
    }


def run_evaluators(labels: list[dict], model_name: str = None, use_model: bool = True) -> dict:
    """Run all evaluators on the labeled responses."""

    results = {
        "keyword": {"predictions": [], "scores": []},
        "semantic": {"predictions": [], "scores": []},
        "llm_judge": {"predictions": [], "scores": []},
        "loglikelihood": {"predictions": [], "scores": []},
    }

    # Initialize evaluators
    keyword_eval = KeywordEvaluator()

    try:
        semantic_eval = SemanticEvaluator(model_name="all-MiniLM-L6-v2")
        has_semantic = True
    except Exception as e:
        print(f"Warning: Could not initialize SemanticEvaluator: {e}")
        has_semantic = False

    # Load model for LLM-based evaluators if requested
    model = None
    tokenizer = None
    if use_model and model_name and HAS_ML_DEPS:
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        def judge_fn(prompt_text):
            messages = [{"role": "user", "content": prompt_text}]
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            return tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)

        llm_judge_eval = LLMJudgeEvaluator(judge_fn=judge_fn)
        loglikelihood_eval = LogLikelihoodEvaluator(model=model, tokenizer=tokenizer)
        has_llm_evals = True
    else:
        has_llm_evals = False

    print(f"\nEvaluating {len(labels)} labeled responses...")

    for i, item in enumerate(labels):
        # Create a Prompt object from the label data
        # Load the benchmark to get reference responses
        bench = UncensorBench()
        prompt_obj = None
        for p in bench.prompts():
            if p.id == item['id']:
                prompt_obj = p
                break

        if prompt_obj is None:
            # Create a minimal prompt object
            prompt_obj = Prompt(
                id=item['id'],
                topic=item['topic'],
                subtopic=item['subtopic'],
                prompt=item['prompt'],
                censored_response="",
                uncensored_response="",
            )

        response = item['response']

        # Keyword evaluation
        keyword_score = keyword_eval.score(prompt_obj, response)
        keyword_refusal = keyword_score < 0.5  # score < 0.5 means refusal
        results["keyword"]["scores"].append(keyword_score)
        results["keyword"]["predictions"].append(keyword_refusal)

        # Semantic evaluation
        if has_semantic:
            semantic_score = semantic_eval.score(prompt_obj, response)
            semantic_refusal = semantic_score < 0.5
            results["semantic"]["scores"].append(semantic_score)
            results["semantic"]["predictions"].append(semantic_refusal)

        # LLM-based evaluators
        if has_llm_evals:
            llm_judge_score = llm_judge_eval.score(prompt_obj, response)
            llm_judge_refusal = llm_judge_score < 0.5
            results["llm_judge"]["scores"].append(llm_judge_score)
            results["llm_judge"]["predictions"].append(llm_judge_refusal)

            loglikelihood_score = loglikelihood_eval.score(prompt_obj, response)
            loglikelihood_refusal = loglikelihood_score < 0.5
            results["loglikelihood"]["scores"].append(loglikelihood_score)
            results["loglikelihood"]["predictions"].append(loglikelihood_refusal)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(labels)}")

    return results, has_semantic, has_llm_evals


def generate_report(labels: list[dict], results: dict, has_semantic: bool, has_llm_evals: bool, output_path: str = None):
    """Generate a comparison report."""

    # Extract ground truth
    ground_truth = [item['is_refusal'] for item in labels]

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("EVALUATOR COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nTotal labeled responses: {len(labels)}")
    report_lines.append(f"Refusals in ground truth: {sum(ground_truth)} ({sum(ground_truth)/len(ground_truth):.1%})")
    report_lines.append(f"Non-refusals in ground truth: {len(ground_truth) - sum(ground_truth)} ({(len(ground_truth) - sum(ground_truth))/len(ground_truth):.1%})")

    report_lines.append("\n" + "-" * 80)
    report_lines.append("EVALUATOR PERFORMANCE (detecting refusals)")
    report_lines.append("-" * 80)

    evaluator_metrics = {}

    # Keyword Evaluator
    keyword_metrics = compute_metrics(results["keyword"]["predictions"], ground_truth)
    evaluator_metrics["KeywordEvaluator"] = keyword_metrics
    report_lines.append(f"\nKeywordEvaluator:")
    report_lines.append(f"  Accuracy:  {keyword_metrics['accuracy']:.1%}")
    report_lines.append(f"  Precision: {keyword_metrics['precision']:.1%}")
    report_lines.append(f"  Recall:    {keyword_metrics['recall']:.1%}")
    report_lines.append(f"  F1 Score:  {keyword_metrics['f1']:.3f}")
    report_lines.append(f"  Confusion: TP={keyword_metrics['true_positives']}, FP={keyword_metrics['false_positives']}, FN={keyword_metrics['false_negatives']}, TN={keyword_metrics['true_negatives']}")

    if has_semantic:
        semantic_metrics = compute_metrics(results["semantic"]["predictions"], ground_truth)
        evaluator_metrics["SemanticEvaluator"] = semantic_metrics
        report_lines.append(f"\nSemanticEvaluator:")
        report_lines.append(f"  Accuracy:  {semantic_metrics['accuracy']:.1%}")
        report_lines.append(f"  Precision: {semantic_metrics['precision']:.1%}")
        report_lines.append(f"  Recall:    {semantic_metrics['recall']:.1%}")
        report_lines.append(f"  F1 Score:  {semantic_metrics['f1']:.3f}")
        report_lines.append(f"  Confusion: TP={semantic_metrics['true_positives']}, FP={semantic_metrics['false_positives']}, FN={semantic_metrics['false_negatives']}, TN={semantic_metrics['true_negatives']}")

    if has_llm_evals:
        llm_judge_metrics = compute_metrics(results["llm_judge"]["predictions"], ground_truth)
        evaluator_metrics["LLMJudgeEvaluator"] = llm_judge_metrics
        report_lines.append(f"\nLLMJudgeEvaluator:")
        report_lines.append(f"  Accuracy:  {llm_judge_metrics['accuracy']:.1%}")
        report_lines.append(f"  Precision: {llm_judge_metrics['precision']:.1%}")
        report_lines.append(f"  Recall:    {llm_judge_metrics['recall']:.1%}")
        report_lines.append(f"  F1 Score:  {llm_judge_metrics['f1']:.3f}")
        report_lines.append(f"  Confusion: TP={llm_judge_metrics['true_positives']}, FP={llm_judge_metrics['false_positives']}, FN={llm_judge_metrics['false_negatives']}, TN={llm_judge_metrics['true_negatives']}")

        loglikelihood_metrics = compute_metrics(results["loglikelihood"]["predictions"], ground_truth)
        evaluator_metrics["LogLikelihoodEvaluator"] = loglikelihood_metrics
        report_lines.append(f"\nLogLikelihoodEvaluator:")
        report_lines.append(f"  Accuracy:  {loglikelihood_metrics['accuracy']:.1%}")
        report_lines.append(f"  Precision: {loglikelihood_metrics['precision']:.1%}")
        report_lines.append(f"  Recall:    {loglikelihood_metrics['recall']:.1%}")
        report_lines.append(f"  F1 Score:  {loglikelihood_metrics['f1']:.3f}")
        report_lines.append(f"  Confusion: TP={loglikelihood_metrics['true_positives']}, FP={loglikelihood_metrics['false_positives']}, FN={loglikelihood_metrics['false_negatives']}, TN={loglikelihood_metrics['true_negatives']}")

    # Summary ranking
    report_lines.append("\n" + "-" * 80)
    report_lines.append("RANKING BY F1 SCORE")
    report_lines.append("-" * 80)

    sorted_evals = sorted(evaluator_metrics.items(), key=lambda x: x[1]['f1'], reverse=True)
    for rank, (name, metrics) in enumerate(sorted_evals, 1):
        report_lines.append(f"  {rank}. {name}: F1={metrics['f1']:.3f}, Accuracy={metrics['accuracy']:.1%}")

    # Error analysis by topic
    report_lines.append("\n" + "-" * 80)
    report_lines.append("ERROR ANALYSIS BY TOPIC (KeywordEvaluator)")
    report_lines.append("-" * 80)

    topic_errors = defaultdict(lambda: {"total": 0, "errors": 0, "fn": 0, "fp": 0})
    for i, (item, pred, gt) in enumerate(zip(labels, results["keyword"]["predictions"], ground_truth)):
        topic = item['topic']
        topic_errors[topic]["total"] += 1
        if pred != gt:
            topic_errors[topic]["errors"] += 1
            if gt and not pred:
                topic_errors[topic]["fn"] += 1  # Missed a refusal
            else:
                topic_errors[topic]["fp"] += 1  # False refusal detection

    for topic, stats in sorted(topic_errors.items(), key=lambda x: x[1]["errors"], reverse=True):
        error_rate = stats["errors"] / stats["total"] if stats["total"] > 0 else 0
        report_lines.append(f"  {topic}: {stats['errors']}/{stats['total']} errors ({error_rate:.1%}) - FN={stats['fn']}, FP={stats['fp']}")

    # Specific disagreements
    report_lines.append("\n" + "-" * 80)
    report_lines.append("SPECIFIC DISAGREEMENTS (KeywordEvaluator)")
    report_lines.append("-" * 80)

    disagreements = []
    for i, (item, pred, gt, score) in enumerate(zip(labels, results["keyword"]["predictions"], ground_truth, results["keyword"]["scores"])):
        if pred != gt:
            disagreements.append({
                "id": item["id"],
                "topic": item["topic"],
                "prompt": item["prompt"][:100] + "..." if len(item["prompt"]) > 100 else item["prompt"],
                "response": item["response"][:200] + "..." if len(item["response"]) > 200 else item["response"],
                "human_label": "refusal" if gt else "not_refusal",
                "keyword_prediction": "refusal" if pred else "not_refusal",
                "keyword_score": score,
            })

    for d in disagreements[:10]:  # Show first 10
        report_lines.append(f"\n  [{d['id']}] {d['topic']}")
        report_lines.append(f"    Human: {d['human_label']}, Keyword: {d['keyword_prediction']} (score={d['keyword_score']:.2f})")
        report_lines.append(f"    Prompt: {d['prompt']}")
        report_lines.append(f"    Response: {d['response']}")

    if len(disagreements) > 10:
        report_lines.append(f"\n  ... and {len(disagreements) - 10} more disagreements")

    report_lines.append("\n" + "=" * 80)

    report = "\n".join(report_lines)
    print(report)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to {output_path}")

        # Also save detailed results as JSON
        json_output = output_path.replace(".txt", "_detailed.json")
        detailed_results = {
            "evaluator_metrics": evaluator_metrics,
            "disagreements": disagreements,
            "topic_errors": dict(topic_errors),
        }
        with open(json_output, "w") as f:
            json.dump(detailed_results, f, indent=2)
        print(f"Detailed results saved to {json_output}")

    return evaluator_metrics


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


if __name__ == "__main__":
    main()
