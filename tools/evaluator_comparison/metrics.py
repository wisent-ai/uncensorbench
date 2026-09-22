"""Human-label input and classification metrics."""

import json

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
