#!/usr/bin/env python3
"""
Web-based labeling tool using Flask for a nicer UI.

Usage:
    python labeling_web.py --responses responses.json --output labels.json --port 5050
"""

import argparse
import json
import os
from pathlib import Path

try:
    from flask import Flask, render_template_string, request, jsonify, redirect, url_for
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

if HAS_FLASK:
    app = Flask(__name__, static_folder="labeling_assets")
else:
    class _MissingFlaskApp:
        def route(self, *_args, **_kwargs):
            return lambda handler: handler

    app = _MissingFlaskApp()

# Global state
RESPONSES = []
LABELS = []
OUTPUT_PATH = "human_labels.json"
CURRENT_INDEX = 0

ASSET_DIRECTORY = Path(__file__).with_name("labeling_assets")
HTML_TEMPLATE = (ASSET_DIRECTORY / "index.html").read_text()


def load_responses(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def load_labels(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return []


def save_labels(filepath, labels):
    with open(filepath, "w") as f:
        json.dump(labels, f, indent=2)


def get_stats(labels):
    stats = {"refusal": 0, "not_refusal": 0, "partial": 0}
    for l in labels:
        label_type = l.get("human_label", "")
        if label_type in stats:
            stats[label_type] += 1
    return stats


@app.route("/")
def index():
    global CURRENT_INDEX

    labeled_ids = {l["id"] for l in LABELS}
    unlabeled = [r for r in RESPONSES if r["id"] not in labeled_ids]

    # Find the current response to show
    if CURRENT_INDEX >= len(RESPONSES):
        CURRENT_INDEX = len(RESPONSES) - 1
    if CURRENT_INDEX < 0:
        CURRENT_INDEX = 0

    current = RESPONSES[CURRENT_INDEX] if RESPONSES else None

    return render_template_string(
        HTML_TEMPLATE,
        current=current,
        index=CURRENT_INDEX,
        total=len(RESPONSES),
        labeled=len(LABELS),
        stats=get_stats(LABELS),
    )


@app.route("/label", methods=["POST"])
def do_label():
    global CURRENT_INDEX, LABELS

    data = request.get_json()
    label_type = data.get("label")

    if CURRENT_INDEX < len(RESPONSES):
        item = RESPONSES[CURRENT_INDEX]

        # Remove any existing label for this item
        LABELS = [l for l in LABELS if l["id"] != item["id"]]

        # Add new label
        LABELS.append({
            "id": item["id"],
            "topic": item["topic"],
            "subtopic": item["subtopic"],
            "prompt": item["prompt"],
            "response": item["response"],
            "human_label": label_type,
            "is_refusal": label_type == "refusal",
        })

        save_labels(OUTPUT_PATH, LABELS)
        CURRENT_INDEX += 1

    return jsonify({"success": True})


@app.route("/skip", methods=["POST"])
def do_skip():
    global CURRENT_INDEX
    CURRENT_INDEX += 1
    return jsonify({"success": True})


@app.route("/navigate", methods=["POST"])
def do_navigate():
    global CURRENT_INDEX
    data = request.get_json()
    delta = data.get("delta", 0)
    CURRENT_INDEX = max(0, min(len(RESPONSES) - 1, CURRENT_INDEX + delta))
    return jsonify({"success": True})


def main():
    global RESPONSES, LABELS, OUTPUT_PATH

    if not HAS_FLASK:
        print("Error: Flask is required for the web interface.")
        print("Install with: pip install flask")
        return

    parser = argparse.ArgumentParser(description="Web-based labeling tool")
    parser.add_argument("--responses", type=str, required=True, help="Path to responses JSON file")
    parser.add_argument("--output", type=str, default="human_labels.json", help="Output path for labels")
    parser.add_argument("--port", type=int, default=5050, help="Port to run server on")

    args = parser.parse_args()

    OUTPUT_PATH = args.output
    RESPONSES = load_responses(args.responses)
    LABELS = load_labels(args.output)

    print(f"Loaded {len(RESPONSES)} responses")
    print(f"Loaded {len(LABELS)} existing labels")
    print(f"\nStarting server at http://localhost:{args.port}")
    print("Press Ctrl+C to stop and save")

    try:
        app.run(host="0.0.0.0", port=args.port, debug=False)
    except KeyboardInterrupt:
        save_labels(OUTPUT_PATH, LABELS)
        print(f"\nSaved {len(LABELS)} labels to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
