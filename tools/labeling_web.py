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

app = Flask(__name__)

# Global state
RESPONSES = []
LABELS = []
OUTPUT_PATH = "human_labels.json"
CURRENT_INDEX = 0

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Response Labeling Tool</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            border-radius: 8px;
            padding: 24px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .progress {
            font-size: 18px;
            color: #666;
        }
        .progress-bar {
            width: 300px;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: #4CAF50;
            transition: width 0.3s;
        }
        .meta {
            background: #f0f0f0;
            padding: 12px;
            border-radius: 4px;
            margin-bottom: 16px;
            font-size: 14px;
        }
        .meta span {
            margin-right: 16px;
        }
        .section {
            margin-bottom: 24px;
        }
        .section-title {
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.5px;
        }
        .content {
            background: #fafafa;
            padding: 16px;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
            white-space: pre-wrap;
            font-family: 'Menlo', 'Monaco', monospace;
            font-size: 13px;
            line-height: 1.5;
            max-height: 400px;
            overflow-y: auto;
        }
        .prompt-content {
            background: #e3f2fd;
            border-color: #90caf9;
        }
        .response-content {
            background: #fff3e0;
            border-color: #ffcc80;
        }
        .buttons {
            display: flex;
            gap: 12px;
            justify-content: center;
            flex-wrap: wrap;
        }
        .btn {
            padding: 16px 32px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .btn-refusal {
            background: #ef5350;
            color: white;
        }
        .btn-not-refusal {
            background: #66bb6a;
            color: white;
        }
        .btn-partial {
            background: #ffa726;
            color: white;
        }
        .btn-skip {
            background: #78909c;
            color: white;
        }
        .nav-buttons {
            display: flex;
            gap: 8px;
            margin-top: 16px;
            justify-content: center;
        }
        .btn-nav {
            padding: 8px 16px;
            font-size: 14px;
            background: #e0e0e0;
            color: #333;
        }
        .keyboard-hint {
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 16px;
        }
        .stats {
            display: flex;
            gap: 24px;
            justify-content: center;
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid #e0e0e0;
        }
        .stat {
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: 600;
        }
        .stat-label {
            font-size: 12px;
            color: #666;
        }
        .refusal { color: #ef5350; }
        .not-refusal { color: #66bb6a; }
        .partial { color: #ffa726; }
        .reference-toggle {
            margin-top: 16px;
            text-align: center;
        }
        .reference-toggle button {
            background: none;
            border: 1px solid #ccc;
            padding: 8px 16px;
            cursor: pointer;
            border-radius: 4px;
        }
        .references {
            display: none;
            margin-top: 16px;
        }
        .references.show {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Response Labeling Tool</h1>
            <div class="progress">
                {{ labeled }} / {{ total }} labeled
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ (labeled / total * 100) if total > 0 else 0 }}%"></div>
                </div>
            </div>
        </div>

        {% if current %}
        <div class="meta">
            <span><strong>ID:</strong> {{ current.id }}</span>
            <span><strong>Topic:</strong> {{ current.topic }}</span>
            <span><strong>Subtopic:</strong> {{ current.subtopic }}</span>
            <span><strong>Index:</strong> {{ index + 1 }} / {{ total }}</span>
        </div>

        <div class="section">
            <div class="section-title">Prompt</div>
            <div class="content prompt-content">{{ current.prompt }}</div>
        </div>

        <div class="section">
            <div class="section-title">Model Response</div>
            <div class="content response-content">{{ current.response }}</div>
        </div>

        <div class="reference-toggle">
            <button onclick="toggleReferences()">Show/Hide Reference Responses</button>
        </div>

        <div class="references" id="references">
            <div class="section">
                <div class="section-title">Reference: Censored Response</div>
                <div class="content">{{ current.censored_reference or 'N/A' }}</div>
            </div>
            <div class="section">
                <div class="section-title">Reference: Uncensored Response</div>
                <div class="content">{{ current.uncensored_reference or 'N/A' }}</div>
            </div>
        </div>

        <div class="buttons">
            <button class="btn btn-refusal" onclick="label('refusal')">
                REFUSAL (R)
            </button>
            <button class="btn btn-not-refusal" onclick="label('not_refusal')">
                NOT REFUSAL (N)
            </button>
            <button class="btn btn-partial" onclick="label('partial')">
                PARTIAL (P)
            </button>
            <button class="btn btn-skip" onclick="skip()">
                SKIP (S)
            </button>
        </div>

        <div class="nav-buttons">
            <button class="btn btn-nav" onclick="navigate(-1)" {% if index == 0 %}disabled{% endif %}>
                ← Previous
            </button>
            <button class="btn btn-nav" onclick="navigate(1)" {% if index >= total - 1 %}disabled{% endif %}>
                Next →
            </button>
        </div>

        <div class="keyboard-hint">
            Keyboard shortcuts: R = Refusal, N = Not Refusal, P = Partial, S = Skip, ← → = Navigate
        </div>
        {% else %}
        <p>All responses have been labeled! You can review by using the Previous/Next buttons.</p>
        {% endif %}

        <div class="stats">
            <div class="stat">
                <div class="stat-value refusal">{{ stats.refusal }}</div>
                <div class="stat-label">Refusals</div>
            </div>
            <div class="stat">
                <div class="stat-value not-refusal">{{ stats.not_refusal }}</div>
                <div class="stat-label">Not Refusals</div>
            </div>
            <div class="stat">
                <div class="stat-value partial">{{ stats.partial }}</div>
                <div class="stat-label">Partial</div>
            </div>
        </div>
    </div>

    <script>
        function label(labelType) {
            fetch('/label', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({label: labelType})
            }).then(() => window.location.reload());
        }

        function skip() {
            fetch('/skip', {method: 'POST'}).then(() => window.location.reload());
        }

        function navigate(delta) {
            fetch('/navigate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({delta: delta})
            }).then(() => window.location.reload());
        }

        function toggleReferences() {
            document.getElementById('references').classList.toggle('show');
        }

        document.addEventListener('keydown', function(e) {
            if (e.key === 'r' || e.key === 'R') label('refusal');
            else if (e.key === 'n' || e.key === 'N') label('not_refusal');
            else if (e.key === 'p' || e.key === 'P') label('partial');
            else if (e.key === 's' || e.key === 'S') skip();
            else if (e.key === 'ArrowLeft') navigate(-1);
            else if (e.key === 'ArrowRight') navigate(1);
        });
    </script>
</body>
</html>
"""


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
