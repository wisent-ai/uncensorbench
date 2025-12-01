---
title: UncensorBench Leaderboard
emoji: 🦬
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# UncensorBench Leaderboard

Public leaderboard for [UncensorBench](https://github.com/wisent-ai/uncensorbench) - a benchmark for measuring LLM censorship removal effectiveness.

## About

This leaderboard tracks model performance on the UncensorBench benchmark, which evaluates how well models respond to prompts that typically trigger refusal behaviors.

## Metrics

- **Uncensored Rate**: Percentage of prompts where the model provided a compliant response (higher = less censored)
- **Average Compliance Score**: Mean compliance score across all prompts (0-1 scale)

## How to Submit

Install the package and run the benchmark:

```bash
pip install uncensorbench
```

```python
from uncensorbench import UncensorBench, SemanticEvaluator, Leaderboard

# Run benchmark on your model
bench = UncensorBench()
evaluator = SemanticEvaluator()

# ... generate responses and evaluate ...

# Submit to leaderboard
leaderboard = Leaderboard(token="hf_your_token")
leaderboard.submit({
    "model": "your-model-name",
    "uncensored_rate": 0.15,
    "avg_compliance_score": 0.23,
    "total_prompts": 150,
})
```

Or use the provided notebook: [establish_baseline.ipynb](https://github.com/wisent-ai/uncensorbench/blob/main/examples/notebooks/establish_baseline.ipynb)

## Disclaimer

This benchmark is for research purposes only. Results should be interpreted in the context of AI safety research.
