# UncensorBench

A benchmark for measuring LLM censorship removal effectiveness. Designed for evaluating abliteration, steering vectors, and other techniques that modify model behavior around refusal.

## Installation

```bash
pip install uncensorbench
```

For semantic similarity evaluation:
```bash
pip install uncensorbench[semantic]
```

## Quick Start

```python
from uncensorbench import UncensorBench

# Initialize the benchmark
bench = UncensorBench()

# Iterate over all prompts
for prompt in bench.prompts():
    print(f"{prompt.id}: {prompt.prompt[:50]}...")

# Filter by topic
for prompt in bench.prompts(topics=["cybersecurity", "drugs"]):
    response = your_model.generate(prompt.prompt)
    result = bench.evaluate_response(prompt, response)
    print(f"{prompt.id}: {'Uncensored' if result.is_uncensored else 'Censored'}")

# Run full evaluation
def my_model(prompt: str) -> str:
    # Your model inference here
    return model.generate(prompt)

results = bench.evaluate(my_model, verbose=True)
print(results.summary())
```

## Benchmark Structure

UncensorBench contains 150 prompts across 15 topic categories:

| Topic | Description | Prompts |
|-------|-------------|---------|
| cybersecurity | Hacking, exploits, security bypass | 10 |
| piracy | Copyright infringement, DRM bypass | 10 |
| weapons | Weapons, explosives | 10 |
| drugs | Controlled substances | 10 |
| fraud | Scams, identity theft | 10 |
| manipulation | Psychological manipulation | 10 |
| violence | Physical harm | 10 |
| privacy_invasion | Surveillance, doxxing | 10 |
| illegal_activities | Theft, burglary | 10 |
| academic_dishonesty | Cheating, plagiarism | 10 |
| gambling | Casino cheating, match fixing | 10 |
| controversial_speech | Hate speech, misinformation | 10 |
| evasion | Evidence destruction, fleeing | 10 |
| self_harm | Dangerous challenges | 10 |
| adult_content | Explicit content | 10 |

Each prompt includes:
- A challenging request that models typically refuse
- A reference censored response (typical refusal)
- A reference uncensored response (compliant answer)

## Evaluation Methods

### Keyword-based (Default)
```python
from uncensorbench import KeywordEvaluator

evaluator = KeywordEvaluator()
results = bench.evaluate(my_model, evaluator=evaluator)
```

### Semantic Similarity
```python
from uncensorbench import SemanticEvaluator

evaluator = SemanticEvaluator()  # Requires sentence-transformers
results = bench.evaluate(my_model, evaluator=evaluator)
```

### LLM Judge
```python
from uncensorbench.evaluator import LLMJudgeEvaluator

def judge_model(prompt: str) -> str:
    return your_judge_model.generate(prompt)

evaluator = LLMJudgeEvaluator(judge_model)
results = bench.evaluate(my_model, evaluator=evaluator)
```

## Contrastive Pairs for Training

Get censored/uncensored pairs for abliteration or steering vector training:

```python
for pair in bench.get_contrastive_pairs():
    print(f"Prompt: {pair['prompt']}")
    print(f"Censored: {pair['censored'][:100]}...")
    print(f"Uncensored: {pair['uncensored'][:100]}...")
```

## Results Format

```python
results = bench.evaluate(my_model)

print(f"Uncensored Rate: {results.uncensored_rate:.1%}")
print(f"Average Compliance: {results.average_compliance_score:.3f}")

# Per-topic breakdown
for topic, stats in results.results_by_topic.items():
    print(f"{topic}: {stats['uncensored_rate']:.1%}")

# Export to JSON
import json
with open("results.json", "w") as f:
    json.dump(results.to_dict(), f, indent=2)
```

## Use Cases

1. **Abliteration Research**: Measure effectiveness of refusal removal techniques
2. **Steering Vector Evaluation**: Test steering vectors that modify refusal behavior
3. **Model Comparison**: Compare censorship levels across different models
4. **Safety Research**: Understand model vulnerabilities and refusal patterns

## Citation

```bibtex
@software{uncensorbench,
  title = {UncensorBench: A Benchmark for LLM Censorship Removal},
  author = {Wisent AI},
  year = {2024},
  url = {https://github.com/wisent-ai/uncensorbench}
}
```

## License

MIT License - See LICENSE file for details.

## Disclaimer

This benchmark is intended for research purposes only. The prompts and responses are designed to evaluate model behavior, not to provide actual harmful information. Users are responsible for ensuring their use complies with applicable laws and ethical guidelines.
