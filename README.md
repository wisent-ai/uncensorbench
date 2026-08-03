# UncensorBench

<!-- wisent-readme-signals:start -->
[![Release](https://img.shields.io/github/v/release/wisent-ai/uncensorbench?display_name=tag&sort=semver)](https://github.com/wisent-ai/uncensorbench/releases)
[![Downloads](https://img.shields.io/github/downloads/wisent-ai/uncensorbench/total)](https://github.com/wisent-ai/uncensorbench/releases)
[![License](https://img.shields.io/github/license/wisent-ai/uncensorbench)](https://github.com/wisent-ai/uncensorbench)
[![Discord](https://img.shields.io/badge/Discord-Join%20Wisent-5865F2?logo=discord&logoColor=white)](https://discord.gg/qRjpkthq54)
<!-- wisent-readme-signals:end -->

**UncensorBench is an offline-first benchmark for measuring how readily a language model complies with sensitive prompts.** It provides a versioned prompt corpus, deterministic and stochastic generation profiles, several response evaluators, JSON export, and an optional Hugging Face leaderboard client.

The score is a measurement of model behavior, not a safety certification. A higher compliance score is not inherently better, and results from different corpus versions, generation settings, or evaluators are not directly comparable.

## Safety warning

The bundled corpus includes harmful, illegal, self-harm, adult, evasion, and code-execution prompts. Review your model provider's terms and your organization's evaluation policy before use. Do not expose prompts or generated answers to untrusted users.

The default `hybrid` evaluator is **not a safe local default**:

- it requires an Anthropic API key for non-code judging, so prompts and responses leave the machine;
- code-category answers are executed in Docker;
- the current Docker runner mounts a writable host directory and uses `--network=host`;
- model and tokenizer loading uses Hugging Face `trust_remote_code=True`.

Use `--evaluator keyword` for a dependency-free inspection run, or `--evaluator semantic` for local semantic scoring. Do not use `hybrid` on untrusted model output until you have isolated the runner outside a developer workstation. Never pass secrets into the evaluation process or container.

## Install

```bash
pip install uncensorbench
```

For local semantic scoring:

```bash
pip install "uncensorbench[semantic]"
```

Running a Hugging Face model through the CLI additionally requires packages that the base distribution does not install:

```bash
pip install torch transformers accelerate sentencepiece
```

Model downloads can be large. Gated models such as Llama require acceptance of the model owner's terms and a Hugging Face token. The benchmark package is MIT-licensed; model weights and third-party datasets keep their own licenses.

## Inspect before running

These commands do not load model weights:

```bash
uncensorbench info
uncensorbench topics
uncensorbench list --topic controversial_speech
uncensorbench export --topic cybersecurity --output cybersecurity.json
```

`list` and `export` expose the bundled prompts and reference answers. Treat exported files as sensitive evaluation material and do not publish them accidentally.

## Run a local evaluation

Start with an explicit evaluator and output path:

```bash
uncensorbench run meta-llama/Llama-3.1-8B-Instruct \
  --topics controversial_speech \
  --inference-mode deterministic \
  --evaluator keyword \
  --output results/llama-3.1-8b.json
```

The CLI supports:

- `--topics TOPIC ...` — restrict the corpus;
- `--inference-mode deterministic|stochastic|both` — one reproducible sample, five sampled responses, or both;
- `--evaluator keyword|semantic|llm-judge|hybrid` — choose the scoring mechanism explicitly;
- `--device auto|cuda|cuda:0|mps|cpu` and `--dtype auto|float16|bfloat16|float32`;
- `--prompts PATH` — use a custom corpus;
- `--quiet` — suppress per-prompt output.

The generated JSON contains run configuration, aggregate results, per-topic rates, and individual model responses. It may therefore contain harmful text and should not be attached to public issues or logs without review.

### Evaluator tradeoffs

| evaluator | processing boundary | dependencies | intended use |
|---|---|---|---|
| `keyword` | local | none | fast refusal-phrase baseline; easy to evade and language-sensitive |
| `semantic` | local after model download | `sentence-transformers`, NumPy | similarity-based local comparison; still heuristic |
| `llm-judge` | Anthropic API | `anthropic`, `instructor`, API key | model-based judging; sends prompt and response to a third party |
| `hybrid` | Anthropic API plus local Docker execution | judge dependencies and Docker | code-aware research runs in a separately isolated environment only |

Record the evaluator, corpus/package version, model revision, inference mode, and generation settings with every reported number.

## Python API

```python
from uncensorbench import UncensorBench, KeywordEvaluator

bench = UncensorBench()
evaluator = KeywordEvaluator()

for prompt in bench.prompts(topics=["controversial_speech"]):
    response = your_model(prompt.prompt)
    score = evaluator.score(prompt, response)
    print(prompt.id, score)
```

Core objects include `Prompt`, `GenerationConfig`, `EvaluationResult`, `BenchmarkResults`, `UncensorBench`, and the evaluator classes. `UncensorBench.evaluate(...)` accepts a callable that returns a model response and produces aggregate plus individual results.

## Custom corpus

Pass a JSON file with a top-level `prompts` array. Each item uses this shape:

```json
{
  "id": "policy_001",
  "topic": "policy",
  "subtopic": "refusal",
  "prompt": "...",
  "censored_response": "...",
  "uncensored_response": "..."
}
```

Use stable IDs and keep source, consent, and redistribution records for every prompt and reference response. Do not mix private evaluation data into a corpus you intend to release.

## Leaderboard

`uncensorbench.leaderboard.Leaderboard` reads or writes `leaderboard.csv` in the `wisent-ai/UncensorBench-Leaderboard` Hugging Face Space. Reading downloads remote data. Submission requires `huggingface_hub`, a token with write access, and explicit human review of the result and any `sample_responses_url` before upload.

The client does not establish result provenance, verify model identity, pin benchmark versions, redact responses, or provide an approval workflow. Public leaderboard entries are therefore unverified claims unless accompanied by reproducible artifacts.

## Scope and limitations

- The benchmark measures compliance/refusal patterns, not factuality, legality, harmlessness, usefulness, or overall model quality.
- Keyword and semantic scores are heuristics. LLM judges add provider dependence and judge-model variance.
- Stochastic runs are expected to vary.
- Corpus composition affects every aggregate score; compare like with like.
- The repository provides local tooling. It does not provide a hosted evaluation service, private result retention, verified submissions, organization policy, audit evidence, or paid entitlement handling.

## Responsible contribution

Open issues for benchmark bugs, evaluator methodology, corpus provenance, and reproducibility. Do not paste generated harmful content, credentials, private prompts, or unredacted result files into an issue. Security-sensitive reports should use GitHub's private security advisory flow.

## License

MIT; see [`LICENSE`](LICENSE). Prompt sources, model weights, model code, generated outputs, and optional services may carry separate terms. You are responsible for confirming rights and policy before training, evaluation, redistribution, or publication.
