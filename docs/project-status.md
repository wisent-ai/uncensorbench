## Project status and support

| Property | Current contract |
|---|---|
| Maturity | Alpha (`Development Status :: 3 - Alpha`, `pyproject.toml`) |
| Latest supported release | `0.3.7` on PyPI. No GitHub releases or tags are published |
| Compatibility | Python `>=3.8`. No compatibility policy or deprecation window is published. `main` is ahead of 0.3.7 in ways that are visible to users: corpus `2.0.0` with 200 prompts across 16 topics versus `0.1.0` with 150 across 15; the `hybrid` evaluator, `CodeExecutionEvaluator`, `CoherenceEvaluator`, `CombinedEvaluator`, `GenerationConfig`, and `InferenceMode` exist only on `main`; the CLI's `--inference-mode` and `--prompts` replace 0.3.7's `--temperature` and `--max-tokens`; the CLI default evaluator is `semantic` in 0.3.7 and `hybrid` on `main`; and `UncensorBench.evaluate` changed signature. Install from source to get `main` |
| Distribution | PyPI package [`uncensorbench`](https://pypi.org/project/uncensorbench/); source at [github.com/wisent-ai/uncensorbench](https://github.com/wisent-ai/uncensorbench) |
| Commercial status | `conditional` (`commercial-status.json`). Uncontrolled hosted evaluation and private-leaderboard-only results are prohibited; re-entry is gated on dual-use controls, access controls, independent result export, and a retention policy |
| License | [MIT](LICENSE). Prompt sources, model weights, model code, generated outputs, and optional third-party services carry their own terms. Confirming your rights before training, evaluation, redistribution, or publication is your responsibility |

- **Use and design questions:** [Wisent Discord](https://discord.gg/qRjpkthq54)
- **Reproducible defects:** [GitHub issues](https://github.com/wisent-ai/uncensorbench/issues). Include the package version, the corpus version, the evaluator, the inference mode, the model and revision, and the exact command
- **Security reports:** use GitHub's private security advisory flow on this repository, or contact@wisent.ai. Never a public issue
- **Contributions:** open a pull request; open an issue first for anything that changes the corpus, an evaluator's scoring rule, or the public surface. The repository publishes no contribution guide
- **Releases:** no changelog is published. The declared version is in `pyproject.toml` and the published public surface is frozen in `released-surface.json`

**In every one of these channels:** never paste generated harmful content, credentials, private prompts, or unredacted result files. Attach a redacted excerpt, or describe the failure.