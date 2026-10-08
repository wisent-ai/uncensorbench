## Project status and support

| Property | Current contract |
|---|---|
| Maturity | Alpha |
| Latest supported release | The Python package last published on PyPI is superseded by the Rust command line in this repository (`Cargo.toml`); no Rust release is tagged yet |
| Compatibility | No compatibility policy or deprecation window is published. The Rust command line replaces the Python CLI and API: generation and judging go through Brama on routes the caller names, the keyword, semantic, likelihood, coherence and code-execution evaluators are gone, and leaderboard writes take the Hub token on standard input |
| Distribution | Source at [github.com/wisent-ai/uncensorbench](https://github.com/wisent-ai/uncensorbench), installed with `cargo install --git` |
| Commercial status | `conditional` (`commercial-status.json`). Uncontrolled hosted evaluation and private-leaderboard-only results are prohibited; re-entry is gated on dual-use controls, access controls, independent result export, and a retention policy |
| License | [MIT](LICENSE). Prompt sources, model weights, model code, generated outputs, and optional third-party services carry their own terms. Confirming your rights before training, evaluation, redistribution, or publication is your responsibility |

- **Use and design questions:** [Wisent Discord](https://discord.gg/qRjpkthq54)
- **Reproducible defects:** [GitHub issues](https://github.com/wisent-ai/uncensorbench/issues). Include the commit, the corpus version, the route and judge route, the settings stated, and the exact command
- **Security reports:** use GitHub's private security advisory flow on this repository, or contact@wisent.ai. Never a public issue
- **Contributions:** open a pull request; open an issue first for anything that changes the corpus, the judge's criteria, or the public surface. The repository publishes no contribution guide
- **Releases:** no changelog is published. The declared version is in `Cargo.toml`; the published public surface is read from the newest released tag by `stado release version-gate app-baseline`

**In every one of these channels:** never paste generated harmful content, credentials, private prompts, or unredacted result files. Attach a redacted excerpt, or describe the failure.