# Changelog

All notable changes to `oracle-varx` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
semantic versioning.

## [0.1.0] - 2026-10-15

Published on PyPI: <https://pypi.org/project/oracle-varx/0.1.0/>
Install with `pip install oracle-varx`.

First release. Implements Tan, Zanoria, Chen, Lyu and Cucuringu (2026),
*Causal Lag Structure Discovery in Confounded Time Series via Orthogonalized
Adaptive Estimation*, `doi:10.48550/arXiv.2610.05618`, reimplemented from the
paper's equations.

### Added

- `OracleVARX` and the one-call `oracle_varx()` wrapper, covering all six rungs of
  the paper's Table 1: `var`, `varx`, `acle-var`, `acle-varx`, `or-varx` and
  `oracle-varx`.
- Orthogonalised second stage with the standard errors of eq. (8), solved for
  every rolling window in one batched call.
- Blocked, memoised cross-fitting: one first-stage fit per (block, lag order,
  target), reused by every window that overlaps the block, giving the fit counts
  of Appendix D.3 exactly.
- Adaptive causal lag estimation: sequential significance testing over a grid of
  levels (Stage 1) with rolling-validation level selection (Stage 2).
- FDR-controlled edge discovery with BH-adjusted q-values per edge.
- Five first-stage learners — Extra Trees, Random Forest, LightGBM, XGBoost and
  TabPFN — all with the paper's Table 10 hyperparameters, plus a fast `linear`
  option and support for any scikit-learn-compatible estimator.
- `make_oracle_dgp`, reproducing the §5.1 benchmark; it regenerates the data
  distributed with the paper to floating-point precision.
- `load_treasury`, a real shipped dataset: daily US Treasury yield-curve changes
  with the real (TIPS) curve as confounders, public domain.
- `fetch_sector_etfs`, `fetch_industry_portfolios`, `fetch_treasury_curve` and
  `fetch_fred` for data that cannot be redistributed.
- `evaluation`: the five metrics of §5.1, `benchmark_table` for Tables 2 and 4,
  and a per-regime breakdown.
- `portfolio`: the three position rules, the dollar hedge and the performance
  table of Appendix C.
- `diagnostics`: assumption checks for (M), (S), (W) and (R), a coefficient
  constancy test, a lag-rule comparison against AIC/BIC/HQ, and a
  leave-one-confounder-out sensitivity probe.
- Journal tables (`summary`, `to_latex`, `to_markdown`) and figures reproducing
  the paper's Figures 3, 4 and 5, plus edge heatmaps, a causal-graph drawing, a 3-D
  surface, FDR/power curves, residual diagnostics and an assumption-(W) guard.
- The Parula colour family, ported from the author's R packages, as the default
  scale on every heatmap, contour and surface.

### Additions beyond the paper

- Benjamini-Yekutieli correction (`correction="fdr_by"`), valid under arbitrary
  dependence between the entry-wise tests, which the paper's §4 concedes it does
  not establish for BH.
- A confidence interval and BH-adjusted q-value for every edge.
- `coef_source="p_hat"`, reporting each day's coefficients from the fit at its own
  selected lag order.
- An admissible-window guard for assumption (W), which decides whether an estimate
  may be read as the coefficient on the day or only as the window average.
- Edge-persistence summaries, per-regime scoring, and the diagnostics listed above.

### Notes on faithfulness

- The synthetic benchmark reproduces the paper's deterministic rows of Table 2 and
  Table 4 within 0.002 on every metric; see `tests/test_reference.py`.
- Two scoring conventions in the paper differ by column, and both are provided:
  its FDR is a per-window average while its power and F1 are pooled, and its
  coefficient error is measured against the truth on the output day rather than
  the window average.
- The paper's own numbers are not bit-reproducible: its tree learners carry no
  random state, so its Extra Trees and Random Forest rows differ run to run. This
  package seeds every learner.
- No code from the authors' repository is used; it carries no licence.

[0.1.0]: https://github.com/merwanroudane/oracle-varx/releases/tag/v0.1.0
