# The method, and how this package implements it

Equation numbers refer to Tan, Zanoria, Chen, Lyu and Cucuringu (2026), *Causal
Lag Structure Discovery in Confounded Time Series via Orthogonalized Adaptive
Estimation*, [doi:10.48550/arXiv.2610.05618](https://doi.org/10.48550/arXiv.2610.05618).

This document is the audit trail: every numbered equation, assumption and
algorithm step, and the function that implements it.

---

## 1. The model

With `n` endogenous series and `d` exogenous confounders, the linear VARX(p) of
eq. (1) is

$$ Y_t = \sum_{k=1}^{p} A_k Y_{t-k} + \sum_{k=1}^{p} B_k W_{t-k} + \varepsilon_t. $$

The paper generalises the confounder part and leaves the lag part linear, giving
the **partially linear VARX** of eq. (2):

$$ Y_t = \sum_{k=1}^{p^*} A_k Y_{t-k} + g(W_{t-1}, \dots, W_{t-p}) + \varepsilon_t,
\qquad \mathbb{E}[\varepsilon_t \mid Y_{t-1}, \dots, W_{t-p}] = 0, $$

with `g` unknown and possibly nonlinear. The lag effects stay interpretable and
testable; the confounder effect is unrestricted. Setting `g` linear recovers
eq. (1).

**Definition 1 (causal lag graph).** The graph has vertex set
$V = \{Y^{(1)}, \dots, Y^{(n)}\}$ and lag-annotated directed edges
$E = \{(Y^{(j)} \to Y^{(i)}, k) : A_k[i,j] \neq 0\}$. The goal is to recover `E`
with valid inference on each edge, under minimal assumptions on `g`.

In this package `A_k[i, j]` is `coef[window, k-1, i, j]`: the effect of series `j`
at lag `k` on series `i`. `CausalLagGraph` holds one window's `E`.

## 2. Stage one: orthogonalisation (§3.2)

Stack the lags and the confounders, eq. (3):

$$ T_t = \begin{pmatrix} Y_{t-1} \\ \vdots \\ Y_{t-p} \end{pmatrix} \in \mathbb{R}^{np},
\qquad
C_t = \begin{pmatrix} W_{t-1} \\ \vdots \\ W_{t-p} \end{pmatrix} \in \mathbb{R}^{dp}, $$

and let `Θ` stack $A_1^\top, \dots, A_p^\top$, so that
$\Theta^\top T_t = \sum_k A_k Y_{t-k}$. Eq. (2) becomes the partially linear
model of eq. (4)–(5):

$$ Y_t = \Theta^\top T_t + g(C_t) + U_t, \qquad T_t = g_T(C_t) + V_t. $$

Three steps follow.

1. **Cross-fitted nuisance estimation.** A flexible learner estimates
   $\hat g_Y$ and $\hat g_T$ with time-series cross-fitting.
2. **Residualisation**, eq. (6):
   $\tilde Y_t = Y_t - \hat g_Y(C_t)$, $\tilde T_t = T_t - \hat g_T(C_t)$.
3. **Orthogonalised OLS**, eq. (7):

$$ \hat\Theta = \Big(\sum_t \tilde T_t \tilde T_t^\top\Big)^{-1}
               \Big(\sum_t \tilde T_t \tilde Y_t^\top\Big). $$

The estimator is Neyman orthogonal, so first-stage error enters `Θ̂` only at
second order — which is exactly what permits machine learning in the first stage
without forfeiting inference.

**The split.** Standard k-fold cross-validation breaks temporal order. The paper
uses a blocked rolling split: fold `k` trains on `[s_k, s_k + n_train)` and
contributes out-of-sample residuals for the next `n_test` rows, with
`s_{k+1} = s_k + n_test`, so the folds tile without overlapping test ranges.

**The memoisation.** Because the blocks lie on one grid of rows that does not
depend on the rolling window, each block is fitted once and reused by every window
that overlaps it. The paper measures this at 203×–466× fewer first-stage fits than
refitting per window, and the reuse is *exact*.

**Standard errors**, eq. (8):

$$ \widehat{\operatorname{Var}}(\hat\Theta_{\cdot i})
   = \hat\sigma_i^2 \Big(\sum_t \tilde T_t \tilde T_t^\top\Big)^{-1},
\qquad
\hat\sigma_i^2 = \frac{1}{T_{\text{eff}} - np}
   \sum_t \big(\tilde Y_{t,i} - \hat\Theta_{\cdot i}^\top \tilde T_t\big)^2. $$

No HAC correction: Theorem 1 shows the martingale central limit theorem leaves no
autocovariance terms under conditional homoskedasticity, assumption (H).

**Remark 1 (buffer zones).** Time-series DML theory often inserts a buffer of
`k_T` observations between training and test folds. The paper uses adjacent folds,
`k_T = 0`, on the ground that the nuisance functions themselves drift, so a buffer
makes every fit staler and costs observations. The `buffer` argument exposes the
alternative.

## 3. Stage two: adaptive causal lag estimation (§3.3)

Fixing `p` in advance is the problem: too low misses genuine edges, too high
inflates the parameter count, and the true order may shift over time.

**The non-adaptive rule**, eq. (9): for each candidate `p`, score rolling
forecasts over the trailing `v` days and take the minimiser,

$$ \mathrm{RMSE}_v(p; t) = \sqrt{\frac{1}{v}\sum_{s=t-v}^{t-1}\big\|Y_s - \hat Y_s^{(p)}\big\|^2},
\qquad \hat p_t = \arg\min_p \mathrm{RMSE}_v(p; t). $$

**ACLE, Stage 1.** Lag 1 is always kept. For `p = 2, …, p_max`, test
$H_0: A_p[i,j] = 0$ for all `(i, j)` simultaneously: entry-wise z-statistics
$z_{i,j} = \hat A_p[i,j] / \hat s(\hat A_p[i,j])$ are taken **from the `p_max`
fit**, and a BH correction at level `α` decides whether any entry of `A_p` is
significant. The first `p` at which nothing is significant stops the search, giving
`p_α = p − 1`.

**ACLE, Stage 2**, eq. (10): the level itself is chosen by the same rolling
validation criterion,

$$ \alpha^*_t = \arg\min_{\alpha \in \{\alpha_1,\dots,\alpha_M\}} \mathrm{RMSE}_v(p_\alpha; t),
\qquad \hat p_t = p_{\alpha^*_t}. $$

So the data sets the sensitivity of lag detection, rather than the analyst.

## 4. Stage three: FDR-controlled edge discovery (§3.4, D.4)

From the `p_max` fit, form $z_{k,i,j} = \hat A_k[i,j]/\hat s(\hat A_k[i,j])$ for
all `k ≤ p̂_t`, take two-sided p-values $p_{k,i,j} = 2(1 - \Phi(|z_{k,i,j}|))$,
and apply Benjamini–Hochberg at level `q` to all $m = n^2 \hat p_t$ tests:

$$ r^* = \max\Big\{ r : p_{(r)} \le \tfrac{r}{m} q \Big\}. $$

The edges with the `r*` smallest p-values are kept; if no `r` satisfies the
inequality, no edge is kept. Entries at lags deeper than `p̂_t` are never edges.

## 5. The method hierarchy (Table 1)

| Method | Lag selection | Confounder adjustment | DML |
|---|---|---|---|
| VAR | RMSE | none | – |
| VARX | RMSE | linear | – |
| ACLE-VAR | ACLE | none | – |
| ACLE-VARX | ACLE | linear | – |
| OR-VARX | RMSE | nonlinear | ✓ |
| ORACLE-VARX | ACLE | nonlinear | ✓ |

Each row adds one capability to the row above, which is what lets a result be
attributed to the component that produced it. `method=` walks the table.

## 6. Theory

Coefficients drift, so the analysis is of a single rolling window `W` with
$\Theta_t = \Theta(t/T)$. What a window estimates is the weighted average of
eq. (11),

$$ \bar\Theta = \Big(\sum_{t \in W} J_t\Big)^{-1} \sum_{t \in W} J_t \Theta_t,
\qquad J_t = \mathbb{E}[V_t V_t^\top], $$

whose weights make the drift cancel exactly, since
$\sum_{t\in W} J_t(\Theta_t - \bar\Theta) = 0$, and which lies within
$O(T_w/T)$ of $\Theta(u_0)$.

**Assumption 1.** (M) the innovations are a martingale difference sequence given
the observed past — all confounders observed and `p_max ≥ p*`; (S) `Θ` is
Lipschitz and $\lambda_{\min}(J_t) \ge c_J > 0$; (W) $T_w \to \infty$ with
$T_w = O(T^{2/3})$; (R) the regularity conditions (D), (H), (N) — bounded moments
and short memory, conditional homoskedasticity, and the usual DML product rate on
the cross-fitted nuisance error.

**Theorem 1 (local asymptotic normality).** Under Assumption 1, for every entry,

$$ \frac{\hat\Theta_{ri} - \bar\Theta_{ri}}{\hat s_{ri}} \xrightarrow{d} N(0,1), $$

and if moreover $T_w = o(T^{2/3})$, then $\bar\Theta_{ri}$ may be replaced by
$\Theta_{ri}(u_0)$. This is what `conf_int` and `pvalues` rest on, and what
`admissible_window` checks: whether the window is small enough for the estimate to
be read as the coefficient *on the day* rather than only as the window average.

**Proposition A.13 (Granger non-causality).** Under causal sufficiency — `W_t`
contains every common cause — $A_k[i,j] = 0$ if and only if $Y^{(j)}$ does not
Granger-cause $Y^{(i)}$ at lag `k`, given a non-degeneracy condition. With a
hidden confounder this breaks: a hidden common cause creates a spurious edge, a
hidden mediator makes an indirect link look direct, and the coefficients become
the best linear lead–lag predictor from the observed variables rather than causal
effects.

**What is not proved.** ACLE is a heuristic. Theorem 1 explains why it tracks the
lag order — above the true order the z-statistics stay of order one, while a true
lag has an entry whose `|z|` grows like $\sqrt{T_w}$ — but a formal consistency
result covering the BH step and the validation-chosen level is left open by the
paper.

## 7. Step → equation map

| Implementation | Paper | What it computes |
|---|---|---|
| `_utils.build_stacks` | eq. (3) | `T_t`, `C_t`, shallowest lag first |
| `_utils.design_with_lags` | eq. (1) | the classical VAR design, via `statsmodels.lagmat` |
| `_utils.blocked_folds` | §3.2 | `s_{k+1} = s_k + n_test`, non-overlapping test ranges |
| `_nuisance.make_learner` | Table 10 B | the five first-stage learners and their hyperparameters |
| `_nuisance.cross_fit` | eq. (6) | out-of-sample residualisation, one fit per block |
| `_nuisance.NuisanceFit` | §3.2, D.3 | the memoised residuals and nuisance predictions |
| `_core.second_stage` | eq. (7)–(8) | `Θ̂` and `se(Θ̂)`, batched over windows |
| `_core.coefficients_from_theta` | Definition 1 | reshape to `A_k[i, j]` |
| `_core._ols_windows` | eq. (1) | the non-DML rungs, rolling, with an intercept |
| `_core._dml_windows` | Alg. 1 line 7 | `ĝ_Y(C_t) + Θ̂ᵀ(T_t − ĝ_T(C_t))` |
| `_core.rmse_lag_selection` | eq. (9) | rolling validation RMSE over `1…p_max` |
| `_core.acle_stage1` | §3.3 Stage 1 | sequential BH tests on the `p_max` fit |
| `_core.rolling_validation` | eq. (10) | the validation-chosen level, per day |
| `_core.bh_edges` | §3.4, D.4 | BH over `n² p̂` tests, with q-values |
| `_core.fit_windows` | Alg. 1, 1–3 | the whole per-window pipeline |
| `_core.cost_report` | D.3 | `S = n p_max(p_max+3)/2`, `F ≈ (T−n_train−p_max)/n_test` |
| `_model.OracleVARX.fit` | Alg. 1 | the pipeline end to end |
| `_results.conf_int` | Theorem 1 | `coef ± z_{1−α/2}·se` |
| `_utils.admissible_window` | assumption (W) | whether to read `Θ(u_0)` or `Θ̄` |
| `evaluation.window_target` | eq. (11) | the window-averaged truth |
| `evaluation.truth_at_day` | metric 2 | `A*_k[i,j](t)`, the truth on the day |
| `evaluation.edge_scores` | metric 4 | FDR, power, F1, in both conventions |
| `datasets.make_oracle_dgp` | §5.1, eq. (12) | the benchmark, with known truth |
| `portfolio.weights` | App. C.1 | naive, weighted, top-q, unit gross exposure |

## 8. Equivalences and departures

**Algebraically equivalent, cheaper.** The authors solve each rolling window
separately on the GPU. This package stacks every window into one array with
`numpy.lib.stride_tricks.sliding_window_view` and solves them in a single
`numpy.linalg.solve` call. Verified equal to a per-window `lstsq` loop to
`1.2e-15`, so PyTorch is not a dependency.

**Delegated rather than rewritten.** `statsmodels.tsa.tsatools.lagmat` for eq. (3)'s
stacking; `statsmodels.stats.multitest.multipletests` for the BH step (verified
identical to the paper's own statement of the rule);
`statsmodels.tsa.api.VAR` as the correctness reference for the non-DML rungs
(coefficients identical, standard errors to `2.8e-17`);
`sklearn.model_selection.TimeSeriesSplit` as the cross-check on the fold
boundaries, which agree exactly whenever the folds tile the sample.

**Deliberate departures, each documented at its call site.**

| Departure | Why |
|---|---|
| `ν = 0.1` in the DGP | the paper's stated value, and the value the distributed benchmark data was generated with; the authors' code defaults to 0.05 |
| every learner seeded | the authors set no random state, so their tree results are not reproducible run to run |
| exactly `ols_window` rows per window for every candidate `p` | the authors give each `p` a fixed lookback, so a shallow lag order silently gets extra rows; the paper describes a fixed window |
| `ridge = 0` by default | the paper is plain OLS; the authors add `1e-6 I` in their VAR path only, which perturbs the standard errors and hence the z-tests. Available as `ridge=1e-6` |
| no intercept in the second stage | eq. (7) is on residuals. The non-DML rungs do carry one, matching the code that produced the published tables |

## 9. Reproduction record

See the table in [the README](../README.md#13-faithfulness-to-the-paper) and
[`tests/test_reference.py`](../tests/test_reference.py). In summary: the DGP
reproduces the distributed benchmark data to `9.6e-16`; the four deterministic
rows of Table 4 reproduce within 0.002 on every metric; the first-stage fit count
matches Appendix D.3 exactly.

Three scoring conventions had to be recovered from the published numbers, and all
three are documented in `evaluation`: FDR is averaged per window while power and
F1 are pooled; the coefficient error is measured against the truth on the output
day while edges are judged against the window average; and the ACLE coefficient
error is read off the selected-lag fit rather than the `p_max` fit.
