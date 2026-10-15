"""Evaluation against known truth: the five metrics of §5.1 and the paper's tables.

The scoring convention matters and is the paper's own: an edge counts as true when
its **window-averaged** coefficient is nonzero, because by Theorem 1 the window
average of eq. (11) is what the estimates target — not the coefficient on the
output day alone.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._config import OracleVARXConfig
from ._model import OracleVARX
from ._results import OracleVARXResults

__all__ = [
    "window_target",
    "truth_at_day",
    "lag_rmse",
    "coefficient_mae",
    "forecast_mae",
    "edge_scores",
    "score_results",
    "benchmark_table",
    "regime_table",
]


def window_target(
    A_true: np.ndarray, days: np.ndarray, window: int, p_max: int
) -> np.ndarray:
    """The window-averaged truth each window estimate aims at (eq. 11).

    Eq. (11) weights the within-window coefficients by the confounder-adjusted
    second-moment matrices ``J_t``, which are unobservable in practice; those
    weights also make the drift cancel, so over a window short enough for
    assumption (W) the weighted and unweighted averages agree to ``O(T_w / T)``
    (Lemma A.3). This function uses the unweighted average, which is what the
    scoring needs and what the paper's own "window-averaged coefficient" means.

    Parameters
    ----------
    A_true : numpy.ndarray
        ``(T, p_true, n, n)`` the true coefficients at every date.
    days : numpy.ndarray
        ``(B,)`` output days.
    window : int
        ``T_w``.
    p_max : int
        Lag depth to pad or trim the truth to.

    Returns
    -------
    numpy.ndarray
        ``(B, p_max, n, n)``.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.zeros((10, 1, 2, 2)); A[:, 0, 0, 1] = np.linspace(1, 0, 10)
    >>> target = window_target(A, days=np.array([5]), window=5, p_max=1)
    >>> float(np.round(target[0, 0, 0, 1], 4))
    0.5556
    """
    T, p_true, n, _ = A_true.shape
    out = np.zeros((days.size, p_max, n, n))
    depth = min(p_true, p_max)
    for b, day in enumerate(days):
        lo = max(int(day) - window, 0)
        hi = max(int(day), lo + 1)
        out[b, :depth] = A_true[lo:hi, :depth].mean(axis=0)
    return out


def truth_at_day(A_true: np.ndarray, days: np.ndarray, p_max: int) -> np.ndarray:
    """The true coefficients on each output day itself, ``A*_k[i, j](t)``.

    This is the target the paper's coefficient-error metric names: metric 2 of
    §5.1 writes ``A*_k[i, j](t)``, the truth at time ``t``, while the
    window-averaged truth of :func:`window_target` is what its edge metric uses
    ("an edge is true if its window-averaged coefficient is nonzero"). Scoring
    coefficients against the window average instead moves the numbers: with the
    truth at the day, the paper's VAR and VARX coefficient errors reproduce
    exactly (0.0605 and 0.0627).

    Parameters
    ----------
    A_true : numpy.ndarray
        ``(T, p_true, n, n)``.
    days : numpy.ndarray
        ``(B,)`` output days.
    p_max : int
        Lag depth to pad to.

    Returns
    -------
    numpy.ndarray
        ``(B, p_max, n, n)``.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.zeros((10, 1, 2, 2)); A[:, 0, 0, 1] = np.arange(10)
    >>> float(truth_at_day(A, np.array([3]), p_max=2)[0, 0, 0, 1])
    3.0
    """
    T, p_true, n, _ = A_true.shape
    out = np.zeros((days.size, p_max, n, n))
    depth = min(p_true, p_max)
    out[:, :depth] = A_true[np.asarray(days)][:, :depth]
    return out


def lag_rmse(lag_hat: np.ndarray, p_true: np.ndarray) -> float:
    """Root mean squared error of the selected lag order (metric 1 of §5.1).

    Examples
    --------
    >>> import numpy as np
    >>> float(lag_rmse(np.array([1, 2, 3]), np.array([1, 2, 2])))
    0.5773502691896257
    """
    lag_hat = np.asarray(lag_hat, dtype=float)
    p_true = np.asarray(p_true, dtype=float)
    if lag_hat.shape != p_true.shape:
        raise ValueError(
            f"lag_hat has shape {lag_hat.shape} but p_true has {p_true.shape}"
        )
    return float(np.sqrt(np.mean((lag_hat - p_true) ** 2)))


def coefficient_mae(coef: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute error on the truly nonzero edges (metric 2 of §5.1).

    Examples
    --------
    >>> import numpy as np
    >>> coef = np.array([[[[0.5, 0.0], [0.0, 0.0]]]])
    >>> target = np.array([[[[0.6, 0.0], [0.0, 0.0]]]])
    >>> round(coefficient_mae(coef, target), 4)
    0.1
    """
    mask = np.abs(target) > 0
    if not mask.any():
        return float("nan")
    return float(np.abs(coef[mask] - target[mask]).mean())


def forecast_mae(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Out-of-sample mean absolute forecast error (metric 3 of §5.1).

    Examples
    --------
    >>> import numpy as np
    >>> round(forecast_mae(np.zeros((2, 2)), np.ones((2, 2))), 3)
    1.0
    """
    return float(np.abs(np.asarray(forecasts) - np.asarray(actuals)).mean())


def edge_scores(edges: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """False discovery rate, power and F1 over the discovered edges (metric 4).

    Both aggregation conventions are returned, because **the paper uses a
    different one for different columns** of its Table 2 and Table 4, and the
    difference is large enough to matter:

    * ``fdr`` is the mean of the per-window false discovery proportions, which
      is what Table 4's caption means by "averaged over the rolling evaluation
      window", and what reproduces its FDR column;
    * ``power`` and ``f1`` are pooled over all windows, which is what reproduces
      its Power and F1 columns.

    Reading all three the same way does not reproduce the paper: on its four
    non-DML rows, the per-window FDR lands within 0.002 of the published value
    while the pooled FDR sits 0.02-0.04 above it, and the published F1 is only
    consistent with the *pooled* precision. ``fdr_pooled``,
    ``power_per_window`` and ``f1_per_window`` give the other convention.

    Parameters
    ----------
    edges : numpy.ndarray
        ``(B, p_max, n, n)`` boolean, the BH selection.
    target : numpy.ndarray
        ``(B, p_max, n, n)`` the window-averaged truth; an entry is a true edge
        when it is nonzero.

    Returns
    -------
    dict
        ``fdr``, ``power``, ``f1``, ``precision``, ``n_discovered``, ``n_true``,
        plus ``fdr_pooled``, ``power_per_window`` and ``f1_per_window``.

    Examples
    --------
    >>> import numpy as np
    >>> edges = np.array([[[[True, False], [False, False]]]])
    >>> target = np.array([[[[0.6, 0.0], [0.0, 0.0]]]])
    >>> scores = edge_scores(edges, target)
    >>> scores["fdr"], scores["power"], scores["f1"]
    (0.0, 1.0, 1.0)
    >>> scores["fdr_pooled"]
    0.0
    """
    edges = np.asarray(edges, dtype=bool)
    truth = np.abs(np.asarray(target)) > 0
    if edges.shape != truth.shape:
        raise ValueError(
            f"edges has shape {edges.shape} but target has {truth.shape}"
        )
    discovered = int(edges.sum())
    true_edges = int(truth.sum())
    hits = int((edges & truth).sum())

    # Pooled over every window.
    fdr_pooled = (discovered - hits) / discovered if discovered else 0.0
    power_pooled = hits / true_edges if true_edges else float("nan")
    precision_pooled = hits / discovered if discovered else 0.0
    denom = precision_pooled + power_pooled
    f1_pooled = (
        2 * precision_pooled * power_pooled / denom
        if discovered and true_edges and denom > 0
        else 0.0
    )

    # Averaged over windows: the convention behind the paper's FDR column.
    axes = tuple(range(1, edges.ndim))
    n_disc = edges.sum(axis=axes)
    n_hit = (edges & truth).sum(axis=axes)
    n_true = truth.sum(axis=axes)
    safe_disc = np.maximum(n_disc, 1)
    fdr_w = np.where(n_disc > 0, (n_disc - n_hit) / safe_disc, 0.0)
    precision_w = np.where(n_disc > 0, n_hit / safe_disc, 0.0)
    power_w = np.where(n_true > 0, n_hit / np.maximum(n_true, 1), np.nan)
    with np.errstate(invalid="ignore"):
        total_w = precision_w + power_w
        f1_w = np.where(total_w > 0, 2 * precision_w * power_w / np.maximum(total_w, 1e-12), 0.0)

    return {
        "fdr": float(np.mean(fdr_w)),
        "power": float(power_pooled),
        "f1": float(f1_pooled),
        "precision": float(precision_pooled),
        "fdr_pooled": float(fdr_pooled),
        "power_per_window": float(np.nanmean(power_w)),
        "f1_per_window": float(np.nanmean(f1_w)),
        "n_discovered": discovered,
        "n_true": true_edges,
    }


def score_results(
    results: OracleVARXResults,
    A_true: np.ndarray,
    p_true: np.ndarray,
) -> dict[str, float]:
    """Score one fit on all five metrics of §5.1.

    Parameters
    ----------
    results : OracleVARXResults
    A_true : numpy.ndarray
        ``(T, p_true, n, n)`` from :func:`oracle_varx.datasets.make_oracle_dgp`.
    p_true : numpy.ndarray
        ``(T,)`` the true lag order at every date.

    Returns
    -------
    dict
        ``coef_mae``, ``fcst_mae``, ``lag_rmse``, ``fdr``, ``power``, ``f1`` in the
        column order of the paper's Table 4, plus ``coef_mae_window`` and the raw
        counts.

    Notes
    -----
    Each metric is scored the way the paper scores it, and the conventions differ:
    the coefficient error is measured against the truth **on the output day**
    (metric 2 writes ``A*_k[i,j](t)``), while edges are judged against the
    **window-averaged** truth, because that is what Theorem 1 says the estimates
    target. The matching coefficient numbers for the ACLE rungs additionally need
    ``coef_source="p_hat"``, since the paper reads those coefficients off the fit
    at the selected lag order.
    """
    days = results.days
    target = window_target(
        A_true, days, results.config.ols_window, results.p_max
    )
    at_day = truth_at_day(A_true, days, results.p_max)
    scores = {
        "coef_mae": coefficient_mae(results.coef, at_day),
        "coef_mae_window": coefficient_mae(results.coef, target),
        "fcst_mae": forecast_mae(results.forecasts, results.actuals),
        "lag_rmse": lag_rmse(results.lag, np.asarray(p_true)[days]),
    }
    scores.update(edge_scores(results.edges, target))
    return scores


def benchmark_table(
    Y,
    W,
    A_true: np.ndarray,
    p_true: np.ndarray,
    methods: Sequence[str] = (
        "var", "acle-var", "varx", "acle-varx", "or-varx", "oracle-varx",
    ),
    learners: Sequence[str] = ("linear",),
    config: OracleVARXConfig | None = None,
    levels: Mapping[str, Any] | None = None,
    seed: int | None = 42,
    n_jobs: int = 1,
    verbose: bool = False,
) -> pd.DataFrame:
    """Reproduce the paper's Table 2 and Table 4 on a known-truth benchmark.

    Fits every requested rung of Table 1, at every requested observability level
    and first-stage learner, and scores them on the five metrics of §5.1 in the
    paper's own column order.

    Parameters
    ----------
    Y, W : pandas.DataFrame or array-like
        The benchmark data and its full confounder set.
    A_true, p_true : numpy.ndarray
        The known truth.
    methods : sequence of str
        Rungs to evaluate.
    learners : sequence of str
        First-stage learners, used only for the DML rungs. ``"linear"`` is fast
        and deterministic; the paper's five are ``"extra_trees"``,
        ``"random_forest"``, ``"lightgbm"``, ``"xgboost"`` and ``"tabpfn"``.
    config : OracleVARXConfig, optional
        Defaults to the paper's synthetic configuration.
    levels : mapping, optional
        Observability levels to run, as ``{"all": W, "partial_2": W.iloc[:, :2]}``.
        Defaults to ``{"all": W}``; the non-confounded rungs always run at
        ``"none"``, as the paper reports them.
    seed : int or None, default 42
    n_jobs : int, default 1
    verbose : bool, default False

    Returns
    -------
    pandas.DataFrame
        One row per (method, learner, level), columns ``coef_mae``, ``fcst_mae``,
        ``lag_rmse``, ``fdr``, ``power``, ``f1``.

    Notes
    -----
    The paper's own numbers are not bit-reproducible: its tree learners are
    unseeded, so Extra Trees and Random Forest give different results run to run.
    Expect agreement within Monte-Carlo error, not to the last digit.
    """
    config = config or OracleVARXConfig.synthetic()
    levels = dict(levels) if levels is not None else {"all": W}
    rows = []

    for method in methods:
        needs_w = method not in ("var", "acle-var")
        is_dml = method in ("or-varx", "oracle-varx")
        method_learners = learners if is_dml else ("--",)
        method_levels = levels if needs_w else {"none": None}

        for learner in method_learners:
            for level, W_level in method_levels.items():
                if verbose:
                    print(f"fitting {method} / {learner} / {level} ...", flush=True)
                model = OracleVARX(
                    config=config,
                    method=method,
                    learner=learner if is_dml else "linear",
                    seed=seed,
                    n_jobs=n_jobs,
                )
                res = model.fit(Y, W_level)
                scores = score_results(res, A_true, p_true)
                rows.append(
                    {
                        "method": method,
                        "learner": learner,
                        "obs": level,
                        "coef_mae": scores["coef_mae"],
                        "fcst_mae": scores["fcst_mae"],
                        "lag_rmse": scores["lag_rmse"],
                        "fdr": scores["fdr"],
                        "power": scores["power"],
                        "f1": scores["f1"],
                    }
                )
    return pd.DataFrame(rows).set_index(["method", "learner", "obs"])


def regime_table(
    results: OracleVARXResults,
    A_true: np.ndarray,
    p_true: np.ndarray,
    regimes: Sequence[tuple[int, int]],
) -> pd.DataFrame:
    """Score a fit separately inside each regime — an addition to the paper.

    The benchmark has three regimes, in which the true lag order is 3, then 2,
    then 1, but the paper only ever reports metrics pooled across all of them. A
    per-regime breakdown shows where the lag tracking and the edge discovery
    actually succeed or fail.

    Parameters
    ----------
    results : OracleVARXResults
    A_true, p_true : numpy.ndarray
    regimes : sequence of (int, int)
        Half-open date ranges, as :class:`~oracle_varx.datasets.ToyGroundTruth`
        supplies them.

    Returns
    -------
    pandas.DataFrame
        One row per regime.
    """
    days = results.days
    target = window_target(A_true, days, results.config.ols_window, results.p_max)
    truth_lag = np.asarray(p_true)[days]
    rows = []
    for start, end in regimes:
        mask = (days >= start) & (days < end)
        if not mask.any():
            continue
        scores = edge_scores(results.edges[mask], target[mask])
        rows.append(
            {
                "regime": f"[{start}, {end})",
                "n_days": int(mask.sum()),
                "p_true": float(truth_lag[mask].mean()),
                "p_hat": float(results.lag[mask].mean()),
                "lag_rmse": lag_rmse(results.lag[mask], truth_lag[mask]),
                "coef_mae": coefficient_mae(results.coef[mask], target[mask]),
                "fcst_mae": forecast_mae(
                    results.forecasts[mask], results.actuals[mask]
                ),
                "fdr": scores["fdr"],
                "power": scores["power"],
                "f1": scores["f1"],
            }
        )
    return pd.DataFrame(rows).set_index("regime")
