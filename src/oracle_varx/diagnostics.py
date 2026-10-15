"""Diagnostics: the assumption checks and sensitivity probes the paper implies.

Everything here is an **addition** to Tan et al. (2026). The paper states its
assumptions and proves Theorem 1 under them, but prints no check of any of them;
these functions turn each assumption into something a user can test on their own
data. Established tests are delegated: Ljung-Box and the lag-order information
criteria come from ``statsmodels``.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2

from ._config import OracleVARXConfig
from ._utils import admissible_window, as_matrix

__all__ = [
    "check_assumptions",
    "constancy_test",
    "lag_rule_comparison",
    "leave_one_confounder_out",
]


def check_assumptions(results, Y=None, lags: int = 20) -> pd.DataFrame:
    """Check the four parts of Assumption 1 on a fitted model.

    ======  ==========================================================
    (M)     the innovations are a martingale difference sequence, so
            their autocorrelation should be negligible — Ljung-Box on
            the forecast errors
    (S)     the coefficients drift slowly — the largest change in a
            coefficient between adjacent windows, scaled by the window
    (W)     ``T_w = O(T^(2/3))``, which decides whether the estimate is
            read as ``Theta(u_0)`` or only as ``Theta_bar``
    (R)     the confounder-adjusted lags are not collinear — the
            smallest relative standard error across the grid stands in
            for ``lambda_min(J_t)``
    ======  ==========================================================

    Parameters
    ----------
    results : OracleVARXResults
    Y : array-like, optional
        The series the model was fitted to, used only to report ``T``.
    lags : int, default 20
        Lags for the Ljung-Box test.

    Returns
    -------
    pandas.DataFrame
        One row per assumption, with a statistic, a verdict and a note.

    Notes
    -----
    These are diagnostics, not proofs: Assumption 1 is asymptotic, and a passing
    row means nothing in the data contradicts it at this sample size.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    rows: list[dict[str, Any]] = []

    errors = (results.actuals - results.forecasts).mean(axis=1)
    test = acorr_ljungbox(errors, lags=[min(lags, max(len(errors) // 5, 1))])
    pvalue = float(test["lb_pvalue"].iloc[0])
    rows.append(
        {
            "assumption": "(M) martingale innovations",
            "statistic": float(test["lb_stat"].iloc[0]),
            "pvalue": pvalue,
            "verdict": "ok" if pvalue > 0.05 else "questionable",
            "note": (
                "Ljung-Box on the forecast errors; a small p-value says the "
                "errors are still predictable, so raise p_max or revisit the "
                "confounder set"
            ),
        }
    )

    drift = np.abs(np.diff(results.coef, axis=0)).max() if results.n_windows > 1 else 0.0
    scaled = drift * results.config.ols_window
    rows.append(
        {
            "assumption": "(S) slowly drifting coefficients",
            "statistic": float(drift),
            "pvalue": np.nan,
            "verdict": "ok" if scaled < 1.0 else "questionable",
            "note": (
                f"largest one-day change in any coefficient is {drift:.4f}; times "
                f"the window that is {scaled:.2f}, and Lipschitz drift wants this "
                f"well below 1"
            ),
        }
    )

    n_obs = len(Y) if Y is not None else int(results.days[-1] + 1)
    guard = admissible_window(n_obs, results.config.ols_window)
    rows.append(
        {
            "assumption": "(W) window scale",
            "statistic": results.config.ols_window / guard["scale"],
            "pvalue": np.nan,
            "verdict": "ok" if guard["target"] == "Theta(u_0)" else "window average only",
            "note": str(guard["message"]),
        }
    )

    scale = np.abs(results.coef).max() or 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = results.se / scale
    worst = float(np.nanmax(relative))
    rows.append(
        {
            "assumption": "(R) no collinearity in the adjusted lags",
            "statistic": worst,
            "pvalue": np.nan,
            "verdict": "ok" if worst < 1.0 else "questionable",
            "note": (
                "largest standard error relative to the largest coefficient; a "
                "value above 1 means some lag block is barely identified once the "
                "confounders are partialled out"
            ),
        }
    )
    return pd.DataFrame(rows).set_index("assumption")


def constancy_test(results, blocks: int = 4) -> pd.DataFrame:
    """Wald test that the window target is constant across sub-periods.

    The premise of the paper is that the coefficients drift, yet it never tests
    the drift. Splitting the output days into non-overlapping blocks and testing
    equality of the block means of each coefficient against its standard error
    gives a direct read on how much structure actually changes.

    Parameters
    ----------
    results : OracleVARXResults
    blocks : int, default 4
        Number of equal, non-overlapping sub-periods.

    Returns
    -------
    pandas.DataFrame
        One row per ``(lag, from, to)``, with the Wald statistic, its degrees of
        freedom, the p-value and the block means. Sorted by the statistic.

    Raises
    ------
    ValueError
        If there are too few windows for the requested number of blocks.

    Notes
    -----
    Adjacent windows overlap by ``T_w - 1`` observations, so the block means are
    strongly dependent and this is a descriptive diagnostic, not an exact test.
    The blocks are taken far enough apart that the statistic is informative about
    *which* edges move, which is what it is for.
    """
    if blocks < 2:
        raise ValueError(f"blocks must be >= 2, got {blocks}")
    if results.n_windows < blocks * 2:
        raise ValueError(
            f"{results.n_windows} windows are too few for {blocks} blocks"
        )

    edges = np.array_split(np.arange(results.n_windows), blocks)
    rows = []
    for k in range(results.p_max):
        for i in range(results.n_series):
            for j in range(results.n_series):
                means = np.array(
                    [results.coef[idx, k, i, j].mean() for idx in edges]
                )
                variances = np.array(
                    [
                        (results.se[idx, k, i, j] ** 2).mean() / max(len(idx), 1)
                        for idx in edges
                    ]
                )
                if not np.isfinite(variances).all() or (variances <= 0).any():
                    continue
                weights = 1.0 / variances
                pooled = float((weights * means).sum() / weights.sum())
                statistic = float((weights * (means - pooled) ** 2).sum())
                dof = blocks - 1
                rows.append(
                    {
                        "lag": k + 1,
                        "from": results.names[j],
                        "to": results.names[i],
                        "wald": statistic,
                        "df": dof,
                        "pvalue": float(chi2.sf(statistic, dof)),
                        "range": float(means.max() - means.min()),
                    }
                )
    frame = pd.DataFrame(rows)
    return frame.sort_values("wald", ascending=False).reset_index(drop=True)


def lag_rule_comparison(
    Y,
    W=None,
    config: OracleVARXConfig | None = None,
    learner: str | Any = "linear",
    seed: int | None = 42,
) -> pd.DataFrame:
    """Compare ACLE with rolling RMSE and the classical information criteria.

    The paper compares ACLE only with its own rolling-RMSE rule. Every applied
    reader will ask what AIC, BIC and Hannan-Quinn would have chosen, and those
    come free from ``statsmodels``.

    Parameters
    ----------
    Y : pandas.DataFrame or array-like
    W : pandas.DataFrame, array-like or None
    config : OracleVARXConfig, optional
    learner : str or estimator, default ``"linear"``
    seed : int or None, default 42

    Returns
    -------
    pandas.DataFrame
        One row per rule, with the mean, median and range of the selected lag
        order, and the resulting forecast error where one is defined.

    Notes
    -----
    The information criteria are evaluated once on the whole sample, as is
    standard practice, so they produce a single lag order rather than a path.
    """
    from statsmodels.tsa.api import VAR

    from ._model import OracleVARX

    config = config or OracleVARXConfig.synthetic()
    Y_values, _, _ = as_matrix(Y, "Y")
    rows = []

    for method, rule in (("acle-varx", "ACLE"), ("varx", "rolling RMSE")):
        if W is None:
            method = method.replace("varx", "var")
        model = OracleVARX(config, method=method, learner=learner, seed=seed)
        res = model.fit(Y, W)
        rows.append(
            {
                "rule": rule,
                "mean_p": float(res.lag.mean()),
                "median_p": float(np.median(res.lag)),
                "min_p": int(res.lag.min()),
                "max_p": int(res.lag.max()),
                "fcst_mae": float(
                    np.abs(res.actuals - res.forecasts).mean()
                ),
            }
        )

    selected = VAR(Y_values).select_order(config.p_max).selected_orders
    for name in ("aic", "bic", "hqic", "fpe"):
        order = int(selected[name])
        rows.append(
            {
                "rule": name.upper(),
                "mean_p": float(order),
                "median_p": float(order),
                "min_p": order,
                "max_p": order,
                "fcst_mae": np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("rule")


def leave_one_confounder_out(
    Y,
    W,
    config: OracleVARXConfig | None = None,
    method: str = "oracle-varx",
    learner: str | Any = "linear",
    seed: int | None = 42,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """How far each edge moves when one confounder is dropped.

    §7 of the paper asks for "tests that flag hidden confounding". Dropping each
    observed confounder in turn and measuring how much every coefficient shifts is
    the orthogonalised analogue of an omitted-variable probe, and it turns the
    paper's own Partial-2 and Partial-1 exercise into a tool a user can run on
    their own data. A coefficient that moves a long way when one observed
    confounder is removed is a coefficient that would move under an unobserved
    one too.

    Parameters
    ----------
    Y : pandas.DataFrame or array-like
    W : pandas.DataFrame or array-like
        At least two confounders.
    config : OracleVARXConfig, optional
    method : str, default ``"oracle-varx"``
    learner : str or estimator, default ``"linear"``
    seed : int or None, default 42
    n_jobs : int, default 1

    Returns
    -------
    pandas.DataFrame
        One row per dropped confounder, with the mean and maximum absolute shift
        in the coefficients, the share of edges that change status, and the name
        of the edge that moves most.

    Raises
    ------
    ValueError
        If fewer than two confounders are supplied.

    Notes
    -----
    This is a sensitivity probe, not a test of causal sufficiency: it bounds how
    fragile the estimates are to the confounders you have, and says nothing
    directly about the ones you do not.
    """
    from ._model import OracleVARX

    W_values, confounder_names, _ = as_matrix(W, "W")
    if W_values.shape[1] < 2:
        raise ValueError(
            "leave-one-out needs at least two confounders; with one there is "
            "nothing to compare against"
        )
    config = config or OracleVARXConfig.synthetic()

    base = OracleVARX(
        config, method=method, learner=learner, seed=seed, n_jobs=n_jobs
    ).fit(Y, W_values)

    rows = []
    for drop in range(W_values.shape[1]):
        kept = np.delete(W_values, drop, axis=1)
        trial = OracleVARX(
            config, method=method, learner=learner, seed=seed, n_jobs=n_jobs
        ).fit(Y, kept)
        shift = np.abs(trial.coef - base.coef)
        flips = trial.edges != base.edges
        flat = np.unravel_index(np.argmax(shift.mean(axis=0)), shift.shape[1:])
        lag, to_idx, from_idx = flat
        rows.append(
            {
                "dropped": confounder_names[drop],
                "mean_shift": float(shift.mean()),
                "max_shift": float(shift.max()),
                "edge_flip_share": float(flips.mean()),
                "most_moved": (
                    f"{base.names[from_idx]} -> {base.names[to_idx]} (lag {lag + 1})"
                ),
            }
        )
    return pd.DataFrame(rows).set_index("dropped").sort_values(
        "mean_shift", ascending=False
    )
