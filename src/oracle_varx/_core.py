"""The ORACLE-VARX estimator: orthogonalised second stage, ACLE, BH edges.

Equation references are to Tan, Zanoria, Chen, Lyu and Cucuringu (2026),
`doi:10.48550/arXiv.2610.05618 <https://doi.org/10.48550/arXiv.2610.05618>`_.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from ._config import OracleVARXConfig
from ._nuisance import cross_fit
from ._utils import blocked_folds, rolling_windows

__all__ = [
    "second_stage",
    "coefficients_from_theta",
    "z_scores",
    "acle_stage1",
    "rolling_validation",
    "rmse_lag_selection",
    "bh_edges",
    "cost_report",
    "fit_windows",
    "METHODS",
    "WindowFit",
]

#: The nested hierarchy of Table 1: each adds one capability to the row above.
METHODS: dict[str, dict[str, object]] = {
    "var": {"lag": "RMSE", "confounders": "none", "dml": False},
    "varx": {"lag": "RMSE", "confounders": "linear", "dml": False},
    "acle-var": {"lag": "ACLE", "confounders": "none", "dml": False},
    "acle-varx": {"lag": "ACLE", "confounders": "linear", "dml": False},
    "or-varx": {"lag": "RMSE", "confounders": "nonlinear", "dml": True},
    "oracle-varx": {"lag": "ACLE", "confounders": "nonlinear", "dml": True},
}


@dataclass
class WindowFit:
    """Per-window output of the first two stages, before lag selection.

    Attributes
    ----------
    days : numpy.ndarray
        ``(B,)`` absolute row index of each output day, one per rolling window.
    coef : numpy.ndarray
        ``(B, p_max, n, n)`` coefficients from the ``p_max`` fit, in the paper's
        convention ``A_k[i, j]`` = effect of series ``j`` at lag ``k + 1`` on
        series ``i``. Algorithm 1 lines 10 and 14 take the z-scores from this fit,
        not from the per-``p`` fits.
    se : numpy.ndarray
        ``(B, p_max, n, n)`` standard errors from eq. (8).
    forecasts : numpy.ndarray
        ``(B, n, p_max)`` one-step forecast under each candidate lag order.
    actuals : numpy.ndarray
        ``(B, n)`` realised values on the output days.
    n_fits : int
        First-stage fits performed (0 for the non-DML rungs).
    dof : int
        Residual degrees of freedom of the ``p_max`` second stage.
    """

    days: np.ndarray
    coef: np.ndarray
    se: np.ndarray
    forecasts: np.ndarray
    actuals: np.ndarray
    n_fits: int = 0
    dof: int = 0
    coef_per_p: np.ndarray | None = None
    se_per_p: np.ndarray | None = None


def second_stage(
    residual_T: np.ndarray,
    residual_Y: np.ndarray,
    window: int,
    ridge: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Orthogonalised OLS and its standard errors on every rolling window.

    Implements eq. (7) and eq. (8):

    .. math::
        \\hat\\Theta = \\Big(\\sum_t \\tilde T_t \\tilde T_t'\\Big)^{-1}
                       \\Big(\\sum_t \\tilde T_t \\tilde Y_t'\\Big), \\qquad
        \\widehat{\\mathrm{Var}}(\\hat\\Theta_{\\cdot i}) =
            \\hat\\sigma_i^2 \\Big(\\sum_t \\tilde T_t \\tilde T_t'\\Big)^{-1},

    with :math:`\\hat\\sigma_i^2 = (T_{\\mathrm{eff}} - np)^{-1}
    \\sum_t (\\tilde Y_{t,i} - \\hat\\Theta_{\\cdot i}' \\tilde T_t)^2`. No
    intercept: the inputs are residuals. No HAC correction: under conditional
    homoskedasticity, assumption (H), Theorem 1 leaves no autocovariance terms.

    Every window is solved in one batched call. ``np.linalg.solve`` on stacked
    Gram matrices was verified equal to a per-window ``lstsq`` loop to 1.2e-15,
    which is why this needs no GPU tensor library.

    Parameters
    ----------
    residual_T : numpy.ndarray
        ``(n_rows, k)`` residualised stacked lags, ``k = n * p``.
    residual_Y : numpy.ndarray
        ``(n_rows, n)`` residualised outcomes.
    window : int
        Window length ``T_w``.
    ridge : float, default 0.0
        Added to the diagonal of the Gram matrix before solving. The paper uses
        plain OLS; the authors' code adds ``1e-6`` in its VAR path only. Keep 0 to
        follow the paper, and set ``1e-6`` to compare against their VAR numbers.

    Returns
    -------
    theta : numpy.ndarray
        ``(B, k, n)``.
    se : numpy.ndarray
        ``(B, k, n)``.
    dof : int
        ``window - k``.

    Raises
    ------
    ValueError
        If the window cannot support ``k`` regressors, or is longer than the data.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> T = rng.normal(size=(120, 2))
    >>> Y = T @ np.array([[0.5, 0.0], [0.0, -0.3]]) + rng.normal(scale=0.01, size=(120, 2))
    >>> theta, se, dof = second_stage(T, Y, window=100)
    >>> theta.shape, se.shape, dof
    ((21, 2, 2), (21, 2, 2), 98)
    >>> float(np.round(theta[0, 0, 0], 2))
    0.5
    """
    n_rows, k = residual_T.shape
    if residual_Y.shape[0] != n_rows:
        raise ValueError(
            f"residual_T has {n_rows} rows but residual_Y has {residual_Y.shape[0]}"
        )
    if window > n_rows:
        raise ValueError(f"window={window} exceeds the {n_rows} residual rows")
    dof = window - k
    if dof <= 0:
        raise ValueError(
            f"window={window} leaves no degrees of freedom for k={k} regressors; "
            f"raise ols_window above {k} or lower p_max"
        )

    T_win = np.ascontiguousarray(rolling_windows(residual_T, window))
    Y_win = np.ascontiguousarray(rolling_windows(residual_Y, window))

    gram = T_win.transpose(0, 2, 1) @ T_win
    if ridge:
        gram = gram + ridge * np.eye(k)
    moment = T_win.transpose(0, 2, 1) @ Y_win

    try:
        theta = np.linalg.solve(gram, moment)
    except np.linalg.LinAlgError:
        # Benign collinearity: fall back to the pseudo-inverse rather than failing.
        theta = np.linalg.pinv(gram) @ moment

    resid = Y_win - T_win @ theta
    sigma2 = (resid**2).sum(axis=1) / dof
    gram_inv = np.linalg.pinv(gram)
    diag = np.einsum("bkk->bk", gram_inv)
    se = np.sqrt(np.maximum(diag[:, :, None] * sigma2[:, None, :], 0.0))
    return theta, se, dof


def coefficients_from_theta(theta: np.ndarray, n: int, p: int) -> np.ndarray:
    """Reshape a stacked ``Theta`` into lag matrices ``A_k[i, j]``.

    Row ``(k - 1) * n + j`` of ``Theta`` is series ``j`` at lag ``k``, and column
    ``i`` is the outcome series, so the result indexes as ``A[k - 1, i, j]`` — the
    paper's convention in Definition 1.

    Parameters
    ----------
    theta : numpy.ndarray
        ``(B, n * p, n)`` or ``(n * p, n)``.
    n, p : int

    Returns
    -------
    numpy.ndarray
        ``(B, p, n, n)``, or ``(p, n, n)`` for 2-D input.

    Examples
    --------
    >>> import numpy as np
    >>> theta = np.arange(8.0).reshape(4, 2)       # n=2, p=2
    >>> A = coefficients_from_theta(theta, n=2, p=2)
    >>> A.shape
    (2, 2, 2)
    >>> bool(A[0, 1, 0] == theta[0, 1])            # series 0 at lag 1 -> series 1
    True
    """
    squeeze = theta.ndim == 2
    stacked = theta[None] if squeeze else theta
    B = stacked.shape[0]
    if stacked.shape[1] != n * p:
        raise ValueError(
            f"theta has {stacked.shape[1]} rows, expected n * p = {n * p}"
        )
    out = stacked.reshape(B, p, n, n).transpose(0, 1, 3, 2)
    return out[0] if squeeze else out


def z_scores(coef: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Entry-wise z-statistics ``A_hat / se(A_hat)``.

    Entries with a non-positive or non-finite standard error — lag blocks that
    were never fitted — are given ``z = 0`` so they can never be selected as
    edges, following the authors' masking of unfitted blocks.

    Parameters
    ----------
    coef, se : numpy.ndarray
        Same shape.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> import numpy as np
    >>> z_scores(np.array([0.4, 0.2]), np.array([0.1, 0.0]))
    array([4., 0.])
    """
    out = np.zeros_like(np.asarray(coef, dtype=np.float64))
    good = np.isfinite(se) & (se > 0) & np.isfinite(coef)
    out[good] = coef[good] / se[good]
    return out


def _two_sided_p(z: np.ndarray) -> np.ndarray:
    """Two-sided normal p-values, ``p = 2 (1 - Phi(|z|))`` (Appendix D.4)."""
    return 2.0 * norm.sf(np.abs(np.asarray(z, dtype=np.float64)))


def _bh_reject(p_values: np.ndarray, level: float, method: str = "fdr_bh") -> np.ndarray:
    """Benjamini-Hochberg (or Yekutieli) step-up on a flat vector of p-values.

    Delegated to :func:`statsmodels.stats.multitest.multipletests`, which was
    verified identical to the paper's own statement of the rule in Appendix D.4.
    """
    flat = np.asarray(p_values, dtype=np.float64).ravel()
    if flat.size == 0:
        return np.zeros(0, dtype=bool)
    return multipletests(flat, alpha=level, method=method)[0]


def acle_stage1(
    coef: np.ndarray,
    se: np.ndarray,
    alpha: float,
    select: Sequence[int] | None = None,
    method: str = "fdr_bh",
) -> np.ndarray:
    """ACLE Stage 1: pick the lag order by sequential significance testing.

    For each window, lag 1 is always kept. Then for ``p = 2, ..., p_max`` the null
    ``H_0: A_p[i, j] = 0`` for all ``(i, j)`` is tested by applying the BH
    correction at level ``alpha`` to the ``n^2`` entry-wise z-tests of the lag-``p``
    block, taken from the ``p_max`` fit. The first ``p`` at which nothing is
    significant stops the search, giving ``p_alpha = p - 1`` (§3.3, Algorithm 1
    line 10).

    Parameters
    ----------
    coef, se : numpy.ndarray
        ``(B, p_max, n, n)`` from the ``p_max`` fit.
    alpha : float
        Stage-1 significance level.
    select : sequence of int, optional
        Series the lag test looks at. For the VARX rung, where the observed
        confounders are stacked into the system, pass the endogenous indices so
        the confounders' own equations do not drive lag selection — the paper's
        convention in the note to Table 4.
    method : {"fdr_bh", "fdr_by"}, default ``"fdr_bh"``
        ``"fdr_by"`` is an addition to the paper: the Benjamini-Yekutieli
        correction, valid under arbitrary dependence between the tests.

    Returns
    -------
    numpy.ndarray
        ``(B,)`` selected lag order per window, at least 1.

    Examples
    --------
    >>> import numpy as np
    >>> coef = np.zeros((1, 3, 2, 2)); se = np.full((1, 3, 2, 2), 0.1)
    >>> coef[0, 0] = 0.6          # lag 1 strong
    >>> coef[0, 1, 0, 0] = 0.5    # lag 2 has one strong entry
    >>> acle_stage1(coef, se, alpha=0.05)
    array([2])
    """
    coef = np.asarray(coef, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    if coef.ndim != 4:
        raise ValueError(f"coef must be 4-D (B, p_max, n, n), got {coef.shape}")
    n_windows, p_max, n, _ = coef.shape
    idx = np.arange(n) if select is None else np.asarray(select, dtype=int)

    selected = np.ones(n_windows, dtype=int)
    if p_max < 2:
        return selected

    z_all = z_scores(coef, se)
    for p in range(2, p_max + 1):
        block = z_all[:, p - 1][:, idx][:, :, idx]
        p_vals = _two_sided_p(block).reshape(n_windows, -1)
        # A window is still deciding only if it accepted every lag up to p - 1.
        active = selected == p - 1
        if not active.any():
            break
        for w in np.flatnonzero(active):
            if _bh_reject(p_vals[w], alpha, method).any():
                selected[w] = p
    return selected


def rolling_validation(
    forecasts: np.ndarray,
    actuals: np.ndarray,
    lag_choice: np.ndarray,
    validation_days: int,
    select: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ACLE Stage 2: choose the significance level by trailing validation error.

    Implements eq. (9)-(10): for output day ``t`` and each candidate level, the
    forecasts made under that level's selected lag order are scored over the
    ``v`` days strictly before ``t``, and the level with the smallest rolling RMSE
    wins. Because RMSE is monotone in MSE, the arg-min is taken on MSE.

    Parameters
    ----------
    forecasts : numpy.ndarray
        ``(B, n, p_max)`` forecast under each candidate lag order.
    actuals : numpy.ndarray
        ``(B, n)``.
    lag_choice : numpy.ndarray
        ``(B, M)`` lag order selected by each of the ``M`` candidate levels.
    validation_days : int
        Length ``v`` of the trailing window.
    select : sequence of int, optional
        Series whose forecast errors the choice is made on.

    Returns
    -------
    chosen : numpy.ndarray
        ``(B - v,)`` index into the candidate grid, per output day.
    lag : numpy.ndarray
        ``(B - v,)`` the lag order that index implies.
    mse : numpy.ndarray
        ``(B - v, M)`` the rolling mean squared error behind the choice.

    Raises
    ------
    ValueError
        If there are no days left after reserving the validation window.

    Examples
    --------
    >>> import numpy as np
    >>> f = np.zeros((6, 1, 2)); f[:, 0, 1] = 1.0       # lag 2 forecasts badly
    >>> a = np.zeros((6, 1))
    >>> lag = np.tile(np.array([1, 2]), (6, 1))
    >>> chosen, lag_hat, _ = rolling_validation(f, a, lag, validation_days=2)
    >>> chosen.tolist(), lag_hat.tolist()
    ([0, 0, 0, 0], [1, 1, 1, 1])
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    lag_choice = np.asarray(lag_choice, dtype=int)

    n_windows, n_series, p_max = forecasts.shape
    n_candidates = lag_choice.shape[1]
    if lag_choice.shape[0] != n_windows:
        raise ValueError("forecasts and lag_choice disagree on the number of windows")
    n_out = n_windows - validation_days
    if n_out < 1:
        raise ValueError(
            f"{n_windows} windows leave nothing after reserving "
            f"validation_days={validation_days}; lengthen the sample"
        )
    idx = np.arange(n_series) if select is None else np.asarray(select, dtype=int)

    # Forecast error of each candidate on every day, averaged over the series.
    taken = np.take_along_axis(
        forecasts[:, idx, :], lag_choice[:, None, :] - 1, axis=2
    )  # (B, |idx|, M)
    errors = ((taken - actuals[:, idx, None]) ** 2).mean(axis=1)  # (B, M)

    cumulative = np.vstack([np.zeros((1, n_candidates)), errors.cumsum(axis=0)])
    starts = np.arange(n_out)
    ends = starts + validation_days
    mse = (cumulative[ends] - cumulative[starts]) / validation_days

    chosen = mse.argmin(axis=1)
    lag = lag_choice[validation_days:][np.arange(n_out), chosen]
    return chosen, lag, mse


def rmse_lag_selection(
    forecasts: np.ndarray,
    actuals: np.ndarray,
    validation_days: int,
    select: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lag selection by rolling validation RMSE over ``1, ..., p_max`` (eq. 9).

    The rule the paper uses for the non-adaptive rungs: VAR, VARX and OR-VARX.

    Parameters
    ----------
    forecasts : numpy.ndarray
        ``(B, n, p_max)``.
    actuals : numpy.ndarray
        ``(B, n)``.
    validation_days : int
        Length ``v`` of the trailing window.
    select : sequence of int, optional
        Series the choice is made on.

    Returns
    -------
    lag : numpy.ndarray
        ``(B - v,)`` selected lag order per output day.
    mse : numpy.ndarray
        ``(B - v, p_max)``.

    Examples
    --------
    >>> import numpy as np
    >>> f = np.zeros((5, 1, 3)); f[:, 0, 0] = 1.0     # lag 1 forecasts badly
    >>> lag, _ = rmse_lag_selection(f, np.zeros((5, 1)), validation_days=2)
    >>> sorted(set(lag.tolist()))
    [2]
    """
    forecasts = np.asarray(forecasts, dtype=np.float64)
    p_max = forecasts.shape[2]
    candidates = np.tile(np.arange(1, p_max + 1), (forecasts.shape[0], 1))
    chosen, lag, mse = rolling_validation(
        forecasts, actuals, candidates, validation_days, select=select
    )
    return lag, mse


def bh_edges(
    coef: np.ndarray,
    se: np.ndarray,
    lag_hat: np.ndarray,
    q: float = 0.05,
    select: Sequence[int] | None = None,
    method: str = "fdr_bh",
) -> tuple[np.ndarray, np.ndarray]:
    """FDR-controlled edge discovery, one BH step per window (§3.4 step 5, D.4).

    For window ``t``, the ``n^2 * p_hat_t`` entries at lags ``1, ..., p_hat_t`` get
    two-sided p-values from ``z = A_hat / se``, and BH at level ``q`` selects the
    edges. Entries at lags deeper than ``p_hat_t`` are never edges.

    Parameters
    ----------
    coef, se : numpy.ndarray
        ``(B, p_max, n, n)`` from the ``p_max`` fit.
    lag_hat : numpy.ndarray
        ``(B,)`` selected lag order per window.
    q : float, default 0.05
        Target false discovery rate.
    select : sequence of int, optional
        Restrict both the tests and the edges to these series.
    method : {"fdr_bh", "fdr_by"}, default ``"fdr_bh"``
        ``"fdr_by"`` is an addition to the paper, valid under arbitrary
        dependence between the entry-wise tests (Benjamini and Yekutieli, 2001).

    Returns
    -------
    edges : numpy.ndarray
        ``(B, p_max, n, n)`` boolean.
    qvalues : numpy.ndarray
        ``(B, p_max, n, n)`` BH-adjusted p-values, ``nan`` outside the tested
        lags. These are an addition to the paper, which reports only the
        selection.

    Examples
    --------
    >>> import numpy as np
    >>> coef = np.zeros((1, 2, 2, 2)); se = np.full((1, 2, 2, 2), 0.05)
    >>> coef[0, 0, 0, 1] = 0.6
    >>> edges, qvals = bh_edges(coef, se, np.array([1]), q=0.05)
    >>> bool(edges[0, 0, 0, 1]), bool(edges[0, 1].any())
    (True, False)
    """
    coef = np.asarray(coef, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    lag_hat = np.asarray(lag_hat, dtype=int)
    n_windows, p_max, n, _ = coef.shape
    if lag_hat.shape[0] != n_windows:
        raise ValueError(
            f"lag_hat has {lag_hat.shape[0]} entries for {n_windows} windows"
        )
    idx = np.arange(n) if select is None else np.asarray(select, dtype=int)

    edges = np.zeros((n_windows, p_max, n, n), dtype=bool)
    qvalues = np.full((n_windows, p_max, n, n), np.nan)
    z_all = z_scores(coef, se)
    rows, cols = np.meshgrid(idx, idx, indexing="ij")

    for w in range(n_windows):
        p_hat = int(min(max(lag_hat[w], 1), p_max))
        block_z = z_all[w, :p_hat][:, idx][:, :, idx]
        p_vals = _two_sided_p(block_z)
        reject, adjusted = multipletests(p_vals.ravel(), alpha=q, method=method)[:2]
        shape = (p_hat, idx.size, idx.size)
        for k in range(p_hat):
            edges[w, k][rows, cols] = reject.reshape(shape)[k]
            qvalues[w, k][rows, cols] = adjusted.reshape(shape)[k]
    return edges, qvalues


def cost_report(
    n_obs: int,
    n_series: int,
    p_max: int,
    block: int,
    n_train: int,
) -> dict[str, int | float]:
    """First-stage fit counts, from the formulas of Appendix D.3.

    The paper's own accounting: each block fits ``n`` learners for the outcome and
    ``n p`` for the stacked lags at each candidate ``p``, so

    .. math:: S = \\sum_{p=1}^{p_{\\max}} n (p + 1) = n\\,p_{\\max}(p_{\\max}+3)/2

    fits per block, and ``F ~ (T - n_train - p_max) / block`` blocks. The two
    per-window alternatives of Table 11 cost roughly ``block`` and ``T_w`` times
    more.

    Parameters
    ----------
    n_obs, n_series, p_max, block, n_train : int

    Returns
    -------
    dict
        ``blocks``, ``fits_per_block``, ``fits``, ``fits_fresh_split_per_window``,
        ``fits_refit_per_window`` and the two ratios.

    Examples
    --------
    >>> report = cost_report(3000, 3, 5, 20, 200)
    >>> report["fits_per_block"], report["blocks"], report["fits"]
    (60, 139, 8340)
    """
    blocks = max(int((n_obs - n_train - p_max) // block), 1)
    per_block = int(n_series * p_max * (p_max + 3) // 2)
    fits = blocks * per_block
    windows = max(n_obs - n_train - p_max, 1)
    return {
        "blocks": blocks,
        "fits_per_block": per_block,
        "fits": fits,
        "windows": windows,
        "fits_fresh_split_per_window": windows * per_block,
        "fits_refit_per_window": windows * per_block * blocks // max(blocks, 1),
        "ratio_fresh_split": windows * per_block / fits if fits else float("nan"),
        "ratio_refit": (windows * per_block * blocks) / (fits * max(blocks, 1))
        if fits
        else float("nan"),
    }


# ---------------------------------------------------------------------------
# Rolling-window engines: one for the OLS rungs, one for the DML rungs
# ---------------------------------------------------------------------------


def _lag_design(system: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Stacked lags and aligned targets, via ``statsmodels``' ``lagmat``."""
    from statsmodels.tsa.tsatools import lagmat

    lags = lagmat(system, maxlag=p, trim="both", original="ex")
    return np.ascontiguousarray(lags), np.ascontiguousarray(system[p:])


def _ols_windows(
    system: np.ndarray,
    p: int,
    window: int,
    days: np.ndarray,
    intercept: bool = True,
    ridge: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling OLS VAR(p) on the raw system, with eq.-(8)-style standard errors.

    The window ending at day ``d`` uses the ``window`` rows whose targets are days
    ``d - window, ..., d - 1``, exactly as §3.2 describes, and forecasts day ``d``.
    (The authors' code instead hands each candidate ``p`` a fixed lookback, so a
    shallow lag order gets a few extra rows; taking exactly ``window`` rows for
    every ``p`` follows the paper and keeps the candidates comparable.)

    Returns ``(theta, se, forecasts)`` with shapes ``(B, n*p, n)``, the same, and
    ``(B, n)``.
    """
    n_obs, n = system.shape
    lags, targets = _lag_design(system, p)
    k = lags.shape[1] + (1 if intercept else 0)
    if intercept:
        lags = np.column_stack([np.ones(len(lags)), lags])

    starts = days - p - window
    if starts.min() < 0:
        raise ValueError("the first requested day has too little history")

    X = np.ascontiguousarray(rolling_windows(lags, window))[starts]
    Yd = np.ascontiguousarray(rolling_windows(targets, window))[starts]

    gram = X.transpose(0, 2, 1) @ X
    if ridge:
        gram = gram + ridge * np.eye(k)
    moment = X.transpose(0, 2, 1) @ Yd
    try:
        beta = np.linalg.solve(gram, moment)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(gram) @ moment

    dof = window - k
    if dof <= 0:
        raise ValueError(
            f"window={window} leaves no degrees of freedom for k={k} regressors"
        )
    resid = Yd - X @ beta
    sigma2 = (resid**2).sum(axis=1) / dof
    diag = np.einsum("bkk->bk", np.linalg.pinv(gram))
    se = np.sqrt(np.maximum(diag[:, :, None] * sigma2[:, None, :], 0.0))

    pred_rows = np.empty((days.size, k), dtype=np.float64)
    offset = 1 if intercept else 0
    if intercept:
        pred_rows[:, 0] = 1.0
    for lag in range(1, p + 1):
        pred_rows[:, offset + (lag - 1) * n : offset + lag * n] = system[days - lag]
    forecasts = np.einsum("bk,bkn->bn", pred_rows, beta)

    if intercept:
        beta, se = beta[:, 1:], se[:, 1:]
    return beta, se, forecasts


def _dml_windows(
    Y: np.ndarray,
    W: np.ndarray,
    p: int,
    config: OracleVARXConfig,
    days: np.ndarray,
    learner,
    seed: int | None,
    n_jobs: int,
    ridge: float,
    folds=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Cross-fit, residualise, and run the orthogonalised rolling second stage."""
    if folds is None:
        folds = blocked_folds(
            n_obs=Y.shape[0],
            n_train=config.n_train,
            block=config.block,
            buffer=config.buffer,
            p_max=config.p_max,
        )
    nuisance = cross_fit(Y, W, p, folds, learner=learner, seed=seed, n_jobs=n_jobs)

    window = config.ols_window
    starts = days - window - nuisance.first_row
    if starts.min() < 0:
        raise ValueError(
            "not enough residual rows before the first output day; shorten "
            "ols_window or lengthen the sample"
        )
    if (days - nuisance.first_row).max() >= nuisance.n_rows:
        raise ValueError("output days run past the last cross-fitted row")

    theta_all, se_all, _ = second_stage(
        nuisance.residual_T, nuisance.residual_Y, window=window, ridge=ridge
    )
    theta, se = theta_all[starts], se_all[starts]

    rows = days - nuisance.first_row
    n = Y.shape[1]
    actual_T = np.empty((days.size, n * p), dtype=np.float64)
    for lag in range(1, p + 1):
        actual_T[:, (lag - 1) * n : lag * n] = Y[days - lag]
    resid_T = actual_T - nuisance.pred_T[rows]
    forecasts = nuisance.pred_Y[rows] + np.einsum("bk,bkn->bn", resid_T, theta)
    return theta, se, forecasts, nuisance.n_fits


def fit_windows(
    Y: np.ndarray,
    W: np.ndarray | None,
    config: OracleVARXConfig,
    method: str = "oracle-varx",
    learner="extra_trees",
    seed: int | None = 42,
    n_jobs: int = 1,
    ridge: float = 0.0,
    verbose: bool = False,
    store_per_p: bool = False,
) -> tuple[WindowFit, np.ndarray]:
    """Run Algorithm 1 steps 1-3 for every rolling window and candidate lag order.

    Walks ``p = 1, ..., p_max``, producing a forecast under each candidate lag
    order, and keeps the coefficients and standard errors of the ``p_max`` fit,
    which is what Algorithm 1 lines 10 and 14 test.

    Parameters
    ----------
    Y : numpy.ndarray
        Endogenous series ``(n_obs, n)``.
    W : numpy.ndarray or None
        Confounders ``(n_obs, d)``.
    config : OracleVARXConfig
    method : str, default ``"oracle-varx"``
        A key of :data:`METHODS`.
    learner : str or estimator, default ``"extra_trees"``
        First-stage learner, for the DML rungs.
    seed : int or None, default 42
    n_jobs : int, default 1
    ridge : float, default 0.0
        Diagonal nudge on the Gram matrix; 0 is the paper.
    verbose : bool, default False
    store_per_p : bool, default False
        Also keep each candidate order's own coefficients and standard errors.
        Algorithm 1 tests the ``p_max`` fit, but the paper's coefficient error and
        its Figure 3 read the coefficients of the **selected** order, which is
        what the authors' own ``coefficient_refit`` step produces; set this to
        make that reading available.

    Returns
    -------
    (fit, select) : tuple of WindowFit and numpy.ndarray
        ``select`` holds the indices of the series lag selection and edge scoring
        apply to, which for the VARX rungs excludes the stacked confounders.

    Raises
    ------
    ValueError
        For an unknown method, a sample too short for the configuration, or a DML
        rung with no confounders.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose one of {sorted(METHODS)}")
    spec = METHODS[method]
    n_obs, n = Y.shape
    has_w = W is not None and W.shape[1] > 0

    if spec["confounders"] == "none":
        system, select = Y, np.arange(n)
    elif spec["dml"]:
        if not has_w:
            raise ValueError(
                "method " + repr(method) + " needs confounders; with none, use "
                "var or acle-var, which the paper reports as its none level"
            )
        system, select = Y, np.arange(n)
    else:
        if not has_w:
            raise ValueError("method " + repr(method) + " needs confounders")
        # Table 4's note: VARX stacks the observed confounders in as extra series,
        # but only the target series drive lag selection and are scored.
        system, select = np.hstack([Y, W]), np.arange(n)

    p_max = config.p_max
    folds = None
    if spec["dml"]:
        # The blocks do not depend on p, so build them once and reuse them for
        # every candidate lag order: that reuse is the memoisation of §3.2.
        folds = blocked_folds(
            n_obs=n_obs,
            n_train=config.n_train,
            block=config.block,
            buffer=config.buffer,
            p_max=p_max,
        )
        start_day = min(f.test_start for f in folds) + config.ols_window
        # A day can only be forecast if some fold supplied its nuisance
        # predictions, so the last incomplete block is not an output day.
        end_day = max(f.test_end for f in folds)
    else:
        start_day = config.ols_window + p_max
        end_day = n_obs
    if start_day >= end_day:
        raise ValueError(
            f"the configuration leaves no output days: it needs more than "
            f"{start_day} usable observations but the sample supports "
            f"{end_day}; lower ols_window (now {config.ols_window}) or p_max "
            f"(now {p_max})"
        )
    days = np.arange(start_day, end_day)

    n_sys = system.shape[1]
    forecasts = np.zeros((days.size, n_sys, p_max), dtype=np.float64)
    coef = np.zeros((days.size, p_max, n_sys, n_sys), dtype=np.float64)
    se = np.zeros((days.size, p_max, n_sys, n_sys), dtype=np.float64)
    n_fits = 0
    dof = 0
    coef_per_p = (
        np.zeros((days.size, p_max, p_max, n_sys, n_sys), dtype=np.float64)
        if store_per_p
        else None
    )
    se_per_p = np.zeros_like(coef_per_p) if store_per_p else None

    for p in range(1, p_max + 1):
        if spec["dml"]:
            theta_p, se_p, fcst_p, fits = _dml_windows(
                Y, W, p, config, days, learner, seed, n_jobs, ridge, folds=folds
            )
            n_fits += fits
            dof = config.ols_window - n_sys * p
        else:
            theta_p, se_p, fcst_p = _ols_windows(
                system, p, config.ols_window, days, intercept=True, ridge=ridge
            )
            fits = 0
            dof = config.ols_window - (n_sys * p + 1)
        forecasts[:, :, p - 1] = fcst_p
        if store_per_p:
            coef_per_p[:, p - 1, :p] = coefficients_from_theta(theta_p, n_sys, p)
            se_per_p[:, p - 1, :p] = coefficients_from_theta(se_p, n_sys, p)
        if p == p_max:
            coef = coefficients_from_theta(theta_p, n_sys, p_max)
            se = coefficients_from_theta(se_p, n_sys, p_max)
        if verbose:
            print(f"  p={p}: {days.size} windows, {fits} first-stage fits")

    fit = WindowFit(
        days=days,
        coef=coef,
        se=se,
        forecasts=forecasts,
        actuals=system[days],
        n_fits=n_fits,
        dof=dof,
        coef_per_p=coef_per_p,
        se_per_p=se_per_p,
    )
    return fit, select
