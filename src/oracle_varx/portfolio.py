"""The long-short portfolio evaluation of Appendix C.

Three position rules, unit gross exposure, a daily SPY dollar hedge and no
transaction costs — exactly as the paper specifies. The paper is careful that this
is a dollar hedge, not a beta hedge, so some market exposure can remain; and that
its own contribution is the quality of the recovered graph, not trading
performance.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "RULES",
    "weights",
    "strategy_returns",
    "performance",
    "performance_table",
]

#: The position rules of Appendix C.1.
RULES: tuple[str, ...] = ("naive", "weighted", "top25", "top50", "top75")

_TRADING_DAYS = 252


def weights(forecasts: np.ndarray, rule: str = "weighted") -> np.ndarray:
    """Portfolio weights from one day's forecasts (Appendix C.1).

    * ``"naive"`` — sign of the forecast, equally weighted.
    * ``"weighted"`` — proportional to the forecast, so stronger predicted moves
      get larger allocations.
    * ``"top25"``, ``"top50"``, ``"top75"`` — trade only the assets whose absolute
      forecast exceeds that quantile of the day's absolute forecasts, sign-based
      and equally weighted among them. The labels name the **quantile threshold**,
      not the share of assets traded, so ``"top75"`` is the most concentrated
      book — the paper is explicit about this, and it is easy to read backwards.

    Every rule is normalised to unit gross exposure, ``sum |w| = 1``.

    Parameters
    ----------
    forecasts : numpy.ndarray
        ``(n,)`` or ``(B, n)`` forecast returns.
    rule : str, default ``"weighted"``
        One of :data:`RULES`.

    Returns
    -------
    numpy.ndarray
        Same shape as ``forecasts``.

    Raises
    ------
    ValueError
        For an unknown rule.

    Examples
    --------
    >>> import numpy as np
    >>> w = weights(np.array([0.02, -0.01, 0.0]), rule="weighted")
    >>> float(np.abs(w).sum())
    1.0
    >>> weights(np.array([0.02, -0.01, 0.005]), rule="naive").round(4).tolist()
    [0.3333, -0.3333, 0.3333]
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; choose one of {RULES}")
    values = np.atleast_2d(np.asarray(forecasts, dtype=np.float64))
    magnitude = np.abs(values)

    if rule == "weighted":
        raw = values
    elif rule == "naive":
        raw = np.sign(values)
    else:
        quantile = {"top25": 0.25, "top50": 0.50, "top75": 0.75}[rule]
        cutoff = np.quantile(magnitude, quantile, axis=1, keepdims=True)
        raw = np.where(magnitude >= cutoff, np.sign(values), 0.0)

    gross = np.abs(raw).sum(axis=1, keepdims=True)
    out = np.divide(raw, gross, out=np.zeros_like(raw), where=gross > 0)
    return out[0] if np.asarray(forecasts).ndim == 1 else out


def strategy_returns(
    forecasts: np.ndarray,
    realised: np.ndarray,
    hedge: np.ndarray | None = None,
    rule: str = "weighted",
) -> np.ndarray:
    """Daily market-adjusted returns of the long-short book (Appendix C.1).

    The net exposure ``G_t = sum_i w_{i,t}`` is generally not zero, so it is
    offset with an opposite position in the hedge instrument:
    ``r_adj = sum_i w_i r_i - G_t * b_t``.

    Parameters
    ----------
    forecasts, realised : numpy.ndarray
        ``(B, n)`` forecasts and realised returns, aligned.
    hedge : numpy.ndarray, optional
        ``(B,)`` returns of the hedge instrument, SPY in the paper. ``None``
        leaves the book unhedged and says so by returning the raw book return.
    rule : str, default ``"weighted"``

    Returns
    -------
    numpy.ndarray
        ``(B,)`` daily returns.

    Examples
    --------
    >>> import numpy as np
    >>> r = strategy_returns(np.array([[1.0, -1.0]]), np.array([[0.02, -0.01]]))
    >>> float(np.round(r[0], 4))
    0.015
    """
    forecasts = np.atleast_2d(np.asarray(forecasts, dtype=np.float64))
    realised = np.atleast_2d(np.asarray(realised, dtype=np.float64))
    if forecasts.shape != realised.shape:
        raise ValueError(
            f"forecasts {forecasts.shape} and realised {realised.shape} must match"
        )
    w = weights(forecasts, rule=rule)
    book = (w * realised).sum(axis=1)
    if hedge is None:
        return book
    hedge = np.asarray(hedge, dtype=np.float64).ravel()
    if hedge.size != book.size:
        raise ValueError(
            f"hedge has {hedge.size} rows but the book has {book.size}"
        )
    net = w.sum(axis=1)
    return book - net * hedge


def performance(returns: np.ndarray, periods: int = _TRADING_DAYS) -> dict[str, float]:
    """Annualised return, volatility, Sharpe ratio and drawdown.

    Returns are compounded multiplicatively, as the paper specifies: the
    cumulative return at day ``T`` is ``prod(1 + r_t) - 1``.

    Parameters
    ----------
    returns : numpy.ndarray
        ``(B,)`` daily returns.
    periods : int, default 252
        Periods per year.

    Returns
    -------
    dict
        ``ann_return``, ``ann_volatility``, ``sharpe``, ``cumulative``,
        ``max_drawdown``, ``hit_rate``, ``n_days``.

    Examples
    --------
    >>> import numpy as np
    >>> stats = performance(np.full(252, 0.0004))
    >>> round(stats["ann_return"], 4), round(stats["sharpe"], 2)
    (0.1062, inf)
    """
    series = np.asarray(returns, dtype=np.float64).ravel()
    series = series[np.isfinite(series)]
    if series.size == 0:
        raise ValueError("no finite returns to score")

    growth = np.cumprod(1.0 + series)
    years = series.size / periods
    ann_return = float(growth[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    volatility = float(series.std(ddof=1) * np.sqrt(periods)) if series.size > 1 else 0.0
    sharpe = float(series.mean() / series.std(ddof=1) * np.sqrt(periods)) if (
        series.size > 1 and series.std(ddof=1) > 0
    ) else float("inf")
    peak = np.maximum.accumulate(growth)
    drawdown = float((growth / peak - 1.0).min())
    return {
        "ann_return": ann_return,
        "ann_volatility": volatility,
        "sharpe": sharpe,
        "cumulative": float(growth[-1] - 1.0),
        "max_drawdown": drawdown,
        "hit_rate": float((series > 0).mean()),
        "n_days": int(series.size),
    }


def performance_table(
    results,
    hedge: np.ndarray | pd.Series | None = None,
    rules: Sequence[str] = RULES,
    periods: int = _TRADING_DAYS,
) -> pd.DataFrame:
    """Score a fit under every position rule — the paper's Tables 6 to 9.

    Parameters
    ----------
    results : OracleVARXResults
    hedge : array-like, optional
        Returns of the hedge instrument on the output days.
    rules : sequence of str, default :data:`RULES`
    periods : int, default 252

    Returns
    -------
    pandas.DataFrame
        One row per rule, in the paper's column order.

    Notes
    -----
    Transaction costs are not modelled, and the hedge is a dollar hedge, so these
    numbers are an upper bound on what the signal is worth. The paper's own
    reading is that its DML rungs trade predictive power for causal validity, so
    the simpler rungs often post the higher Sharpe ratio.
    """
    if hedge is not None:
        hedge = np.asarray(
            hedge.to_numpy() if isinstance(hedge, pd.Series) else hedge, dtype=float
        ).ravel()
        if hedge.size != results.n_windows:
            raise ValueError(
                f"hedge has {hedge.size} rows but the fit has "
                f"{results.n_windows} output days"
            )

    rows = []
    for rule in rules:
        series = strategy_returns(
            results.forecasts, results.actuals, hedge=hedge, rule=rule
        )
        stats = performance(series, periods=periods)
        rows.append({"rule": rule, **stats})
    frame = pd.DataFrame(rows).set_index("rule")
    return frame[
        ["ann_return", "ann_volatility", "sharpe", "cumulative", "max_drawdown",
         "hit_rate", "n_days"]
    ]
