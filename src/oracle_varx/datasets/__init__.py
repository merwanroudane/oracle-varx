"""Datasets: the paper's synthetic benchmark, and a real shipped dataset.

* :func:`make_oracle_dgp` reproduces the data-generating process of §5.1 and
  eq. (12) exactly, with its known time-varying ground truth, which is the
  known-truth harness the estimator is validated against.
* :func:`load_treasury` loads the real dataset shipped inside the package: daily
  US Treasury yield-curve changes with the real (TIPS) curve as confounders. A
  work of the US federal government, so public domain and redistributable, unlike
  the CRSP returns the paper uses.
* :mod:`oracle_varx.datasets._sources` fetches closer substitutes on demand: the
  paper's own sector ETFs via Yahoo, or the Fama-French industry portfolios.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import numpy as np
import pandas as pd

from ._sources import (
    FRED_PRESETS,
    SECTOR_ETFS,
    fetch_fred,
    fetch_industry_portfolios,
    fetch_sector_etfs,
    fetch_treasury_curve,
)

__all__ = [
    "ToyGroundTruth",
    "make_oracle_dgp",
    "observed_confounders",
    "load_treasury",
    "fetch_treasury_curve",
    "fetch_industry_portfolios",
    "fetch_sector_etfs",
    "fetch_fred",
    "SECTOR_ETFS",
    "FRED_PRESETS",
]


@dataclass
class ToyGroundTruth:
    """Known truth for the synthetic benchmark of §5.1.

    Attributes
    ----------
    A : numpy.ndarray
        ``(T, 3, n, n)`` with ``A[t, k, i, j]`` the effect of series ``j`` at lag
        ``k + 1`` on series ``i`` at time ``t``.
    p_true : numpy.ndarray
        ``(T,)`` the true maximum causal lag at each date: 3, then 2, then 1.
    regimes : tuple
        The three ``(start, end)`` regime boundaries.
    names : list of str
    confounder_names : list of str
    """

    A: np.ndarray
    p_true: np.ndarray
    regimes: tuple[tuple[int, int], ...]
    names: list[str]
    confounder_names: list[str]


def _g_nonlinear(W: np.ndarray) -> np.ndarray:
    """The three nuisance functions of eq. (12), each normalised to unit scale."""
    w1, w2, w3 = W[..., 0], W[..., 1], W[..., 2]
    g1 = (np.tanh(2.0 * w1) + w2**2 + np.sin(np.pi * w3)) / 2.2
    g2 = (np.maximum(0.0, w1 - 0.5) + np.sin(np.pi * w2) + w3**2) / 2.1
    g3 = (w1 * w2 + np.tanh(w3) + np.cos(np.pi * w1)) / 1.7
    return np.stack([g1, g2, g3], axis=-1)


def make_oracle_dgp(
    n_obs: int = 3000,
    noise: float = 0.1,
    confounder_strength: float = 1.0,
    ar_rho: float = 0.5,
    cross_effect: float = 0.60,
    self_effect: float = 0.30,
    decay_lag2: int = 2000,
    decay_lag3: int = 1000,
    burn_in: int = 50,
    seed: int | None = 42,
    legacy_rng: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, ToyGroundTruth]:
    """Generate the paper's three-variable time-varying confounded benchmark.

    Reproduces §5.1 exactly. The lag-1 cycle ``Z -> X -> Y -> Z`` carries a fixed
    coefficient of 0.60; the lag-2 self-effects of ``X`` and ``Y`` and the lag-3
    self-effect of ``Z`` decay linearly to zero, creating three regimes in which
    the true maximum causal lag is 3, then 2, then 1. Three independent AR(1)
    confounders enter each equation through the nonlinear functions of eq. (12).

    Parameters
    ----------
    n_obs : int, default 3000
        Observations kept after burn-in; the paper's ``T``.
    noise : float, default 0.1
        The innovation scale ``nu``. **The paper states 0.1** and the benchmark
        data distributed with the authors' code was generated with 0.1, although
        the default inside their code is 0.05; this package follows the paper.
    confounder_strength : float, default 1.0
        The nuisance scale ``lambda``.
    ar_rho : float, default 0.5
        AR(1) coefficient of each confounder.
    cross_effect : float, default 0.60
        The lag-1 cycle coefficient.
    self_effect : float, default 0.30
        Initial value of each decaying self-effect.
    decay_lag2, decay_lag3 : int
        Dates at which the lag-2 and lag-3 self-effects reach zero.
    burn_in : int, default 50
        Discarded leading observations.
    seed : int or None, default 42
    legacy_rng : bool, default True
        Draw with :class:`numpy.random.RandomState`, which is what the authors'
        generator uses. With the default, this function reproduces the benchmark
        data distributed with the paper to floating-point precision (verified at
        ``max|difference| = 9.6e-16`` on all 3,000 rows of both ``Y`` and ``W``),
        so the paper's own benchmark can be regenerated here exactly. Set
        ``False`` to draw from the modern :func:`numpy.random.default_rng`, which
        is preferable for new simulation work but gives different numbers.

    Returns
    -------
    (Y, W, truth) : tuple
        ``Y`` has columns ``X``, ``Y``, ``Z``; ``W`` has ``W1``, ``W2``, ``W3``;
        ``truth`` is a :class:`ToyGroundTruth`.

    Notes
    -----
    The confounders are drawn once and shared across the three equations, as the
    paper specifies: anything the design defines as common across series must be
    generated once, never redrawn per equation, or the "no-structure" branch of a
    test becomes a test of the wrong thing.

    Examples
    --------
    >>> Y, W, truth = make_oracle_dgp(n_obs=300, seed=0)
    >>> Y.shape, W.shape, truth.A.shape
    ((300, 3), (300, 3), (300, 3, 3, 3))
    >>> truth.p_true[0], truth.p_true[-1]
    (3, 1)
    >>> float(truth.A[0, 0, 0, 2])          # Z at lag 1 -> X
    0.6
    """
    if n_obs < 10:
        raise ValueError(f"n_obs must be >= 10, got {n_obs}")
    if noise <= 0:
        raise ValueError(f"noise must be > 0, got {noise}")

    total = n_obs + burn_in
    max_lag = 3

    # The draw order matters: the confounder innovations are drawn as one block
    # before the structural innovations, which is what makes the legacy branch
    # reproduce the paper's distributed benchmark exactly.
    if legacy_rng:
        rng = np.random.RandomState(seed)
        eta = rng.randn(total, 3)
    else:
        generator = np.random.default_rng(seed)
        eta = generator.standard_normal((total, 3))

    # One shared draw of the confounders: they drive every equation.
    W_all = np.zeros((total, 3))
    for t in range(1, total):
        W_all[t] = ar_rho * W_all[t - 1] + eta[t]

    eps = rng.randn(total, 3) if legacy_rng else generator.standard_normal((total, 3))
    Y_all = np.zeros((total, 3))
    nu_lambda = noise * confounder_strength

    for t in range(max_lag, total):
        date = max(0, t - burn_in)
        a_xx = self_effect * max(0.0, 1.0 - date / decay_lag2)
        a_yy = self_effect * max(0.0, 1.0 - date / decay_lag2)
        a_zz = self_effect * max(0.0, 1.0 - date / decay_lag3)
        g = _g_nonlinear(W_all[t - 1])

        Y_all[t, 0] = (
            cross_effect * Y_all[t - 1, 2]
            + a_xx * Y_all[t - 2, 0]
            + nu_lambda * g[0]
            + noise * eps[t, 0]
        )
        Y_all[t, 1] = (
            cross_effect * Y_all[t - 1, 0]
            + a_yy * Y_all[t - 2, 1]
            + nu_lambda * g[1]
            + noise * eps[t, 1]
        )
        Y_all[t, 2] = (
            cross_effect * Y_all[t - 1, 1]
            + a_zz * Y_all[t - 3, 2]
            + nu_lambda * g[2]
            + noise * eps[t, 2]
        )

    Y = Y_all[burn_in:]
    W = W_all[burn_in:]

    dates = np.arange(n_obs, dtype=float)
    A = np.zeros((n_obs, 3, 3, 3))
    A[:, 0, 0, 2] = cross_effect  # Z_{t-1} -> X_t
    A[:, 0, 1, 0] = cross_effect  # X_{t-1} -> Y_t
    A[:, 0, 2, 1] = cross_effect  # Y_{t-1} -> Z_t
    A[:, 1, 0, 0] = self_effect * np.maximum(0.0, 1.0 - dates / decay_lag2)
    A[:, 1, 1, 1] = self_effect * np.maximum(0.0, 1.0 - dates / decay_lag2)
    A[:, 2, 2, 2] = self_effect * np.maximum(0.0, 1.0 - dates / decay_lag3)

    nonzero = np.abs(A) > 0
    p_true = np.where(nonzero.any(axis=(2, 3)).any(axis=1), 0, 0)
    deepest = np.zeros(n_obs, dtype=int)
    for k in range(3):
        deepest = np.where(nonzero[:, k].any(axis=(1, 2)), k + 1, deepest)
    p_true = deepest

    names = ["X", "Y", "Z"]
    confounder_names = ["W1", "W2", "W3"]
    # Clip the regime boundaries to the sample: with a short n_obs the later
    # regimes may not be reached at all, and an empty regime is not a regime.
    bounds = [0, min(decay_lag3, n_obs), min(decay_lag2, n_obs), n_obs]
    regimes = tuple(
        (lo, hi) for lo, hi in zip(bounds, bounds[1:]) if hi > lo
    )
    truth = ToyGroundTruth(
        A=A,
        p_true=p_true,
        regimes=regimes,
        names=names,
        confounder_names=confounder_names,
    )
    return (
        pd.DataFrame(Y, columns=names),
        pd.DataFrame(W, columns=confounder_names),
        truth,
    )


def observed_confounders(
    W: pd.DataFrame | np.ndarray, level: str = "all"
) -> pd.DataFrame | np.ndarray:
    """Slice the confounders to one of the paper's observability levels.

    The data is always generated by every confounder; the level says how many of
    them the method is given. Below ``"all"``, causal sufficiency fails: the edges
    stop being causal and the FDR guarantee lapses (§4, Appendix B.1). The paper
    reports these levels to test robustness outside its own assumptions.

    Parameters
    ----------
    W : pandas.DataFrame or numpy.ndarray
        The full confounder set.
    level : {"all", "partial_2", "partial_1", "none"}

    Returns
    -------
    Same type as ``W``, with the retained columns.

    Examples
    --------
    >>> _, W, _ = make_oracle_dgp(n_obs=60, seed=0)
    >>> observed_confounders(W, "partial_1").shape
    (60, 1)
    >>> observed_confounders(W, "none").shape
    (60, 0)
    """
    keep = {"all": None, "partial_2": 2, "partial_1": 1, "none": 0}
    if level not in keep:
        raise ValueError(
            f"unknown observability level {level!r}; choose one of {sorted(keep)}"
        )
    n_keep = keep[level]
    if n_keep is None:
        return W
    if isinstance(W, pd.DataFrame):
        return W.iloc[:, :n_keep]
    return W[:, :n_keep]


def load_treasury(as_frame: bool = True):
    """Load the real dataset shipped with the package.

    Daily changes, in basis points, of the US Treasury par yield curve at nine
    maturities (the endogenous system ``Y``) together with the real (TIPS) par
    yield curve at five maturities (the observed confounders ``W``). The real
    curve carries the common real-rate factors that move every nominal maturity at
    once; left unadjusted, exactly the sort of common driver that manufactures
    spurious lagged links, which is what the paper's §5.2 attributes its own
    weakest results to.

    Parameters
    ----------
    as_frame : bool, default True
        Return pandas objects. ``False`` returns ``(Y, W)`` as arrays.

    Returns
    -------
    (Y, W) : tuple
        ``Y`` with nine columns, ``W`` with five, indexed by trading date.

    Notes
    -----
    Source: U.S. Department of the Treasury, Daily Treasury Par Yield Curve Rates
    and Daily Treasury Par Real Yield Curve Rates. A work of the U.S. federal
    government: public domain. See ``datasets/SOURCES.md`` for the exact pull.

    Examples
    --------
    >>> Y, W = load_treasury()
    >>> Y.shape[1], W.shape[1]
    (9, 5)
    >>> bool((Y.index == W.index).all())
    True
    """
    with resources.as_file(
        resources.files("oracle_varx.datasets") / "treasury_curve.csv.gz"
    ) as path:
        frame = pd.read_csv(path, index_col=0, parse_dates=True, compression="gzip")
    confounders = [c for c in frame.columns if c.startswith("real_")]
    endogenous = [c for c in frame.columns if c not in confounders]
    Y, W = frame[endogenous], frame[confounders]
    if as_frame:
        return Y, W
    return Y.to_numpy(), W.to_numpy()
