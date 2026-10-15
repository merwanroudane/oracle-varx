"""First stage: cross-fitted nuisance estimation and residualisation.

This is one of the few parts of the method that no library provides. The obvious
delegate, :func:`sklearn.model_selection.cross_val_predict`, refuses the split the
paper needs — ``ValueError: cross_val_predict only works for partitions`` — because
:class:`~sklearn.model_selection.TimeSeriesSplit` leaves the first training block in
no test fold. So the out-of-sample residuals of eq. (6) are assembled here.

The memoisation of §3.2 and Appendix D.3 lives here too: because the blocks sit on
one grid of rows that does not depend on the rolling window, each block's nuisance
models are fitted **once** and reused by every window that overlaps it. The paper
measures this at 203x-466x fewer first-stage fits than refitting per window, and the
reuse is exact — recomputing inside each window would give the same numbers.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.multioutput import MultiOutputRegressor

from ._utils import Fold, build_stacks

__all__ = ["NuisanceFit", "make_learner", "cross_fit", "LEARNERS"]

#: First-stage learners of Table 10, Panel B, by the names the paper uses.
LEARNERS: tuple[str, ...] = (
    "extra_trees",
    "random_forest",
    "lightgbm",
    "xgboost",
    "tabpfn",
    "linear",
)


@dataclass
class NuisanceFit:
    """Out-of-sample residuals and nuisance predictions for one lag order.

    Attributes
    ----------
    first_row : int
        Absolute row index of the first residual row.
    residual_Y : numpy.ndarray
        ``(n_rows, n)`` — the ``Y_tilde`` of eq. (6).
    residual_T : numpy.ndarray
        ``(n_rows, n * p)`` — the ``T_tilde`` of eq. (6).
    pred_Y : numpy.ndarray
        ``(n_rows, n)`` — ``g_hat_Y(C_t)``, reused for the forecast of Algorithm 1
        line 7 rather than recomputed.
    pred_T : numpy.ndarray
        ``(n_rows, n * p)`` — ``g_hat_T(C_t)``.
    n_fits : int
        First-stage fits actually performed, for the cost report of Table 11.
    """

    first_row: int
    residual_Y: np.ndarray
    residual_T: np.ndarray
    pred_Y: np.ndarray
    pred_T: np.ndarray
    n_fits: int

    @property
    def n_rows(self) -> int:
        return self.residual_Y.shape[0]

    @property
    def last_row(self) -> int:
        """One past the last residual row, in absolute terms."""
        return self.first_row + self.n_rows


def make_learner(name: str | Any = "extra_trees", seed: int | None = 42, n_jobs: int = 1):
    """Build a first-stage learner with the paper's Table 10, Panel B settings.

    Parameters
    ----------
    name : str or estimator, default ``"extra_trees"``
        One of :data:`LEARNERS`, or any scikit-learn-compatible regressor, which is
        returned as given. ``"linear"`` is not in the paper; it is a fast
        least-squares first stage, useful for tests and for the linear VARX of
        eq. (1).
    seed : int or None, default 42
        Random state. **This is an addition to the paper.** The authors' code sets
        no random state on the tree learners, so Extra Trees and Random Forest
        results there are not reproducible run to run; seeding makes them so, at
        the cost of not matching the paper's tables digit for digit.
    n_jobs : int, default 1
        Threads inside the learner. Parallelism over folds is handled by
        :func:`cross_fit`, so leave this at 1 unless running a single fold.

    Returns
    -------
    estimator
        An unfitted regressor with ``fit`` and ``predict``.

    Raises
    ------
    ValueError
        If ``name`` is an unknown string.
    ImportError
        If the named learner's package is not installed.

    Examples
    --------
    >>> est = make_learner("extra_trees", seed=0)
    >>> est.get_params()["max_depth"]
    5
    >>> make_learner("linear").__class__.__name__
    'LinearRegression'
    """
    if not isinstance(name, str):
        return name

    key = name.strip().lower()
    if key in {"extra_trees", "et", "extratrees"}:
        from sklearn.ensemble import ExtraTreesRegressor

        return ExtraTreesRegressor(
            n_estimators=100,
            max_features=1.0,
            max_depth=5,
            random_state=seed,
            n_jobs=n_jobs,
        )
    if key in {"random_forest", "rf"}:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=100,
            max_features=1.0,
            max_depth=5,
            random_state=seed,
            n_jobs=n_jobs,
        )
    if key in {"lightgbm", "lgbm"}:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                'the "lightgbm" learner needs: pip install "oracle-varx[boost]"'
            ) from exc

        return LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            num_leaves=31,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
    if key in {"xgboost", "xgb"}:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                'the "xgboost" learner needs: pip install "oracle-varx[boost]"'
            ) from exc

        return XGBRegressor(
            n_estimators=100,
            learning_rate=0.3,
            max_depth=5,
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
        )
    if key == "tabpfn":
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                'the "tabpfn" learner needs: pip install "oracle-varx[tabpfn]", a '
                "Hugging Face token, and realistically a GPU"
            ) from exc

        return TabPFNRegressor(random_state=seed)
    if key in {"linear", "ols"}:
        from sklearn.linear_model import LinearRegression

        return LinearRegression()

    raise ValueError(
        f"unknown learner {name!r}; choose one of {LEARNERS} or pass an estimator"
    )


def _fit_one_fold(
    fold: Fold,
    Y: np.ndarray,
    W: np.ndarray,
    p: int,
    learner,
    seed_value: int,
) -> tuple[Fold, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Fit one block's nuisance models and return its out-of-sample pieces."""
    train_Y, train_T, train_C = build_stacks(
        Y, W, p, fold.train_start, fold.train_end, allow_history=False
    )
    test_Y, test_T, test_C = build_stacks(
        Y, W, p, fold.test_start, fold.test_end, allow_history=True
    )

    model_y = MultiOutputRegressor(clone(learner), n_jobs=1)
    model_t = MultiOutputRegressor(clone(learner), n_jobs=1)
    for model in (model_y, model_t):
        for key in ("random_state", "seed"):
            if key in model.estimator.get_params():
                model.estimator.set_params(**{key: seed_value})
                break

    model_y.fit(train_C, train_Y)
    model_t.fit(train_C, train_T)

    pred_Y = np.asarray(model_y.predict(test_C), dtype=np.float64)
    pred_T = np.asarray(model_t.predict(test_C), dtype=np.float64)
    if pred_Y.ndim == 1:
        pred_Y = pred_Y.reshape(-1, 1)
    if pred_T.ndim == 1:
        pred_T = pred_T.reshape(-1, 1)

    # One fit per output: n for the outcome, n * p for the stacked treatment.
    n_fits = train_Y.shape[1] + train_T.shape[1]
    return fold, test_Y - pred_Y, test_T - pred_T, pred_Y, pred_T, n_fits


def cross_fit(
    Y: np.ndarray,
    W: np.ndarray,
    p: int,
    folds: Sequence[Fold],
    learner: str | Any = "extra_trees",
    seed: int | None = 42,
    n_jobs: int = 1,
) -> NuisanceFit:
    """Residualise the outcome and the stacked lags on the confounders, eq. (6).

    For each blocked fold, the learners ``g_hat_Y`` and ``g_hat_T`` are fitted on
    the fold's training rows and used to predict its test rows only, so every
    residual is out of sample. Fitting happens once per block: windows that
    overlap the block reuse the stored residual rows.

    Parameters
    ----------
    Y : numpy.ndarray
        Endogenous series ``(n_obs, n)``.
    W : numpy.ndarray
        Confounders ``(n_obs, d)`` with ``d >= 1``.
    p : int
        Lag order.
    folds : sequence of Fold
        From :func:`oracle_varx._utils.blocked_folds`.
    learner : str or estimator, default ``"extra_trees"``
        Passed to :func:`make_learner`.
    seed : int or None, default 42
        Base seed. Each fold receives an independent child seed spawned from it, so
        results never depend on ``n_jobs``.
    n_jobs : int, default 1
        Folds fitted in parallel.

    Returns
    -------
    NuisanceFit

    Raises
    ------
    ValueError
        If there are no confounders, or no folds.

    Notes
    -----
    The test ranges of distinct folds do not overlap, so residual rows are written
    exactly once and the result does not depend on the order folds complete in.

    Examples
    --------
    >>> import numpy as np
    >>> from oracle_varx._utils import blocked_folds
    >>> rng = np.random.default_rng(0)
    >>> W = rng.normal(size=(300, 2))
    >>> Y = np.tanh(W[:, :1]) + rng.normal(scale=0.1, size=(300, 2))
    >>> fit = cross_fit(Y, W, p=1, folds=blocked_folds(300, 200, 20, p_max=1),
    ...                 learner="linear")
    >>> fit.residual_Y.shape, fit.first_row
    ((80, 2), 201)
    """
    if W is None or W.shape[1] == 0:
        raise ValueError(
            "the orthogonalised methods need at least one confounder; with none, "
            'use method="var" or method="acle-var", which the paper reports as the '
            '"none" observability level'
        )
    if not folds:
        raise ValueError("no folds to cross-fit over")

    base = make_learner(learner, seed=seed, n_jobs=1)
    child_seeds = (
        np.random.SeedSequence(seed).generate_state(len(folds))
        if seed is not None
        else [None] * len(folds)
    )

    results = Parallel(n_jobs=n_jobs, prefer="processes" if n_jobs != 1 else "threads")(
        delayed(_fit_one_fold)(
            fold, Y, W, p, base, int(child) % (2**31 - 1) if child is not None else None
        )
        for fold, child in zip(folds, child_seeds, strict=True)
    )

    first_row = min(f.test_start for f in folds)
    last_row = max(f.test_end for f in folds)
    n_rows = last_row - first_row
    n = Y.shape[1]

    residual_Y = np.zeros((n_rows, n), dtype=np.float64)
    residual_T = np.zeros((n_rows, n * p), dtype=np.float64)
    pred_Y = np.zeros((n_rows, n), dtype=np.float64)
    pred_T = np.zeros((n_rows, n * p), dtype=np.float64)
    n_fits = 0

    for fold, res_y, res_t, py, pt, fits in results:
        lo = fold.test_start - first_row
        hi = fold.test_end - first_row
        residual_Y[lo:hi] = res_y
        residual_T[lo:hi] = res_t
        pred_Y[lo:hi] = py
        pred_T[lo:hi] = pt
        n_fits += fits

    return NuisanceFit(
        first_row=first_row,
        residual_Y=residual_Y,
        residual_T=residual_T,
        pred_Y=pred_Y,
        pred_T=pred_T,
        n_fits=n_fits,
    )
