"""The public estimator: :class:`OracleVARX` and the :func:`oracle_varx` wrapper.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from ._config import OracleVARXConfig
from ._core import METHODS, acle_stage1, bh_edges, fit_windows, rmse_lag_selection, rolling_validation
from ._results import OracleVARXResults
from ._utils import admissible_window, as_matrix

__all__ = ["OracleVARX", "oracle_varx", "method_hierarchy"]


def method_hierarchy() -> pd.DataFrame:
    """Table 1 of the paper: the nested hierarchy of methods.

    Each row adds one capability to the row above, which is what lets the paper
    attribute its gains to individual components.

    Returns
    -------
    pandas.DataFrame

    Examples
    --------
    >>> method_hierarchy().loc["oracle-varx", "lag_selection"]
    'ACLE'
    """
    frame = pd.DataFrame(
        [
            {
                "method": key,
                "lag_selection": spec["lag"],
                "confounder_adjustment": spec["confounders"],
                "dml": bool(spec["dml"]),
            }
            for key, spec in METHODS.items()
        ]
    )
    return frame.set_index("method")


class OracleVARX:
    """Orthogonalised regression for adaptive causal lag estimation in a VARX.

    Implements Tan, Zanoria, Chen, Lyu and Cucuringu (2026),
    `doi:10.48550/arXiv.2610.05618 <https://doi.org/10.48550/arXiv.2610.05618>`_:
    double machine learning removes the confounders' nonlinear effects from the
    outcomes and the lagged series (§3.2), adaptive causal lag estimation picks
    the lag order in each rolling window by sequential significance testing tuned
    on validation forecasts (§3.3), and a Benjamini-Hochberg step selects directed
    edges at a target false discovery rate (§3.4).

    Parameters
    ----------
    config : OracleVARXConfig, optional
        Window sizes and the significance grid. Defaults to the paper's synthetic
        configuration; :meth:`OracleVARXConfig.financial` is its daily-finance one.
    method : str, default ``"oracle-varx"``
        A rung of Table 1: ``"var"``, ``"varx"``, ``"acle-var"``, ``"acle-varx"``,
        ``"or-varx"`` or ``"oracle-varx"``. See :func:`method_hierarchy`.
    learner : str or estimator, default ``"extra_trees"``
        First-stage learner for the DML rungs: ``"extra_trees"``,
        ``"random_forest"``, ``"lightgbm"``, ``"xgboost"``, ``"tabpfn"``,
        ``"linear"``, or any estimator with ``fit`` and ``predict``.
    q : float, optional
        Edge FDR level; overrides ``config.q``.
    correction : {"fdr_bh", "fdr_by"}, default ``"fdr_bh"``
        Multiplicity correction. ``"fdr_by"`` is an addition to the paper: the
        Benjamini-Yekutieli form, valid under arbitrary dependence between the
        entry-wise tests, which is the gap §4 of the paper concedes.
    seed : int or None, default 42
        Threaded through every learner and every parallel worker, so results never
        depend on ``n_jobs``.
    n_jobs : int, default 1
        Blocks fitted in parallel in the first stage.
    ridge : float, default 0.0
        Diagonal nudge on the Gram matrix before solving. 0 is the paper.
    coef_source : {"p_max", "p_hat"}, default ``"p_max"``
        Which fit the reported coefficients come from. Algorithm 1 lines 10 and 14
        test the ``p_max`` fit, so ``"p_max"`` is what the inference is built on
        and the default. ``"p_hat"`` reports each day's coefficients from the fit
        at its own selected lag order, which is what the paper's coefficient-error
        column and its Figure 3 show; it reproduces those more closely, at the
        cost that the reported coefficients are no longer the ones tested.
    verbose : bool, default False

    Attributes
    ----------
    results_ : OracleVARXResults
        Set by :meth:`fit`.

    Notes
    -----
    Below full observability of the confounders, causal sufficiency fails: the
    coefficients become the best linear lead-lag predictor from the observed
    series rather than direct causal effects, and the FDR guarantee lapses (§4).
    The package will still fit; the interpretation is the user's responsibility.

    Examples
    --------
    >>> from oracle_varx import OracleVARX, OracleVARXConfig
    >>> from oracle_varx.datasets import make_oracle_dgp
    >>> Y, W, truth = make_oracle_dgp(n_obs=400, seed=0)
    >>> cfg = OracleVARXConfig(p_max=3, ols_window=60, block=20, validation_days=10)
    >>> model = OracleVARX(cfg, method="acle-var")
    >>> res = model.fit(Y)
    >>> res.lag.min() >= 1
    True
    """

    def __init__(
        self,
        config: OracleVARXConfig | None = None,
        method: str = "oracle-varx",
        learner: str | Any = "extra_trees",
        q: float | None = None,
        correction: str = "fdr_bh",
        seed: int | None = 42,
        n_jobs: int = 1,
        ridge: float = 0.0,
        coef_source: str = "p_max",
        verbose: bool = False,
    ) -> None:
        if method not in METHODS:
            raise ValueError(
                f"unknown method {method!r}; choose one of {sorted(METHODS)}"
            )
        if coef_source not in {"p_max", "p_hat"}:
            raise ValueError(
                f'coef_source must be "p_max" or "p_hat", got {coef_source!r}'
            )
        if correction not in {"fdr_bh", "fdr_by"}:
            raise ValueError(
                f'correction must be "fdr_bh" or "fdr_by", got {correction!r}'
            )
        self.config = config or OracleVARXConfig()
        if q is not None:
            self.config = self.config.replace(q=q)
        self.method = method
        self.learner = learner
        self.correction = correction
        self.seed = seed
        self.n_jobs = n_jobs
        self.ridge = ridge
        self.coef_source = coef_source
        self.verbose = verbose
        self.results_: OracleVARXResults | None = None

    # ---------------------------------------------------------------------- fit
    def fit(self, Y, W=None) -> OracleVARXResults:
        """Run the full pipeline of Algorithm 1 over every rolling window.

        Parameters
        ----------
        Y : pandas.DataFrame, pandas.Series or array-like
            Endogenous series, ``(n_obs, n)``. Names and the index are preserved
            and carried into every table and figure.
        W : pandas.DataFrame, pandas.Series, array-like or None
            Exogenous confounders, ``(n_obs, d)``. Required for every rung except
            ``"var"`` and ``"acle-var"``.

        Returns
        -------
        OracleVARXResults

        Raises
        ------
        ValueError
            If the sample is too short for the configuration, if ``Y`` and ``W``
            disagree on length, or if a method that needs confounders gets none.
        """
        Y_values, names, index = as_matrix(Y, "Y")
        if W is None:
            W_values, confounder_names = np.empty((Y_values.shape[0], 0)), []
        else:
            W_values, confounder_names, _ = as_matrix(W, "W")
            if W_values.shape[0] != Y_values.shape[0]:
                raise ValueError(
                    f"Y has {Y_values.shape[0]} rows but W has {W_values.shape[0]}"
                )

        n_obs, n = Y_values.shape
        if n < 2:
            raise ValueError(
                f"Y must hold at least 2 series for a lead-lag graph, got {n}"
            )

        guard = admissible_window(n_obs, self.config.ols_window)
        if self.verbose:
            print(self.config.summary())
            print(guard["message"])

        fit, select = fit_windows(
            Y_values,
            W_values,
            self.config,
            method=self.method,
            learner=self.learner,
            seed=self.seed,
            n_jobs=self.n_jobs,
            ridge=self.ridge,
            verbose=self.verbose,
            store_per_p=self.coef_source == "p_hat",
        )

        spec = METHODS[self.method]
        v = self.config.validation_days

        if spec["lag"] == "ACLE":
            candidates = np.column_stack(
                [
                    acle_stage1(
                        fit.coef, fit.se, alpha=a, select=select,
                        method=self.correction,
                    )
                    for a in self.config.alpha_grid
                ]
            )
            chosen, lag_hat, _ = rolling_validation(
                fit.forecasts, fit.actuals, candidates, v, select=select
            )
            alpha_hat = np.asarray(self.config.alpha_grid, dtype=float)[chosen]
        else:
            lag_hat, _ = rmse_lag_selection(
                fit.forecasts, fit.actuals, v, select=select
            )
            alpha_hat = None

        keep = slice(v, None)
        days = fit.days[keep]
        if self.coef_source == "p_hat" and fit.coef_per_p is not None:
            rows = np.arange(lag_hat.size)
            coef_raw = fit.coef_per_p[keep][rows, lag_hat - 1]
            se_raw = fit.se_per_p[keep][rows, lag_hat - 1]
        else:
            coef_raw, se_raw = fit.coef[keep], fit.se[keep]
        coef = coef_raw[:, :, select][:, :, :, select]
        se = se_raw[:, :, select][:, :, :, select]

        edges, qvalues = bh_edges(
            coef, se, lag_hat, q=self.config.q, method=self.correction
        )

        forecasts = np.take_along_axis(
            fit.forecasts[keep][:, select, :], lag_hat[:, None, None] - 1, axis=2
        )[:, :, 0]
        actuals = fit.actuals[keep][:, select]

        self.results_ = OracleVARXResults(
            coef=coef,
            se=se,
            edges=edges,
            qvalues=qvalues,
            lag=lag_hat,
            alpha=alpha_hat,
            forecasts=forecasts,
            actuals=actuals,
            days=days,
            names=[names[i] for i in select],
            config=self.config,
            method=self.method,
            learner=self.learner if isinstance(self.learner, str) else "custom",
            index=None if index is None else index[days],
            confounder_names=confounder_names,
            dof=fit.dof,
            n_fits=fit.n_fits,
            variance=(
                "classical (eq. 8), conditional homoskedasticity"
                if self.correction == "fdr_bh"
                else "classical (eq. 8); Benjamini-Yekutieli correction"
            ),
            select=select,
        )
        return self.results_


def oracle_varx(
    Y,
    W=None,
    p_max: int = 5,
    method: str = "oracle-varx",
    learner: str | Any = "extra_trees",
    ols_window: int = 200,
    block: int = 20,
    validation_days: int = 20,
    q: float = 0.05,
    correction: str = "fdr_bh",
    seed: int | None = 42,
    n_jobs: int = 1,
    verbose: bool = False,
) -> OracleVARXResults:
    """Fit ORACLE-VARX in one call.

    The functional front door, for users who will never write a class. Every
    argument has the paper's value as its default.

    Parameters
    ----------
    Y : pandas.DataFrame or array-like
        Endogenous series ``(n_obs, n)``.
    W : pandas.DataFrame, array-like or None
        Confounders ``(n_obs, d)``.
    p_max : int, default 5
        Maximum lag order (eq. 2 needs ``p_max >= p*``).
    method : str, default ``"oracle-varx"``
        A rung of Table 1; see :func:`method_hierarchy`.
    learner : str or estimator, default ``"extra_trees"``
        First-stage learner.
    ols_window : int, default 200
        The second-stage rolling window ``T_w``.
    block : int, default 20
        Cross-fitting block length and refit interval.
    validation_days : int, default 20
        Trailing validation window ``v`` of eq. (9).
    q : float, default 0.05
        Edge FDR level.
    correction : {"fdr_bh", "fdr_by"}, default ``"fdr_bh"``
    seed : int or None, default 42
    n_jobs : int, default 1
    verbose : bool, default False

    Returns
    -------
    OracleVARXResults

    Examples
    --------
    >>> from oracle_varx import oracle_varx
    >>> from oracle_varx.datasets import make_oracle_dgp
    >>> Y, W, _ = make_oracle_dgp(n_obs=320, seed=0)
    >>> res = oracle_varx(Y, W, p_max=3, method="or-varx", learner="linear",
    ...                   ols_window=60, block=20, validation_days=10)
    >>> res.n_windows > 0
    True
    """
    config = OracleVARXConfig(
        p_max=p_max,
        ols_window=ols_window,
        block=block,
        validation_days=validation_days,
        q=q,
        lag_offset=p_max,
    )
    model = OracleVARX(
        config=config,
        method=method,
        learner=learner,
        correction=correction,
        seed=seed,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return model.fit(Y, W)
