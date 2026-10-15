"""Results objects: the causal lag graph and everything read off a fitted model.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from ._config import OracleVARXConfig

__all__ = ["CausalLagGraph", "OracleVARXResults"]


@dataclass
class CausalLagGraph:
    """The lag-annotated directed graph discovered in one window (Definition 1).

    Vertices are the series; an edge ``(j -> i, k)`` means series ``j`` at lag
    ``k`` Granger-causes series ``i``. Under causal sufficiency — every common
    cause of the series observed in ``W`` — a zero coefficient is equivalent to
    Granger non-causality (Proposition A.13). When a confounder is hidden, the
    coefficients estimate the best linear lead-lag predictor from the observed
    series instead, and the edges are predictive, not causal.

    Attributes
    ----------
    adjacency : numpy.ndarray
        ``(p_max, n, n)`` boolean, ``adjacency[k - 1, i, j]`` for edge
        ``(j -> i, k)``.
    coef, se, qvalues : numpy.ndarray
        Same shape, the estimates behind the selection.
    names : list of str
    lag : int
        The selected lag order; lags beyond it were never tested.
    label : str
        What the graph is dated by, for printing.
    """

    adjacency: np.ndarray
    coef: np.ndarray
    se: np.ndarray
    qvalues: np.ndarray
    names: list[str]
    lag: int
    label: str = ""

    @property
    def n_edges(self) -> int:
        """Number of discovered edges."""
        return int(self.adjacency.sum())

    def to_frame(self) -> pd.DataFrame:
        """The edges as a tidy frame, strongest first.

        Returns
        -------
        pandas.DataFrame
            Columns ``from``, ``to``, ``lag``, ``coef``, ``se``, ``z``, ``qvalue``.

        Examples
        --------
        >>> import numpy as np
        >>> g = CausalLagGraph(
        ...     adjacency=np.array([[[False, True], [False, False]]]),
        ...     coef=np.array([[[0.0, 0.6], [0.0, 0.0]]]),
        ...     se=np.array([[[0.1, 0.1], [0.1, 0.1]]]),
        ...     qvalues=np.array([[[1.0, 0.001], [1.0, 1.0]]]),
        ...     names=["X", "Y"], lag=1)
        >>> g.to_frame()[["from", "to", "lag", "coef"]]
          from to  lag  coef
        0    Y  X    1   0.6
        """
        lags, rows, cols = np.nonzero(self.adjacency)
        frame = pd.DataFrame(
            {
                "from": [self.names[j] for j in cols],
                "to": [self.names[i] for i in rows],
                "lag": lags + 1,
                "coef": self.coef[lags, rows, cols],
                "se": self.se[lags, rows, cols],
                "qvalue": self.qvalues[lags, rows, cols],
            }
        )
        frame["z"] = np.where(frame["se"] > 0, frame["coef"] / frame["se"], np.nan)
        frame = frame[["from", "to", "lag", "coef", "se", "z", "qvalue"]]
        order = frame["coef"].abs().sort_values(ascending=False).index
        return frame.loc[order].reset_index(drop=True)

    def adjacency_frame(self, lag: int = 1) -> pd.DataFrame:
        """The adjacency matrix at one lag, as a labelled frame.

        Examples
        --------
        >>> import numpy as np
        >>> g = CausalLagGraph(np.zeros((1, 2, 2), bool), np.zeros((1, 2, 2)),
        ...                    np.ones((1, 2, 2)), np.ones((1, 2, 2)), ["X", "Y"], 1)
        >>> g.adjacency_frame().shape
        (2, 2)
        """
        if not 1 <= lag <= self.adjacency.shape[0]:
            raise ValueError(f"lag must lie in 1..{self.adjacency.shape[0]}")
        return pd.DataFrame(
            self.adjacency[lag - 1].astype(int),
            index=pd.Index(self.names, name="to"),
            columns=pd.Index(self.names, name="from"),
        )

    def __str__(self) -> str:
        head = f"Causal lag graph{f' ({self.label})' if self.label else ''}"
        body = self.to_frame()
        return (
            f"{head}\n"
            f"  selected lag order : {self.lag}\n"
            f"  discovered edges   : {self.n_edges}\n"
            + (body.to_string(index=False) if self.n_edges else "  (no edges)")
        )

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return self.__str__()


@dataclass
class OracleVARXResults:
    """Everything a fitted ORACLE-VARX model carries.

    Attributes
    ----------
    coef, se : numpy.ndarray
        ``(B, p_max, n, n)`` coefficients and eq.-(8) standard errors from the
        ``p_max`` fit, per output day.
    edges : numpy.ndarray
        ``(B, p_max, n, n)`` boolean, the BH-selected edges.
    qvalues : numpy.ndarray
        ``(B, p_max, n, n)`` BH-adjusted p-values; an addition to the paper.
    lag : numpy.ndarray
        ``(B,)`` selected lag order per output day.
    alpha : numpy.ndarray or None
        ``(B,)`` significance level chosen by Stage 2, for the ACLE rungs.
    forecasts, actuals : numpy.ndarray
        ``(B, n)``.
    days : numpy.ndarray
        ``(B,)`` row index of each output day in the input data.
    index : pandas.Index or None
        Dates of the output days, when the input carried them.
    names, confounder_names : list of str
    method, learner : str
    config : OracleVARXConfig
    dof : int
    n_fits : int
    variance : str
        How the standard errors were computed, for the table footer.
    """

    coef: np.ndarray
    se: np.ndarray
    edges: np.ndarray
    qvalues: np.ndarray
    lag: np.ndarray
    forecasts: np.ndarray
    actuals: np.ndarray
    days: np.ndarray
    names: list[str]
    config: OracleVARXConfig
    method: str = "oracle-varx"
    learner: str = "extra_trees"
    alpha: np.ndarray | None = None
    index: pd.Index | None = None
    confounder_names: list[str] = field(default_factory=list)
    dof: int = 0
    n_fits: int = 0
    variance: str = "classical (eq. 8), conditional homoskedasticity"
    select: np.ndarray | None = None

    # ------------------------------------------------------------------ shapes
    @property
    def n_windows(self) -> int:
        return self.coef.shape[0]

    @property
    def n_series(self) -> int:
        return len(self.names)

    @property
    def p_max(self) -> int:
        return self.coef.shape[1]

    @property
    def z(self) -> np.ndarray:
        """Entry-wise z-statistics, ``(B, p_max, n, n)``."""
        out = np.zeros_like(self.coef)
        good = (self.se > 0) & np.isfinite(self.se)
        out[good] = self.coef[good] / self.se[good]
        return out

    #: Alias, for users coming from statsmodels.
    @property
    def tvalues(self) -> np.ndarray:
        return self.z

    @property
    def pvalues(self) -> np.ndarray:
        """Two-sided normal p-values, ``(B, p_max, n, n)`` (Appendix D.4)."""
        return 2.0 * norm.sf(np.abs(self.z))

    def conf_int(self, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Confidence bounds for every coefficient, from Theorem 1.

        The paper lists a confidence interval per lag coefficient as one of its
        contributions but prints none; Theorem 1 delivers it directly.

        Parameters
        ----------
        level : float, default 0.95

        Returns
        -------
        (lower, upper) : tuple of numpy.ndarray
            Each ``(B, p_max, n, n)``.

        Examples
        --------
        >>> import numpy as np
        >>> from oracle_varx import OracleVARXConfig
        >>> res = OracleVARXResults(
        ...     coef=np.full((1, 1, 1, 1), 0.5), se=np.full((1, 1, 1, 1), 0.1),
        ...     edges=np.ones((1, 1, 1, 1), bool), qvalues=np.zeros((1, 1, 1, 1)),
        ...     lag=np.array([1]), forecasts=np.zeros((1, 1)),
        ...     actuals=np.zeros((1, 1)), days=np.array([0]), names=["X"],
        ...     config=OracleVARXConfig.synthetic())
        >>> lower, upper = res.conf_int(0.95)
        >>> float(np.round(lower[0, 0, 0, 0], 3)), float(np.round(upper[0, 0, 0, 0], 3))
        (0.304, 0.696)
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie in (0, 1), got {level}")
        crit = norm.ppf(0.5 + level / 2.0)
        return self.coef - crit * self.se, self.coef + crit * self.se

    # ------------------------------------------------------------------ tidying
    def _labels(self) -> np.ndarray:
        if self.index is not None:
            return np.asarray(self.index)
        return self.days

    def to_frame(self, window: int | None = None, tested_only: bool = True) -> pd.DataFrame:
        """Coefficients as a tidy frame: one row per ``(day, lag, from, to)``.

        Parameters
        ----------
        window : int, optional
            Position of a single output day. ``None`` returns every day, which
            for a long sample is a large frame.
        tested_only : bool, default True
            Keep only entries at lags within that day's selected lag order — the
            entries that were actually tested.

        Returns
        -------
        pandas.DataFrame
            Columns ``day``, ``lag``, ``from``, ``to``, ``coef``, ``se``, ``z``,
            ``pvalue``, ``qvalue``, ``edge``.
        """
        windows = range(self.n_windows) if window is None else [window]
        labels = self._labels()
        z_all, p_all = self.z, self.pvalues
        rows = []
        for w in windows:
            p_hat = int(self.lag[w])
            depth = p_hat if tested_only else self.p_max
            for k in range(depth):
                for i in range(self.n_series):
                    for j in range(self.n_series):
                        rows.append(
                            (
                                labels[w],
                                k + 1,
                                self.names[j],
                                self.names[i],
                                self.coef[w, k, i, j],
                                self.se[w, k, i, j],
                                z_all[w, k, i, j],
                                p_all[w, k, i, j],
                                self.qvalues[w, k, i, j],
                                bool(self.edges[w, k, i, j]),
                            )
                        )
        return pd.DataFrame(
            rows,
            columns=[
                "day", "lag", "from", "to", "coef", "se", "z", "pvalue",
                "qvalue", "edge",
            ],
        )

    def lag_series(self) -> pd.Series:
        """The selected lag order over time, ``p_hat_t``.

        Examples
        --------
        >>> import numpy as np
        >>> from oracle_varx import OracleVARXConfig
        >>> res = OracleVARXResults(
        ...     coef=np.zeros((2, 1, 1, 1)), se=np.ones((2, 1, 1, 1)),
        ...     edges=np.zeros((2, 1, 1, 1), bool), qvalues=np.ones((2, 1, 1, 1)),
        ...     lag=np.array([1, 2]), forecasts=np.zeros((2, 1)),
        ...     actuals=np.zeros((2, 1)), days=np.array([5, 6]), names=["X"],
        ...     config=OracleVARXConfig.synthetic())
        >>> res.lag_series().tolist()
        [1, 2]
        """
        return pd.Series(self.lag, index=self._labels(), name="p_hat")

    def alpha_series(self) -> pd.Series | None:
        """The Stage-2 significance level over time, or ``None`` for a non-ACLE fit."""
        if self.alpha is None:
            return None
        return pd.Series(self.alpha, index=self._labels(), name="alpha")

    def graph(self, window: int = -1) -> CausalLagGraph:
        """The causal lag graph discovered in one window.

        Parameters
        ----------
        window : int, default -1
            Position of the output day; ``-1`` is the last.

        Returns
        -------
        CausalLagGraph
        """
        w = range(self.n_windows)[window]
        label = str(self._labels()[w])
        return CausalLagGraph(
            adjacency=self.edges[w],
            coef=self.coef[w],
            se=self.se[w],
            qvalues=self.qvalues[w],
            names=list(self.names),
            lag=int(self.lag[w]),
            label=label,
        )

    def edge_persistence(self) -> pd.DataFrame:
        """How often each edge survives BH — an addition to the paper.

        The paper reports a sequence of graphs and plots the lag order over time,
        but never summarises how stable the individual edges are. The natural
        summary of a sequence of graphs is the share of windows in which each
        ``(j -> i, k)`` is discovered.

        Returns
        -------
        pandas.DataFrame
            Columns ``from``, ``to``, ``lag``, ``share``, ``n_windows``,
            ``mean_coef``, sorted by ``share``.
        """
        share = self.edges.mean(axis=0)
        counts = self.edges.sum(axis=0)
        with np.errstate(invalid="ignore"):
            mean_coef = np.where(
                counts > 0,
                (self.coef * self.edges).sum(axis=0) / np.maximum(counts, 1),
                np.nan,
            )
        rows = []
        for k in range(self.p_max):
            for i in range(self.n_series):
                for j in range(self.n_series):
                    if counts[k, i, j] == 0:
                        continue
                    rows.append(
                        (
                            self.names[j],
                            self.names[i],
                            k + 1,
                            float(share[k, i, j]),
                            int(counts[k, i, j]),
                            float(mean_coef[k, i, j]),
                        )
                    )
        frame = pd.DataFrame(
            rows, columns=["from", "to", "lag", "share", "n_windows", "mean_coef"]
        )
        return frame.sort_values("share", ascending=False).reset_index(drop=True)

    def forecast_frame(self) -> pd.DataFrame:
        """Forecasts, realisations and errors on the output days."""
        labels = self._labels()
        out = pd.DataFrame(self.forecasts, index=labels, columns=self.names)
        actual = pd.DataFrame(self.actuals, index=labels, columns=self.names)
        return pd.concat({"forecast": out, "actual": actual, "error": actual - out}, axis=1)

    # ------------------------------------------------------------------- output
    def summary(self, window: int = -1, level: float = 0.95):
        """The journal table for one output day.

        Parameters
        ----------
        window : int, default -1
            Position of the output day.
        level : float, default 0.95

        Returns
        -------
        Summary
            Prints as an aligned text table; also carries ``as_text()``,
            ``as_latex()`` and ``as_markdown()``.
        """
        from ._tables import build_summary

        return build_summary(self, window=window, level=level)

    def to_latex(self, window: int = -1, **kwargs) -> str:
        """The same table as a booktabs LaTeX table."""
        from ._tables import to_latex

        return to_latex(self, window=window, **kwargs)

    def to_markdown(self, window: int = -1, **kwargs) -> str:
        """The same table as Markdown."""
        from ._tables import to_markdown

        return to_markdown(self, window=window, **kwargs)

    # -------------------------------------------------------------------- plots
    def plot_lag_tracking(self, **kwargs):
        """Selected lag order over time; the paper's Figure 4."""
        from ._plots import plot_lag_tracking

        return plot_lag_tracking(self, **kwargs)

    def plot_coefficient_paths(self, **kwargs):
        """Estimated edge coefficients over time; the paper's Figure 3."""
        from ._plots import plot_coefficient_paths

        return plot_coefficient_paths(self, **kwargs)

    def plot_edge_heatmap(self, **kwargs):
        """Parula heatmap of the z-statistics through time; an addition."""
        from ._plots import plot_edge_heatmap

        return plot_edge_heatmap(self, **kwargs)

    def plot_graph(self, **kwargs):
        """The discovered edges as a network; an addition."""
        from ._plots import plot_graph

        return plot_graph(self, **kwargs)

    def plot_surface(self, **kwargs):
        """Interactive 3-D surface over (lag, time); an addition, needs plotly."""
        from ._plots import plot_surface

        return plot_surface(self, **kwargs)

    def __str__(self) -> str:
        name = {
            "var": "VAR", "varx": "VARX", "acle-var": "ACLE-VAR",
            "acle-varx": "ACLE-VARX", "or-varx": "OR-VARX",
            "oracle-varx": "ORACLE-VARX",
        }.get(self.method, self.method)
        lag = self.lag
        lines = [
            f"{name} fit",
            f"  output days          : {self.n_windows}",
            f"  series               : {self.n_series} ({', '.join(self.names)})",
            f"  confounders          : {len(self.confounder_names)}",
            f"  selected lag p_hat   : mean {lag.mean():.2f}, "
            f"median {int(np.median(lag))}, range {lag.min()}-{lag.max()}",
            f"  edges per window     : {self.edges.sum(axis=(1, 2, 3)).mean():.2f}",
            f"  first-stage fits     : {self.n_fits}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return self.__str__()
