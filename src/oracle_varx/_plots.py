"""Figures: the paper's own, and the additions.

One shared style, applied through ``plt.rc_context`` so nothing leaks into the
user's other figures; every function returns its ``Figure`` and ``Axes`` and none
calls ``plt.show()``. ``"Parula"`` is the default scale on every heatmap, contour
and surface, as in the R packages.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ._colors import bluered_colors, parula_cmap, parula_colors, resolve_colorscale

__all__ = [
    "JOURNAL_RC",
    "plot_lag_tracking",
    "plot_coefficient_paths",
    "plot_edge_heatmap",
    "plot_graph",
    "plot_lag_vs_volatility",
    "plot_surface",
    "plot_fdr_power_curve",
    "plot_residual_diagnostics",
    "plot_window_guard",
]

#: The shared figure style: sized for a journal column, no top or right spine.
JOURNAL_RC: dict[str, object] = {
    "figure.figsize": (6.5, 4.0),
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
}

_INDIGO = "#4f46e5"
_TEAL = "#0d9488"
_AMBER = "#f59e0b"
_ROSE = "#e11d48"
_INK = "#16202e"


def _axis_values(results):
    """Dates when the input carried them, row indices otherwise."""
    return results.index if results.index is not None else results.days


def plot_lag_tracking(
    results,
    p_true: np.ndarray | None = None,
    regimes: Sequence[tuple[int, int]] | None = None,
    ax=None,
    title: str | None = None,
):
    """The selected lag order over time, against the truth — the paper's Figure 4.

    Parameters
    ----------
    results : OracleVARXResults
    p_true : numpy.ndarray, optional
        ``(T,)`` the true lag order at every date, drawn as a black step function.
    regimes : sequence of (int, int), optional
        Regime boundaries to mark with dashed vertical lines.
    ax : matplotlib.axes.Axes, optional
    title : str, optional

    Returns
    -------
    (fig, ax) : tuple

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from oracle_varx import OracleVARX, OracleVARXConfig
    >>> from oracle_varx.datasets import make_oracle_dgp
    >>> Y, W, truth = make_oracle_dgp(n_obs=400, seed=0)
    >>> cfg = OracleVARXConfig(p_max=3, ols_window=60, block=20, validation_days=10)
    >>> res = OracleVARX(cfg, method="acle-var").fit(Y)
    >>> fig, ax = plot_lag_tracking(res, truth.p_true, truth.regimes)
    >>> ax.get_ylabel()
    'lag order'
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(JOURNAL_RC):
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
        x = _axis_values(results)
        ax.step(
            x, results.lag, where="post", color=_INDIGO, label=r"selected $\hat{p}_t$"
        )
        if p_true is not None:
            ax.step(
                x,
                np.asarray(p_true)[results.days],
                where="post",
                color=_INK,
                linestyle="--",
                linewidth=1.2,
                label=r"true $p^*(t)$",
            )
        if regimes is not None and results.index is None:
            for _, end in list(regimes)[:-1]:
                ax.axvline(end, color="0.5", linestyle=":", linewidth=0.9)
        ax.set_xlabel("time" if results.index is None else "date")
        ax.set_ylabel("lag order")
        ax.set_yticks(np.arange(0, results.p_max + 1))
        ax.set_title(
            title or "Adaptive lag order over time", loc="left"
        )
        ax.legend(loc="upper right", ncols=2)
    return fig, ax


def plot_coefficient_paths(
    results,
    A_true: np.ndarray | None = None,
    edges: Sequence[tuple[str, str, int]] | None = None,
    regimes: Sequence[tuple[int, int]] | None = None,
    level: float | None = 0.95,
    max_panels: int = 6,
    ncols: int = 3,
):
    """Estimated edge coefficients over time — the paper's Figure 3.

    One panel per edge: the estimate, its confidence band (an addition to the
    paper, which plots the point estimate alone), and the truth when it is known.

    Parameters
    ----------
    results : OracleVARXResults
    A_true : numpy.ndarray, optional
        ``(T, p_true, n, n)`` the known truth.
    edges : sequence of (from, to, lag), optional
        Which edges to draw. Defaults to the most persistently discovered ones.
    regimes : sequence of (int, int), optional
    level : float or None, default 0.95
        Confidence level for the band; ``None`` omits it.
    max_panels : int, default 6
    ncols : int, default 3

    Returns
    -------
    (fig, axes) : tuple
    """
    import matplotlib.pyplot as plt

    if edges is None:
        persistence = results.edge_persistence()
        if len(persistence):
            edges = [
                (row["from"], row["to"], int(row["lag"]))
                for _, row in persistence.head(max_panels).iterrows()
            ]
        else:
            edges = [(results.names[0], results.names[0], 1)]
    edges = list(edges)[:max_panels]

    lower, upper = results.conf_int(level) if level else (None, None)
    x = _axis_values(results)
    nrows = int(np.ceil(len(edges) / ncols))

    with plt.rc_context(JOURNAL_RC):
        fig, axes = plt.subplots(
            nrows,
            min(ncols, len(edges)),
            figsize=(3.1 * min(ncols, len(edges)), 2.3 * nrows),
            squeeze=False,
            sharex=True,
        )
        flat = axes.ravel()
        for panel, (src, dst, lag) in enumerate(edges):
            ax = flat[panel]
            j = results.names.index(src)
            i = results.names.index(dst)
            k = lag - 1
            ax.plot(x, results.coef[:, k, i, j], color=_TEAL, label="estimate")
            if lower is not None:
                ax.fill_between(
                    x,
                    lower[:, k, i, j],
                    upper[:, k, i, j],
                    color=_TEAL,
                    alpha=0.20,
                    linewidth=0,
                )
            if A_true is not None and k < A_true.shape[1]:
                ax.plot(
                    x,
                    A_true[results.days, k, i, j],
                    color=_INK,
                    linestyle="--",
                    linewidth=1.1,
                    label="truth",
                )
            ax.axhline(0.0, color="0.3", linewidth=0.8)
            if regimes is not None and results.index is None:
                for _, end in list(regimes)[:-1]:
                    ax.axvline(end, color="0.5", linestyle=":", linewidth=0.9)
            ax.set_title(f"{src} → {dst}, lag {lag}", loc="left")
        for ax in flat[len(edges) :]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("time" if results.index is None else "date")
        for row in axes:
            row[0].set_ylabel("coefficient")
        flat[0].legend(loc="upper right")
        fig.tight_layout()
    return fig, axes


def plot_edge_heatmap(
    results,
    lag: int = 1,
    statistic: str = "z",
    ax=None,
    colorscale: str = "Parula",
):
    """Heatmap of every edge's statistic through time — an addition to the paper.

    Rows are the ``n^2`` ordered pairs at one lag, columns are the output days, so
    the whole sequence of graphs is visible at once rather than one window at a
    time.

    Parameters
    ----------
    results : OracleVARXResults
    lag : int, default 1
    statistic : {"z", "coef", "qvalue"}, default ``"z"``
    ax : matplotlib.axes.Axes, optional
    colorscale : str, default ``"Parula"``
        ``"coef"`` is signed, so it is drawn on the diverging blue-red scale
        centred on zero whatever this says.

    Returns
    -------
    (fig, ax) : tuple
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if not 1 <= lag <= results.p_max:
        raise ValueError(f"lag must lie in 1..{results.p_max}")
    source = {"z": results.z, "coef": results.coef, "qvalue": results.qvalues}
    if statistic not in source:
        raise ValueError(f'statistic must be one of {sorted(source)}')

    values = source[statistic][:, lag - 1]
    labels, matrix = [], []
    for i in range(results.n_series):
        for j in range(results.n_series):
            labels.append(f"{results.names[j]}→{results.names[i]}")
            matrix.append(values[:, i, j])
    matrix = np.asarray(matrix)

    if statistic == "coef":
        cmap = LinearSegmentedColormap.from_list("bluered", bluered_colors(256))
        bound = float(np.nanmax(np.abs(matrix))) or 1.0
        vmin, vmax = -bound, bound
    else:
        cmap = parula_cmap() if colorscale.lower() == "parula" else plt.get_cmap("viridis")
        vmin = vmax = None

    with plt.rc_context(JOURNAL_RC):
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(
            figsize=(7.2, 0.28 * len(labels) + 1.6)
        )
        image = ax.imshow(
            matrix, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
            extent=(0, matrix.shape[1], -0.5, len(labels) - 0.5),
        )
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("output day")
        ax.set_title(
            f"{statistic} statistic at lag {lag}", loc="left"
        )
        ax.grid(False)
        bar = fig.colorbar(image, ax=ax, pad=0.02)
        bar.set_label({"z": "z", "coef": "coefficient", "qvalue": "q-value"}[statistic])
    return fig, ax


def plot_graph(results, window: int = -1, ax=None, layout: str = "circular"):
    """The discovered edges as a network — an addition to the paper.

    Node positions are on a circle; edge width scales with ``|coefficient|`` and
    edge colour encodes the lag, on the Parula scale. Drawn directly in
    matplotlib, so the package needs no graph library.

    Parameters
    ----------
    results : OracleVARXResults
    window : int, default -1
        Position of the output day.
    ax : matplotlib.axes.Axes, optional
    layout : {"circular"}, default ``"circular"``

    Returns
    -------
    (fig, ax) : tuple
    """
    import matplotlib.pyplot as plt

    if layout != "circular":
        raise ValueError('only layout="circular" is implemented')
    graph = results.graph(window)
    n = len(graph.names)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + np.pi / 2
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    frame = graph.to_frame()
    palette = parula_colors(max(results.p_max, 2))

    with plt.rc_context(JOURNAL_RC):
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(5.0, 5.0))
        scale = frame["coef"].abs().max() if len(frame) else 1.0
        for _, row in frame.iterrows():
            start = positions[graph.names.index(row["from"])]
            end = positions[graph.names.index(row["to"])]
            colour = palette[int(row["lag"]) - 1]
            if row["from"] == row["to"]:
                ax.add_patch(
                    plt.Circle(
                        start * 1.16, 0.085, fill=False, edgecolor=colour,
                        linewidth=1.0 + 2.5 * abs(row["coef"]) / scale,
                    )
                )
                continue
            ax.annotate(
                "",
                xy=end * 0.86,
                xytext=start * 0.86,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": colour,
                    "linewidth": 0.8 + 2.6 * abs(row["coef"]) / scale,
                    "connectionstyle": "arc3,rad=0.16",
                    "shrinkA": 6,
                    "shrinkB": 6,
                },
            )
        ax.scatter(positions[:, 0], positions[:, 1], s=620, color="white",
                   edgecolor=_INK, zorder=3, linewidth=1.0)
        for name, (x, y) in zip(graph.names, positions):
            ax.text(x, y, name, ha="center", va="center", zorder=4, fontsize=9)
        handles = [
            plt.Line2D([], [], color=palette[k], label=f"lag {k + 1}")
            for k in range(results.p_max)
            if (frame["lag"] == k + 1).any()
        ]
        if handles:
            ax.legend(handles=handles, loc="lower center", ncols=min(len(handles), 4),
                      bbox_to_anchor=(0.5, -0.08))
        ax.set_title(
            f"Causal lag graph{f' — {graph.label}' if graph.label else ''}"
            f"  ($\\hat{{p}}$ = {graph.lag}, {graph.n_edges} edges)",
            loc="left",
        )
        ax.set_xlim(-1.45, 1.45)
        ax.set_ylim(-1.45, 1.45)
        ax.set_aspect("equal")
        ax.axis("off")
    return fig, ax


def plot_lag_vs_volatility(
    results,
    reference: pd.Series,
    window: int = 21,
    ax=None,
    reference_label: str = "21-day realised volatility",
):
    """The selected lag order against a volatility series — the paper's Figure 5.

    The paper's reading is that the lag structure elongates in stressed markets:
    the selected order rises with realised volatility.

    Parameters
    ----------
    results : OracleVARXResults
    reference : pandas.Series
        A series to plot on the right axis — returns are rolled into a standard
        deviation, anything else is drawn as given.
    window : int, default 21
        Rolling window for the volatility, when ``reference`` holds returns.
    ax : matplotlib.axes.Axes, optional
    reference_label : str

    Returns
    -------
    (fig, (ax_left, ax_right)) : tuple
    """
    import matplotlib.pyplot as plt

    series = pd.Series(reference).astype(float)
    if series.std() < 1.0:  # looks like returns rather than a level
        series = series.rolling(window).std()
    x = _axis_values(results)

    with plt.rc_context(JOURNAL_RC):
        fig, left = (ax.figure, ax) if ax is not None else plt.subplots()
        left.step(x, results.lag, where="post", color=_INDIGO,
                  label=r"selected $\hat{p}_t$")
        left.set_ylabel(r"lag order $\hat{p}_t$", color=_INDIGO)
        left.set_yticks(np.arange(0, results.p_max + 1))
        left.set_xlabel("date" if results.index is not None else "time")
        right = left.twinx()
        aligned = series.reindex(pd.Index(x)) if results.index is not None else series
        right.plot(x, np.asarray(aligned, dtype=float), color=_AMBER,
                   linestyle="--", linewidth=1.2, label=reference_label)
        right.set_ylabel(reference_label, color=_AMBER)
        right.grid(False)
        right.spines["right"].set_visible(True)
        left.set_title("Adaptive lag order and market stress", loc="left")
    return fig, (left, right)


def plot_surface(results, statistic: str = "z", colorscale: str = "Parula"):
    """Interactive 3-D surface of a statistic over (lag, time) — an addition.

    Parameters
    ----------
    results : OracleVARXResults
    statistic : {"z", "coef"}, default ``"z"``
    colorscale : str, default ``"Parula"``

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ImportError
        If plotly is not installed.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            'plot_surface requires plotly: pip install "oracle-varx[plotly]"'
        ) from exc

    source = {"z": np.abs(results.z), "coef": results.coef}
    if statistic not in source:
        raise ValueError('statistic must be "z" or "coef"')
    # Strongest entry per (lag, day), so the surface is readable for any n.
    values = source[statistic]
    surface = values.reshape(values.shape[0], values.shape[1], -1)
    surface = (
        np.abs(surface).max(axis=2) if statistic == "z" else
        surface[np.arange(surface.shape[0])[:, None],
                np.arange(surface.shape[1])[None, :],
                np.abs(surface).argmax(axis=2)]
    )
    figure = go.Figure(
        go.Surface(
            z=surface.T,
            x=np.arange(values.shape[0]),
            y=np.arange(1, values.shape[1] + 1),
            colorscale=resolve_colorscale(colorscale),
            colorbar={"title": "|z|" if statistic == "z" else "coefficient"},
        )
    )
    figure.update_layout(
        template="plotly_white",
        scene={
            "xaxis_title": "output day",
            "yaxis_title": "lag",
            "zaxis_title": "|z|" if statistic == "z" else "coefficient",
            "camera": {"eye": {"x": 1.6, "y": -1.5, "z": 0.9}},
        },
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
        title="Edge strength over lag and time",
    )
    return figure


def plot_fdr_power_curve(
    results,
    A_true: np.ndarray,
    levels: Sequence[float] = (0.01, 0.025, 0.05, 0.10, 0.15, 0.20),
    ax=None,
):
    """Realised FDR and power against the nominal BH level — an addition.

    The paper fixes ``q = 0.05``. Sweeping it shows whether the BH step actually
    controls the false discovery rate at other levels, which is the practical
    question a user has.

    Parameters
    ----------
    results : OracleVARXResults
    A_true : numpy.ndarray
        ``(T, p_true, n, n)``.
    levels : sequence of float
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    (fig, ax) : tuple
    """
    import matplotlib.pyplot as plt

    from ._core import bh_edges
    from .evaluation import edge_scores, window_target

    target = window_target(
        A_true, results.days, results.config.ols_window, results.p_max
    )
    realised, power = [], []
    for level in levels:
        edges, _ = bh_edges(results.coef, results.se, results.lag, q=level)
        scores = edge_scores(edges, target)
        realised.append(scores["fdr"])
        power.append(scores["power"])

    with plt.rc_context(JOURNAL_RC):
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
        ax.plot(levels, realised, marker="o", color=_ROSE, label="realised FDR")
        ax.plot(levels, power, marker="s", color=_TEAL, label="power")
        ax.plot(levels, levels, color="0.4", linestyle=":", linewidth=1.0,
                label="nominal level")
        ax.set_xlabel("nominal BH level $q$")
        ax.set_ylabel("rate")
        ax.set_title("Edge discovery against the nominal level", loc="left")
        ax.legend(loc="center right")
    return fig, ax


def plot_residual_diagnostics(results, lags: int = 20):
    """Autocorrelation and a QQ plot of the forecast errors — an addition.

    Assumption (M) makes the innovations a martingale difference sequence, so
    their autocorrelations should be negligible. This panel is how a user checks
    that ``p_max`` was deep enough.

    Parameters
    ----------
    results : OracleVARXResults
    lags : int, default 20

    Returns
    -------
    (fig, axes) : tuple
    """
    import matplotlib.pyplot as plt
    from statsmodels.graphics.gofplots import qqplot
    from statsmodels.graphics.tsaplots import plot_acf

    errors = (results.actuals - results.forecasts).mean(axis=1)
    with plt.rc_context(JOURNAL_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
        plot_acf(errors, lags=lags, ax=axes[0], color=_INDIGO, title="")
        axes[0].set_title("Forecast-error autocorrelation", loc="left")
        axes[0].set_xlabel("lag")
        qqplot(errors, line="s", ax=axes[1], markerfacecolor=_TEAL,
               markeredgecolor=_TEAL, alpha=0.5)
        axes[1].set_title("Normal QQ plot of the errors", loc="left")
        fig.tight_layout()
    return fig, axes


def plot_window_guard(n_obs: int, windows: Sequence[int] | None = None, ax=None):
    """Which windows satisfy assumption (W) at a given sample size — an addition.

    Assumption (W) ties the window to ``T^(2/3)``, and whether it holds decides
    whether the estimate may be read as the coefficient on the day or only as the
    window average. Nothing else implements this check, and it is where a user
    silently loses Theorem 1.

    Parameters
    ----------
    n_obs : int
    windows : sequence of int, optional
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    (fig, ax) : tuple
    """
    import matplotlib.pyplot as plt

    from ._utils import admissible_window

    scale = n_obs ** (2.0 / 3.0)
    windows = windows or np.unique(
        np.clip(np.round(np.linspace(0.2, 2.0, 25) * scale).astype(int), 10, n_obs - 1)
    )
    targets = [admissible_window(n_obs, int(w))["target"] for w in windows]
    colours = {"Theta(u_0)": _TEAL, "Theta_bar": _AMBER}

    with plt.rc_context(JOURNAL_RC):
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
        for label, colour in colours.items():
            mask = [t == label for t in targets]
            ax.scatter(
                np.asarray(windows)[mask],
                np.asarray([w / scale for w in windows])[mask],
                color=colour, s=26,
                label={"Theta(u_0)": r"readable as $\Theta(u_0)$",
                       "Theta_bar": r"only as $\bar{\Theta}$"}[label],
            )
        ax.axhline(1.0, color=_ROSE, linestyle="--", linewidth=1.0,
                   label=r"$T_w = T^{2/3}$")
        ax.set_xlabel(r"window $T_w$")
        ax.set_ylabel(r"$T_w / T^{2/3}$")
        ax.set_title(f"Assumption (W) at T = {n_obs}", loc="left")
        ax.legend(loc="upper left")
    return fig, ax
