"""Colour scales for oracle_varx figures.

The Parula family is the house standard across Dr Merwan Roudane's packages, so a
user moving between the R packages (QuantileOnQuantile, mqqr, qqkrls, mqqcause) and
this one gets identical figures. ``"Parula"`` is the default colour scale on every
3-D surface, heatmap and contour plot.

The 64 MATLAB R2014b Parula stops below are ported verbatim from ``parula_colors()``
in the mqqcause R package rather than re-derived.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "PARULA_STOPS",
    "parula_colors",
    "parula_cmap",
    "matlab_jet_colors",
    "turbo_colors",
    "bluered_colors",
    "sinha_colors",
    "resolve_colorscale",
    "available_colorscales",
]

#: The 64 RGB stops of the MATLAB R2014b Parula colormap, as in the R packages.
PARULA_STOPS: tuple[tuple[float, float, float], ...] = (
    (0.2422, 0.1504, 0.6603), (0.2504, 0.1650, 0.7076),
    (0.2578, 0.1818, 0.7511), (0.2647, 0.1978, 0.7952),
    (0.2706, 0.2147, 0.8364), (0.2751, 0.2342, 0.8710),
    (0.2783, 0.2559, 0.8991), (0.2803, 0.2782, 0.9221),
    (0.2813, 0.3006, 0.9414), (0.2810, 0.3228, 0.9579),
    (0.2795, 0.3447, 0.9717), (0.2760, 0.3667, 0.9829),
    (0.2699, 0.3892, 0.9906), (0.2602, 0.4123, 0.9952),
    (0.2440, 0.4358, 0.9988), (0.2206, 0.4603, 0.9973),
    (0.1963, 0.4847, 0.9892), (0.1834, 0.5074, 0.9798),
    (0.1786, 0.5289, 0.9682), (0.1764, 0.5499, 0.9520),
    (0.1687, 0.5703, 0.9359), (0.1540, 0.5902, 0.9218),
    (0.1460, 0.6091, 0.9079), (0.1380, 0.6276, 0.8973),
    (0.1248, 0.6459, 0.8883), (0.1113, 0.6635, 0.8763),
    (0.0952, 0.6798, 0.8598), (0.0689, 0.6948, 0.8394),
    (0.0297, 0.7082, 0.8163), (0.0036, 0.7203, 0.7917),
    (0.0067, 0.7312, 0.7660), (0.0433, 0.7411, 0.7394),
    (0.0964, 0.7500, 0.7120), (0.1408, 0.7584, 0.6842),
    (0.1717, 0.7670, 0.6554), (0.1938, 0.7758, 0.6251),
    (0.2161, 0.7843, 0.5923), (0.2470, 0.7918, 0.5567),
    (0.2906, 0.7973, 0.5188), (0.3406, 0.8008, 0.4789),
    (0.3909, 0.8029, 0.4354), (0.4456, 0.8024, 0.3909),
    (0.5044, 0.7993, 0.3480), (0.5616, 0.7942, 0.3045),
    (0.6174, 0.7876, 0.2612), (0.6720, 0.7793, 0.2227),
    (0.7242, 0.7698, 0.1910), (0.7738, 0.7598, 0.1646),
    (0.8203, 0.7498, 0.1535), (0.8634, 0.7406, 0.1596),
    (0.9035, 0.7330, 0.1774), (0.9393, 0.7288, 0.2100),
    (0.9728, 0.7298, 0.2394), (0.9956, 0.7434, 0.2371),
    (0.9970, 0.7659, 0.2199), (0.9952, 0.7893, 0.2028),
    (0.9892, 0.8129, 0.1885), (0.9786, 0.8360, 0.1766),
    (0.9676, 0.8587, 0.1643), (0.9610, 0.8806, 0.1537),
    (0.9597, 0.9023, 0.1423), (0.9628, 0.9234, 0.1330),
    (0.9691, 0.9438, 0.1241), (0.9769, 0.9839, 0.0805),
)

_JET_ANCHORS = ("#000090", "#0000FF", "#00FFFF", "#FFFF00", "#FF0000", "#800000")

_TURBO_ANCHORS = (
    "#30123B", "#4145AB", "#4675ED", "#39A2FC", "#1BCFD4",
    "#24EAA7", "#61FC6C", "#A4FC3C", "#D1E834", "#F3C63A",
    "#FE9B2D", "#F36315", "#D93806", "#A91201", "#7A0403",
)

_BLUERED_ANCHORS = (
    "#053061", "#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B", "#67001F",
)

_SINHA_ANCHORS = ("#000000", "#4A0E00", "#B22222", "#FF8C00", "#FFFF66")

_NATIVE_PLOTLY = frozenset(
    {
        "viridis", "plasma", "cividis", "inferno", "magma", "rdbu", "rdylbu",
        "blues", "greens", "reds", "greys", "ylgnbu", "ylorrd", "bluered", "rainbow",
    }
)


def _rgb_to_hex(rgb: Sequence[float]) -> str:
    r, g, b = (int(round(float(c) * 255)) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def _ramp(anchors: Sequence[str], n: int) -> list[str]:
    """Interpolate ``n`` hex colours through ``anchors`` in RGB space.

    The Python counterpart of ``grDevices::colorRampPalette``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rgb = np.array(
        [[int(a[i : i + 2], 16) / 255.0 for i in (1, 3, 5)] for a in anchors],
        dtype=float,
    )
    if n == 1:
        return [_rgb_to_hex(rgb[0])]
    src = np.linspace(0.0, 1.0, len(rgb))
    dst = np.linspace(0.0, 1.0, n)
    out = np.column_stack([np.interp(dst, src, rgb[:, j]) for j in range(3)])
    return [_rgb_to_hex(row) for row in out]


def parula_colors(n: int = 256) -> list[str]:
    """``n`` hex colours interpolated from the 64 MATLAB R2014b Parula stops.

    Parameters
    ----------
    n : int, default 256
        Number of colours to return. ``n = 64`` returns the stops unchanged.

    Returns
    -------
    list of str
        Hex colour strings such as ``"#3E26A8"``.

    Examples
    --------
    >>> from oracle_varx import parula_colors
    >>> parula_colors(3)
    ['#3E26A8', '#12BEBA', '#F9FB15']
    """
    stops = [_rgb_to_hex(s) for s in PARULA_STOPS]
    if n == len(stops):
        return stops
    return _ramp(stops, n)


def parula_cmap(n: int = 256):
    """The Parula colormap as a matplotlib ``LinearSegmentedColormap``.

    Examples
    --------
    >>> from oracle_varx import parula_cmap
    >>> parula_cmap().name
    'parula'
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("parula", list(PARULA_STOPS), N=n)


def matlab_jet_colors(n: int = 256) -> list[str]:
    """``n`` hex colours from the classic MATLAB Jet rainbow.

    Examples
    --------
    >>> from oracle_varx import matlab_jet_colors
    >>> len(matlab_jet_colors(8))
    8
    """
    return _ramp(_JET_ANCHORS, n)


def turbo_colors(n: int = 256) -> list[str]:
    """``n`` hex colours from Google's Turbo scale.

    Examples
    --------
    >>> from oracle_varx import turbo_colors
    >>> turbo_colors(2)
    ['#30123B', '#7A0403']
    """
    return _ramp(_TURBO_ANCHORS, n)


def bluered_colors(n: int = 256) -> list[str]:
    """``n`` hex colours on a diverging blue-white-red scale.

    Use this, centred on zero (``vmin=-max|Z|``, ``vmax=+max|Z|``), for signed
    effect surfaces, so that zero sits on white.

    Examples
    --------
    >>> from oracle_varx import bluered_colors
    >>> bluered_colors(3)[1]
    '#F7F7F7'
    """
    return _ramp(_BLUERED_ANCHORS, n)


def sinha_colors(n: int = 256) -> list[str]:
    """``n`` hex colours on the Sinha black-red-yellow heatmap scale.

    Examples
    --------
    >>> from oracle_varx import sinha_colors
    >>> sinha_colors(2)
    ['#000000', '#FFFF66']
    """
    return _ramp(_SINHA_ANCHORS, n)


def resolve_colorscale(name: str = "Parula", n_breaks: int = 32):
    """Resolve a colour-scale name to a plotly colorscale.

    Parameters
    ----------
    name : str or list, default ``"Parula"``
        One of ``"Parula"``, ``"Jet"``, ``"Turbo"``, ``"BlueRed"``, ``"Sinha"``, any
        native plotly scale (``"Viridis"``, ``"RdBu"``, ...), or an already-built
        colorscale, which is returned unchanged.
    n_breaks : int, default 32
        Number of ``[value, hex]`` pairs in the returned scale.

    Returns
    -------
    list of [float, str], or str
        A plotly colorscale for the house scales; the capitalised name for scales
        plotly already ships.

    Examples
    --------
    >>> from oracle_varx import resolve_colorscale
    >>> resolve_colorscale("Parula", n_breaks=2)
    [[0.0, '#3E26A8'], [1.0, '#F9FB15']]
    >>> resolve_colorscale("Viridis")
    'Viridis'
    """
    if isinstance(name, (list, tuple)):
        return list(name)
    if not isinstance(name, str):
        raise TypeError(f"'name' must be a string or a colorscale, got {type(name)!r}")

    n_breaks = max(2, int(n_breaks))
    key = name.strip().lower()
    builders = {
        "parula": parula_colors,
        "jet": matlab_jet_colors,
        "turbo": turbo_colors,
        "bluered": bluered_colors,
        "blue_red": bluered_colors,
        "sinha": sinha_colors,
        "red_yellow_black": sinha_colors,
    }
    if key in builders:
        cols = builders[key](n_breaks)
        vals = np.linspace(0.0, 1.0, n_breaks)
        return [[float(v), c] for v, c in zip(vals, cols)]
    if key in _NATIVE_PLOTLY:
        return key.capitalize() if key != "rdbu" else "RdBu"
    return name


def available_colorscales(show: bool = True) -> list[str]:
    """List the colour scales this package understands.

    Parameters
    ----------
    show : bool, default True
        Print the names with one-line descriptions.

    Returns
    -------
    list of str

    Examples
    --------
    >>> from oracle_varx import available_colorscales
    >>> "Parula" in available_colorscales(show=False)
    True
    """
    scales = {
        "Parula": "MATLAB R2014b default, perceptually uniform (package default)",
        "Jet": "Classic MATLAB rainbow",
        "Turbo": "Google Turbo",
        "BlueRed": "Diverging, blue low to red high; centre on zero",
        "Sinha": "Black-red-yellow heatmap scale",
        "Viridis": "Perceptually uniform",
        "Plasma": "Perceptually uniform, high contrast",
        "Cividis": "Optimised for colour-vision deficiency",
        "Inferno": "Dark to bright",
        "Magma": "Dark purple to white",
        "RdBu": "Diverging red/blue",
    }
    if show:
        print("\nColour scales available in oracle_varx")
        print("=" * 38)
        for key, desc in scales.items():
            print(f"  {key:<8}: {desc}")
        print('\n  Default on every surface, heatmap and contour: "Parula"\n')
    return list(scales)
