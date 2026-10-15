"""oracle_varx: causal lag discovery in confounded time series.

A Python implementation of ORACLE-VARX, from Hong Kiat Tan, Isaac-Neil Zanoria,
James Chen, Haoyang Lyu and Mihai Cucuringu, "Causal Lag Structure Discovery in
Confounded Time Series via Orthogonalized Adaptive Estimation", 2026,
`doi:10.48550/arXiv.2610.05618 <https://doi.org/10.48550/arXiv.2610.05618>`_.

Three stages in one pipeline: double machine learning strips the confounders'
nonlinear effects out of the outcomes and the lagged series, adaptive causal lag
estimation (ACLE) picks the lag order in each rolling window by sequential
significance tests tuned on validation forecasts, and a Benjamini-Hochberg step
selects directed, lag-annotated edges at a target false discovery rate — with a
coefficient, a standard error and a confidence interval for every edge.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane
"""

from __future__ import annotations

from ._colors import (
    PARULA_STOPS,
    available_colorscales,
    bluered_colors,
    matlab_jet_colors,
    parula_cmap,
    parula_colors,
    resolve_colorscale,
    sinha_colors,
    turbo_colors,
)
from ._config import ALPHA_GRID, OracleVARXConfig
from ._core import METHODS, cost_report
from ._model import OracleVARX, method_hierarchy, oracle_varx
from ._nuisance import LEARNERS, make_learner
from ._results import CausalLagGraph, OracleVARXResults
from ._utils import admissible_window

try:  # pragma: no cover - the installed-metadata path
    from importlib.metadata import version as _version

    __version__ = _version("oracle-varx")
except Exception:  # pragma: no cover - running from a source checkout
    __version__ = "0.1.0"

__author__ = "Dr Merwan Roudane"
__email__ = "merwanroudane920@gmail.com"

__all__ = [
    # estimator
    "OracleVARX",
    "oracle_varx",
    "OracleVARXConfig",
    "OracleVARXResults",
    "CausalLagGraph",
    "method_hierarchy",
    "METHODS",
    "ALPHA_GRID",
    "LEARNERS",
    "make_learner",
    # diagnostics and cost
    "admissible_window",
    "cost_report",
    # colours
    "PARULA_STOPS",
    "parula_colors",
    "parula_cmap",
    "matlab_jet_colors",
    "turbo_colors",
    "bluered_colors",
    "sinha_colors",
    "resolve_colorscale",
    "available_colorscales",
    # metadata
    "__version__",
    "__author__",
    "__email__",
]
