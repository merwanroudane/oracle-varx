"""Journal-quality tables: one frame, three renderings.

Every table is computed once into a :class:`pandas.DataFrame` and rendered as
aligned text, as booktabs LaTeX and as Markdown, so the three can never disagree.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["Summary", "build_summary", "to_latex", "to_markdown", "stars_for"]

_METHOD_LABELS = {
    "var": "VAR",
    "varx": "VARX",
    "acle-var": "ACLE-VAR",
    "acle-varx": "ACLE-VARX",
    "or-varx": "OR-VARX",
    "oracle-varx": "ORACLE-VARX",
}

_CITATION = "Tan, Zanoria, Chen, Lyu and Cucuringu (2026)"


def stars_for(pvalue: float) -> str:
    """Significance stars at the 1, 5 and 10 percent levels.

    Examples
    --------
    >>> stars_for(0.004), stars_for(0.03), stars_for(0.07), stars_for(0.5)
    ('***', '**', '*', '')
    """
    if not np.isfinite(pvalue):
        return ""
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.10:
        return "*"
    return ""


@dataclass
class Summary:
    """A rendered table that prints as text and exports to LaTeX or Markdown.

    Attributes
    ----------
    frame : pandas.DataFrame
        The single source the three renderings are built from.
    header : list of str
        The title and model/sample block.
    footer : list of str
        The significance legend and the variance source.
    """

    frame: pd.DataFrame
    header: list[str]
    footer: list[str]
    interest: pd.DataFrame | None = None
    width: int = 96

    def as_text(self) -> str:
        """The aligned console table."""
        rule_eq = "=" * self.width
        rule_dash = "-" * self.width
        lines = [rule_eq]
        lines.extend(self.header)
        lines.append(rule_dash)
        lines.append(
            f"{'lag':>4}{'from':>10}{'to':>10}{'Coef.':>12}{'Std.Err.':>12}"
            f"{'z':>10}{'P>|z|':>10}{'[95% Conf. Int.]':>22}   "
        )
        lines.append(rule_dash)
        for _, row in self.frame.iterrows():
            lines.append(
                f"{int(row['lag']):>4}{row['from']:>10}{row['to']:>10}"
                f"{row['coef']:>12.4f}{row['se']:>12.4f}{row['z']:>10.3f}"
                f"{row['pvalue']:>10.3f}{row['ci_low']:>11.4f}{row['ci_high']:>11.4f}"
                f" {row['stars']:<3}"
            )
        if self.interest is not None and len(self.interest):
            lines.append(rule_dash)
            for _, row in self.interest.iterrows():
                lines.append(f"{row['quantity']:<34}{row['value']:>20}")
        lines.append(rule_eq)
        lines.extend(self.footer)
        return "\n".join(lines)

    def as_latex(self, **kwargs) -> str:
        """The booktabs LaTeX table."""
        return _latex_from_frame(self.frame, self.header, self.footer, **kwargs)

    def as_markdown(self) -> str:
        """The Markdown table."""
        return _markdown_from_frame(self.frame, self.header, self.footer)

    def __str__(self) -> str:
        return self.as_text()

    def __repr__(self) -> str:  # pragma: no cover - Jupyter convenience
        return self.as_text()


def _edge_frame(results, window: int, level: float) -> pd.DataFrame:
    """One row per tested coefficient on the chosen output day."""
    w = range(results.n_windows)[window]
    p_hat = int(results.lag[w])
    lower, upper = results.conf_int(level)
    z_all, p_all = results.z, results.pvalues
    rows = []
    for k in range(p_hat):
        for i in range(results.n_series):
            for j in range(results.n_series):
                pval = float(p_all[w, k, i, j])
                rows.append(
                    {
                        "lag": k + 1,
                        "from": results.names[j],
                        "to": results.names[i],
                        "coef": float(results.coef[w, k, i, j]),
                        "se": float(results.se[w, k, i, j]),
                        "z": float(z_all[w, k, i, j]),
                        "pvalue": pval,
                        "qvalue": float(results.qvalues[w, k, i, j]),
                        "ci_low": float(lower[w, k, i, j]),
                        "ci_high": float(upper[w, k, i, j]),
                        "edge": bool(results.edges[w, k, i, j]),
                        "stars": stars_for(pval),
                    }
                )
    return pd.DataFrame(rows)


def build_summary(results, window: int = -1, level: float = 0.95, edges_only: bool = False):
    """Build the journal table for one output day.

    Parameters
    ----------
    results : OracleVARXResults
    window : int, default -1
        Position of the output day; ``-1`` is the last.
    level : float, default 0.95
        Confidence level for the interval columns.
    edges_only : bool, default False
        Show only the coefficients BH selected as edges.

    Returns
    -------
    Summary
    """
    w = range(results.n_windows)[window]
    frame = _edge_frame(results, window, level)
    if edges_only:
        frame = frame[frame["edge"]].reset_index(drop=True)
    frame = frame.sort_values(["lag", "to", "from"]).reset_index(drop=True)

    label = _METHOD_LABELS.get(results.method, results.method)
    day = results.index[w] if results.index is not None else results.days[w]
    cfg = results.config
    alpha = "--" if results.alpha is None else f"{float(results.alpha[w]):.2f}"
    n_edges = int(results.edges[w].sum())

    header = [
        f"{label}: causal lag discovery  ({_CITATION})".center(96).rstrip(),
        "",
        f"  Output day         : {str(day):<24}"
        f"Series             : {results.n_series:>8d}",
        f"  Rolling window T_w : {cfg.ols_window:<24d}"
        f"Confounders        : {len(results.confounder_names):>8d}",
        f"  Maximum lag p_max  : {cfg.p_max:<24d}"
        f"Selected lag p_hat : {int(results.lag[w]):>8d}",
        f"  First stage        : {results.learner:<24}"
        f"Stage-1 alpha*     : {alpha:>8}",
        f"  Edge FDR level q   : {cfg.q:<24.3f}"
        f"Discovered edges   : {n_edges:>8d}",
    ]

    strongest = (
        frame.loc[frame["coef"].abs().idxmax()] if len(frame) else None
    )
    interest = pd.DataFrame(
        [
            {
                "quantity": "Selected lag order (p_hat)",
                "value": f"{int(results.lag[w])}",
            },
            {
                "quantity": f"Edges at FDR q = {cfg.q:g}",
                "value": f"{n_edges} of {results.n_series ** 2 * int(results.lag[w])}",
            },
            {
                "quantity": "Strongest edge",
                "value": (
                    f"{strongest['from']} -> {strongest['to']} "
                    f"(lag {int(strongest['lag'])}, {strongest['coef']:.4f})"
                    if strongest is not None
                    else "none"
                ),
            },
            {
                "quantity": "Residual d.o.f.",
                "value": f"{results.dof}",
            },
        ]
    )

    footer = [
        "Significance: *** p<0.01, ** p<0.05, * p<0.10 (two-sided z-tests, Theorem 1).",
        f"Standard errors: {results.variance}.",
        "Edges: Benjamini-Hochberg over the n^2 * p_hat entry-wise tests (Appendix D.4).",
        "Edges are causal only under causal sufficiency: every common cause observed.",
    ]
    return Summary(frame=frame, header=header, footer=footer, interest=interest)


def _latex_from_frame(
    frame: pd.DataFrame,
    header: list[str],
    footer: list[str],
    booktabs: bool = True,
    stars: bool = True,
    notes: bool = True,
    float_format: str = "%.4f",
    caption: str = "Causal lag coefficients",
    label: str = "tab:oracle-varx",
) -> str:
    """Render the table as LaTeX, booktabs only, standard errors in parentheses."""
    if not booktabs:
        raise ValueError(
            "only booktabs output is supported; vertical rules and \\hline have no "
            "place in a journal table"
        )

    def fmt(value: float) -> str:
        return (float_format % value) if np.isfinite(value) else ""

    lines = [
        "% \\newcommand{\\sym}[1]{\\ensuremath{^{#1}}}  % put this in your preamble",
        "\\begin{table}[htbp]\\centering",
        f"\\caption{{{caption}}}\\label{{{label}}}",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Edge & Lag & Coefficient & 95\\% CI \\\\",
        "\\midrule",
    ]
    for _, row in frame.iterrows():
        edge = (
            f"{_escape(str(row['from']))} $\\rightarrow$ {_escape(str(row['to']))}"
        )
        star = f"\\sym{{{row['stars']}}}" if stars and row["stars"] else ""
        lines.append(
            f"{edge} & {int(row['lag'])} & {fmt(row['coef'])}{star} & "
            f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] \\\\"
        )
        lines.append(f" & & ({fmt(row['se'])}) & \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if notes:
        lines.extend(
            [
                "\\begin{tablenotes}\\footnotesize",
                "\\item Standard errors in parentheses. \\sym{***} $p<0.01$, "
                "\\sym{**} $p<0.05$, \\sym{*} $p<0.10$.",
                f"\\item {_escape(footer[1])}",
                "\\end{tablenotes}",
            ]
        )
    lines.append("\\end{table}")
    return "\n".join(lines)


def _escape(text: str) -> str:
    """Escape the LaTeX specials that appear in real variable names."""
    for char, replacement in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
        ("$", r"\$"),
    ):
        text = text.replace(char, replacement)
    return text


def _markdown_from_frame(
    frame: pd.DataFrame, header: list[str], footer: list[str]
) -> str:
    """Render the same frame as a Markdown table."""
    out = frame[
        ["lag", "from", "to", "coef", "se", "z", "pvalue", "qvalue", "stars"]
    ].copy()
    for column in ("coef", "se"):
        out[column] = out[column].map(lambda v: f"{v:.4f}")
    for column in ("z",):
        out[column] = out[column].map(lambda v: f"{v:.3f}")
    for column in ("pvalue", "qvalue"):
        out[column] = out[column].map(lambda v: f"{v:.3f}")
    body = out.to_markdown(index=False)
    return "\n".join(["", body, "", f"*{footer[0]}*", f"*{footer[1]}*"])


def to_latex(results, window: int = -1, **kwargs) -> str:
    """Build the summary for one day and render it as LaTeX."""
    level = kwargs.pop("level", 0.95)
    edges_only = kwargs.pop("edges_only", False)
    return build_summary(
        results, window=window, level=level, edges_only=edges_only
    ).as_latex(**kwargs)


def to_markdown(results, window: int = -1, **kwargs) -> str:
    """Build the summary for one day and render it as Markdown."""
    level = kwargs.pop("level", 0.95)
    edges_only = kwargs.pop("edges_only", False)
    return build_summary(
        results, window=window, level=level, edges_only=edges_only
    ).as_markdown()
