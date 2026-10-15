"""Complete worked example on real data: the US Treasury yield curve.

Which maturities lead which along the yield curve, at what lags, once the common
real-rate factors are partialled out nonlinearly? The nominal curve is the
endogenous system, the real (TIPS) curve is the observed confounder set, and the
question is the paper's own, asked of a public-domain dataset.

Run it:

    python examples/treasury_curve.py                 # about 5 minutes
    python examples/treasury_curve.py --quick         # seconds, linear first stage
    python examples/treasury_curve.py --block 21      # the paper's monthly refit

It prints every table and writes the figures to ``docs/assets/``. This script is
the single source of the README walkthrough and the documentation site, so the
three can never drift.

**On runtime.** The first stage dominates, at ``F * S`` fits with
``S = n p_max (p_max + 3) / 2`` per block and ``F`` blocks. With nine series,
``p_max = 5`` and the paper's monthly refit (``block=21``) that is 29,700 Extra
Trees fits — around half an hour on a laptop. The default here refits quarterly
(``block=63``), which is 9,900 fits and the same estimator on a coarser grid of
blocks; ``--block 21`` restores the paper's setting. ``cost_report`` prints the
count for any configuration before you commit to it.

Author: Dr Merwan Roudane <merwanroudane920@gmail.com>
GitHub: https://github.com/merwanroudane
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, open no windows

import pandas as pd

from oracle_varx import OracleVARX, OracleVARXConfig, admissible_window, cost_report
from oracle_varx.datasets import load_treasury
from oracle_varx.diagnostics import check_assumptions, constancy_test, lag_rule_comparison

ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 140)


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--learner", default="extra_trees",
        help="first-stage learner (default: extra_trees)",
    )
    parser.add_argument(
        "--block", type=int, default=63,
        help="cross-fitting block length and refit interval; the paper uses 21",
    )
    parser.add_argument(
        "--p-max", type=int, default=5, help="maximum lag order (default: 5)"
    )
    parser.add_argument(
        "--n-jobs", type=int, default=4, help="blocks fitted in parallel"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="use the linear first stage: seconds instead of minutes",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()
    learner = "linear" if args.quick else args.learner
    # ---------------------------------------------------------------- Step 1
    rule("Step 1 — load the shipped data")
    Y, W = load_treasury()
    print(f"nominal curve  Y : {Y.shape[0]} days x {Y.shape[1]} maturities")
    print(f"real curve     W : {W.shape[0]} days x {W.shape[1]} maturities")
    print(f"sample           : {Y.index[0].date()} to {Y.index[-1].date()}")
    print()
    print("daily changes in basis points, first rows:")
    print(Y.head(3).to_string())
    print()
    print("summary statistics:")
    print(Y.describe().loc[["mean", "std", "min", "max"]].round(2).to_string())

    # ---------------------------------------------------------------- Step 2
    rule("Step 2 — choose the configuration, and check assumption (W)")
    config = OracleVARXConfig(
        p_max=args.p_max,
        ols_window=252,
        block=args.block,
        validation_days=21,
        lag_offset=args.p_max,
        q=0.05,
    )
    print(config.summary())
    print()
    print(admissible_window(len(Y), config.ols_window)["message"])
    print()
    cost = cost_report(len(Y), Y.shape[1], config.p_max, config.block, config.n_train)
    print(
        f"first-stage fits: {cost['fits']:,} over {cost['blocks']} blocks "
        f"({cost['fits_per_block']} per block), against "
        f"{cost['fits_fresh_split_per_window']:,} for a fresh split per window "
        f"— a factor of {cost['ratio_fresh_split']:.0f}"
    )

    # ---------------------------------------------------------------- Step 3
    rule("Step 3 — estimate")
    print(f"first-stage learner: {learner}")
    started = time.time()
    model = OracleVARX(
        config, method="oracle-varx", learner=learner, seed=42, n_jobs=args.n_jobs
    )
    results = model.fit(Y, W)
    print(results)
    print(f"  fitted in           : {time.time() - started:.1f} s")

    # ---------------------------------------------------------------- Step 4
    rule("Step 4 — the journal table for the last trading day")
    print(results.summary())

    # ---------------------------------------------------------------- Step 5
    rule("Step 5 — which edges are stable across the whole sample")
    persistence = results.edge_persistence()
    print(persistence.head(12).round(4).to_string(index=False))

    # ---------------------------------------------------------------- Step 6
    rule("Step 6 — the adaptive lag order over time")
    lag = results.lag_series()
    print(f"mean {lag.mean():.2f}, median {int(lag.median())}, "
          f"range {lag.min()}-{lag.max()}")
    print()
    print("yearly average of the selected lag order:")
    print(lag.groupby(lag.index.year).mean().round(2).to_string())
    alpha = results.alpha_series()
    print()
    print("how often each Stage-1 level won:")
    print((alpha.value_counts(normalize=True).sort_index() * 100).round(1).to_string())

    # ---------------------------------------------------------------- Step 7
    rule("Step 7 — diagnostics on the paper's own assumptions")
    print(check_assumptions(results, Y).to_string())
    print()
    print("coefficients that move most across sub-periods:")
    print(constancy_test(results, blocks=4).head(6).round(4).to_string(index=False))

    # ---------------------------------------------------------------- Step 8
    rule("Step 8 — how other lag rules would have chosen")
    print(lag_rule_comparison(Y, W, config=config, learner="linear").round(4).to_string())

    # ---------------------------------------------------------------- Step 9
    rule("Step 9 — figures")
    figures = {}

    fig, _ = results.plot_lag_tracking()
    figures["lag_tracking"] = fig

    volatility = Y["10 Yr"].rolling(21).std()
    fig, _ = results.plot_lag_vs_volatility(
        volatility, reference_label="21-day volatility of the 10-year yield"
    )
    figures["lag_vs_volatility"] = fig

    fig, _ = results.plot_coefficient_paths(max_panels=6)
    figures["coefficient_paths"] = fig

    fig, _ = results.plot_edge_heatmap(lag=1, statistic="z")
    figures["edge_heatmap"] = fig

    fig, _ = results.plot_graph(window=-1)
    figures["causal_graph"] = fig

    fig, _ = results.plot_residual_diagnostics()
    figures["residual_diagnostics"] = fig

    for name, figure in figures.items():
        figure.savefig(ASSETS / f"{name}.png", dpi=150)
        figure.savefig(ASSETS / f"{name}.pdf")
        print(f"  wrote docs/assets/{name}.png and .pdf")

    # ---------------------------------------------------------------- Step 10
    rule("Step 10 — export the table")
    print("LaTeX (booktabs, ready for a manuscript):")
    print(results.to_latex(edges_only=True))

    print()
    print("Done. Figures are in docs/assets/.")


if __name__ == "__main__":
    main()
