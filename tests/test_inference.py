"""Inference: size, power, and the shape of the standard errors."""

from __future__ import annotations

import numpy as np
import pytest

from oracle_varx import OracleVARX, OracleVARXConfig
from oracle_varx._core import bh_edges, second_stage, z_scores
from oracle_varx.datasets import make_oracle_dgp
from oracle_varx.evaluation import edge_scores, window_target

CONFIG = OracleVARXConfig(
    p_max=3, ols_window=120, block=20, validation_days=12, lag_offset=3
)


def test_z_statistics_are_standard_normal_under_the_null():
    """Eq. (8) is correctly scaled: under no effect, z has unit variance."""
    rng = np.random.default_rng(100)
    statistics = []
    for _ in range(400):
        treatment = rng.normal(size=(200, 2))
        outcome = rng.normal(size=(200, 1))  # no relationship at all
        theta, se, _ = second_stage(treatment, outcome, window=200)
        statistics.extend(z_scores(theta[0], se[0]).ravel().tolist())
    statistics = np.asarray(statistics)
    assert abs(statistics.mean()) < 0.06
    assert abs(statistics.std() - 1.0) < 0.06


def test_single_test_size_is_near_nominal():
    """A single two-sided z-test rejects about 5 percent of the time under the null."""
    rng = np.random.default_rng(101)
    rejections = 0
    trials = 600
    for _ in range(trials):
        treatment = rng.normal(size=(150, 2))
        outcome = rng.normal(size=(150, 1))
        theta, se, _ = second_stage(treatment, outcome, window=150)
        rejections += int(np.abs(z_scores(theta[0, :1], se[0, :1])).item() > 1.959963985)
    rate = rejections / trials
    assert 0.03 < rate < 0.075, f"rejection rate {rate:.3f} is far from 0.05"


def test_power_rises_with_the_effect_size():
    """A larger true coefficient is found more often."""
    rng = np.random.default_rng(102)

    def detection_rate(beta: float, trials: int = 150) -> float:
        hits = 0
        for _ in range(trials):
            treatment = rng.normal(size=(150, 2))
            outcome = treatment[:, :1] * beta + rng.normal(scale=1.0, size=(150, 1))
            theta, se, _ = second_stage(treatment, outcome, window=150)
            hits += int(np.abs(z_scores(theta[0, :1], se[0, :1])).item() > 1.959963985)
        return hits / trials

    assert detection_rate(0.0) < detection_rate(0.15) < detection_rate(0.4)
    assert detection_rate(0.4) > 0.8


def test_bh_controls_the_false_discovery_rate_under_the_global_null():
    """With no true edges, the realised FDR must not exceed the nominal level."""
    rng = np.random.default_rng(103)
    windows = 300
    coef = rng.normal(scale=0.05, size=(windows, 2, 3, 3))
    se = np.full_like(coef, 0.05)
    for level in (0.05, 0.10, 0.20):
        edges, _ = bh_edges(coef, se, np.full(windows, 2), q=level)
        # Everything discovered is a false discovery here.
        per_window = edges.reshape(windows, -1).any(axis=1).mean()
        assert per_window <= level * 1.6, (
            f"at q={level}, {per_window:.3f} of windows made a false discovery"
        )


def test_realised_fdr_is_near_nominal_on_the_benchmark():
    """On the paper's own design, BH holds its target once confounders are adjusted."""
    Y, W, truth = make_oracle_dgp(n_obs=900, seed=42)
    results = OracleVARX(CONFIG, method="oracle-varx", learner="linear", seed=0).fit(Y, W)
    target = window_target(truth.A, results.days, CONFIG.ols_window, results.p_max)
    scores = edge_scores(results.edges, target)
    assert scores["fdr"] < 0.25
    assert scores["power"] > 0.4


def test_adjusting_for_confounders_lowers_the_false_discovery_rate():
    """The paper's Finding 2, on its own benchmark."""
    Y, W, truth = make_oracle_dgp(n_obs=900, seed=42)
    ignored = OracleVARX(CONFIG, method="acle-var", seed=0).fit(Y)
    adjusted = OracleVARX(CONFIG, method="acle-varx", seed=0).fit(Y, W)

    target_ignored = window_target(
        truth.A, ignored.days, CONFIG.ols_window, ignored.p_max
    )
    target_adjusted = window_target(
        truth.A, adjusted.days, CONFIG.ols_window, adjusted.p_max
    )
    assert (
        edge_scores(adjusted.edges, target_adjusted)["fdr"]
        < edge_scores(ignored.edges, target_ignored)["fdr"]
    )


def test_confidence_intervals_cover_the_truth_at_about_the_stated_rate():
    """Theorem 1, checked as coverage on the known-truth benchmark."""
    Y, W, truth = make_oracle_dgp(n_obs=1200, seed=5)
    results = OracleVARX(CONFIG, method="or-varx", learner="linear", seed=0).fit(Y, W)
    lower, upper = results.conf_int(0.95)
    target = window_target(truth.A, results.days, CONFIG.ols_window, results.p_max)

    # Only the lag-1 cycle, where the truth is constant and the signal is strong.
    cells = [(0, 0, 2), (0, 1, 0), (0, 2, 1)]
    covered = [
        ((lower[:, k, i, j] <= target[:, k, i, j]) & (target[:, k, i, j] <= upper[:, k, i, j])).mean()
        for k, i, j in cells
    ]
    assert min(covered) > 0.60, f"coverage {covered} is far below nominal"


def test_standard_errors_shrink_as_the_window_grows():
    """Eq. (8) scales like 1/sqrt(T_w)."""
    rng = np.random.default_rng(104)
    treatment = rng.normal(size=(2000, 3))
    outcome = treatment @ rng.normal(size=(3, 2)) + rng.normal(scale=0.5, size=(2000, 2))
    _, se_small, _ = second_stage(treatment, outcome, window=125)
    _, se_large, _ = second_stage(treatment, outcome, window=500)
    ratio = se_small.mean() / se_large.mean()
    assert 1.6 < ratio < 2.4, f"ratio {ratio:.2f} is not near the expected 2"


def test_no_hac_by_construction():
    """Theorem 1 leaves no autocovariance terms, so eq. (8) is the plain OLS form."""
    import statsmodels.api as sm

    rng = np.random.default_rng(105)
    treatment = rng.normal(size=(300, 2))
    outcome = rng.normal(size=(300, 1))
    _, se, _ = second_stage(treatment, outcome, window=300)
    plain = sm.OLS(outcome[:, 0], treatment).fit()
    hac = sm.OLS(outcome[:, 0], treatment).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    np.testing.assert_allclose(se[0, :, 0], plain.bse, rtol=1e-9)
    assert not np.allclose(se[0, :, 0], hac.bse, rtol=1e-6)


@pytest.mark.parametrize("correction", ["fdr_bh", "fdr_by"])
def test_both_corrections_produce_valid_fits(correction):
    Y, W, _ = make_oracle_dgp(n_obs=500, seed=6)
    results = OracleVARX(
        CONFIG, method="oracle-varx", learner="linear", seed=0, correction=correction
    ).fit(Y, W)
    assert np.isfinite(results.qvalues[~np.isnan(results.qvalues)]).all()
    tested = ~np.isnan(results.qvalues)
    assert (results.qvalues[tested] >= 0).all()
    assert (results.qvalues[tested] <= 1).all()
