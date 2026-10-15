"""Known-truth tests: DGP recovery, closed forms, and the equation-level checks."""

from __future__ import annotations

import numpy as np
import pytest

from oracle_varx import OracleVARX, OracleVARXConfig, cost_report
from oracle_varx._core import (
    acle_stage1,
    bh_edges,
    coefficients_from_theta,
    rolling_validation,
    second_stage,
)
from oracle_varx._utils import admissible_window, blocked_folds, build_stacks
from oracle_varx.datasets import make_oracle_dgp
from oracle_varx.evaluation import score_results


@pytest.fixture(scope="module")
def benchmark():
    """The paper's benchmark, short enough to fit in a test run."""
    return make_oracle_dgp(n_obs=900, seed=42)


def test_second_stage_recovers_a_known_linear_map():
    """Eq. (7) on noiseless residuals returns the generating matrix exactly."""
    rng = np.random.default_rng(0)
    treatment = rng.normal(size=(300, 3))
    true = np.array([[0.5, 0.0], [0.0, -0.3], [0.2, 0.1]])
    outcome = treatment @ true
    theta, se, dof = second_stage(treatment, outcome, window=200)
    np.testing.assert_allclose(theta[0], true, rtol=1e-10, atol=1e-10)
    assert dof == 200 - 3
    # With a perfect fit the residual variance, and so every standard error, is 0.
    np.testing.assert_allclose(se[0], 0.0, atol=1e-8)


def test_second_stage_matches_statsmodels_ols():
    """Eq. (7)-(8) agree with a plain statsmodels OLS on the same residuals."""
    import statsmodels.api as sm

    rng = np.random.default_rng(1)
    treatment = rng.normal(size=(150, 4))
    outcome = treatment @ rng.normal(size=(4, 2)) + rng.normal(scale=0.3, size=(150, 2))
    theta, se, _ = second_stage(treatment, outcome, window=150)
    for column in range(2):
        fitted = sm.OLS(outcome[:, column], treatment).fit()
        np.testing.assert_allclose(theta[0, :, column], fitted.params, rtol=1e-9)
        np.testing.assert_allclose(se[0, :, column], fitted.bse, rtol=1e-9)


def test_second_stage_error_shrinks_with_the_window():
    """The estimate converges on the truth as the window grows."""
    rng = np.random.default_rng(2)
    treatment = rng.normal(size=(4000, 2))
    true = np.array([[0.6], [-0.2]])
    outcome = treatment @ true + rng.normal(scale=0.5, size=(4000, 1))
    errors = []
    for window in (100, 400, 1600):
        theta, _, _ = second_stage(treatment, outcome, window=window)
        errors.append(np.abs(theta - true).mean())
    assert errors[0] > errors[1] > errors[2]
    # Roughly the root-n rate: quadrupling the window should roughly halve the error.
    assert errors[0] / errors[2] > 2.0


def test_build_stacks_puts_the_shallowest_lag_first():
    """Eq. (3): row t holds Y[t-1] in its first block, never Y[t]."""
    Y = np.arange(10.0).reshape(5, 2)
    outcome, treatment, controls = build_stacks(Y, None, p=2, start=0, end=5)
    np.testing.assert_array_equal(outcome, Y[2:])
    np.testing.assert_array_equal(treatment[:, :2], Y[1:4])
    np.testing.assert_array_equal(treatment[:, 2:], Y[0:3])
    assert controls.shape == (3, 0)


def test_build_stacks_with_history_covers_the_whole_range():
    """A test fold sources its lags from the training fold before it."""
    Y = np.arange(20.0).reshape(10, 2)
    outcome, treatment, _ = build_stacks(Y, None, p=2, start=5, end=9, allow_history=True)
    assert outcome.shape == (4, 2)
    np.testing.assert_array_equal(treatment[:, :2], Y[4:8])


def test_coefficients_reshape_follows_the_paper_convention():
    """A[k-1, i, j] is the effect of series j at lag k on series i."""
    n, p = 3, 2
    theta = np.zeros((n * p, n))
    theta[0 * n + 2, 1] = 0.6  # series 2 at lag 1 -> series 1
    theta[1 * n + 0, 0] = 0.3  # series 0 at lag 2 -> series 0
    A = coefficients_from_theta(theta, n=n, p=p)
    assert A[0, 1, 2] == pytest.approx(0.6)
    assert A[1, 0, 0] == pytest.approx(0.3)
    assert A.sum() == pytest.approx(0.9)


def test_blocked_folds_tile_without_overlapping_test_ranges():
    """§3.2: s_{k+1} = s_k + block, and the test ranges never overlap."""
    folds = blocked_folds(n_obs=1000, n_train=200, block=20, p_max=5)
    assert all(
        later.test_start >= earlier.test_end
        for earlier, later in zip(folds, folds[1:])
    )
    assert all(fold.n_train == 200 for fold in folds)
    assert folds[1].train_start - folds[0].train_start == 20


def test_blocked_folds_agree_with_sklearn_when_they_tile_exactly():
    """The delegation claim: identical to TimeSeriesSplit in the exact-tiling case."""
    from oracle_varx._utils import sklearn_folds

    ours = [
        (f.train_start, f.train_end, f.test_start, f.test_end)
        for f in blocked_folds(n_obs=300, n_train=200, block=20, p_max=0)
    ]
    assert ours == sklearn_folds(300, 200, 20)


def test_acle_keeps_lag_one_and_stops_at_the_first_insignificant_lag():
    """§3.3 Stage 1 and Algorithm 1 line 10."""
    coef = np.zeros((1, 4, 2, 2))
    se = np.full((1, 4, 2, 2), 0.05)
    coef[0, 0] = 0.6      # lag 1: strong, and never tested anyway
    coef[0, 1, 0, 0] = 0.5  # lag 2: significant
    coef[0, 3, 1, 1] = 0.9  # lag 4: significant, but unreachable past the gap at 3
    assert acle_stage1(coef, se, alpha=0.05)[0] == 2

    # With nothing beyond lag 1 significant, Stage 1 returns 1, never 0.
    quiet = np.zeros((1, 4, 2, 2))
    quiet[0, 0] = 0.6
    assert acle_stage1(quiet, se, alpha=0.05)[0] == 1


def test_acle_is_monotone_in_the_significance_level():
    """A looser level can only keep more lags, never fewer."""
    rng = np.random.default_rng(3)
    coef = rng.normal(scale=0.05, size=(40, 5, 3, 3))
    coef[:, 0] += 0.5
    se = np.full_like(coef, 0.05)
    previous = acle_stage1(coef, se, alpha=0.01)
    for alpha in (0.05, 0.10, 0.30):
        current = acle_stage1(coef, se, alpha=alpha)
        assert (current >= previous).all()
        previous = current


def test_rolling_validation_uses_only_the_trailing_window():
    """Eq. (9)-(10): day t is scored on the v days strictly before it."""
    forecasts = np.zeros((8, 1, 2))
    forecasts[:4, 0, 0] = 5.0   # candidate 0 is bad early, perfect later
    forecasts[4:, 0, 1] = 5.0   # candidate 1 is the mirror image
    actuals = np.zeros((8, 1))
    candidates = np.tile(np.array([1, 2]), (8, 1))
    chosen, lag, _ = rolling_validation(forecasts, actuals, candidates, validation_days=2)
    assert chosen[0] == 1   # early days: candidate 0 has just been bad
    assert chosen[-1] == 0  # late days: candidate 1 has just been bad


def test_bh_edges_never_look_past_the_selected_lag():
    """§3.4 step 5: entries deeper than p_hat can never be edges."""
    coef = np.full((1, 3, 2, 2), 0.9)
    se = np.full((1, 3, 2, 2), 0.01)
    edges, qvalues = bh_edges(coef, se, np.array([2]), q=0.05)
    assert edges[0, :2].all()
    assert not edges[0, 2].any()
    assert np.isnan(qvalues[0, 2]).all()


def test_bh_is_more_conservative_than_the_uncorrected_tests():
    """The BH step rejects no more than the raw 5% tests would."""
    rng = np.random.default_rng(4)
    coef = rng.normal(scale=0.1, size=(30, 2, 4, 4))
    se = np.full_like(coef, 0.05)
    edges, _ = bh_edges(coef, se, np.full(30, 2), q=0.05)
    raw = np.abs(coef / se) > 1.959963985
    assert edges.sum() <= raw[:, :2].sum()


def test_benjamini_yekutieli_is_more_conservative_than_bh():
    """The addition beyond the paper behaves as the theory says it must."""
    rng = np.random.default_rng(5)
    coef = rng.normal(scale=0.12, size=(25, 2, 3, 3))
    se = np.full_like(coef, 0.05)
    bh, _ = bh_edges(coef, se, np.full(25, 2), q=0.05, method="fdr_bh")
    by, _ = bh_edges(coef, se, np.full(25, 2), q=0.05, method="fdr_by")
    assert by.sum() <= bh.sum()


def test_dgp_has_the_regimes_the_paper_describes():
    """§5.1: the true lag order steps down from 3 to 2 to 1 at t = 1000 and 2000."""
    _, _, truth = make_oracle_dgp(n_obs=3000, seed=42)
    assert truth.p_true[0] == 3
    assert truth.p_true[999] == 3
    assert truth.p_true[1000] == 2
    assert truth.p_true[1999] == 2
    assert truth.p_true[2000] == 1
    assert truth.p_true[-1] == 1
    assert truth.A[0, 0, 0, 2] == pytest.approx(0.60)
    assert truth.A[:, 2, 2, 2].max() == pytest.approx(0.30)
    assert truth.regimes == ((0, 1000), (1000, 2000), (2000, 3000))


def test_short_samples_get_clipped_regimes():
    """A sample that never reaches the later regimes reports only the ones it has."""
    _, _, truth = make_oracle_dgp(n_obs=600, seed=0)
    assert truth.regimes == ((0, 600),)
    assert (truth.p_true == 3).all()


def test_dgp_is_reproducible_and_seed_sensitive():
    """Same seed, same data; different seed, different data."""
    first, _, _ = make_oracle_dgp(n_obs=200, seed=7)
    again, _, _ = make_oracle_dgp(n_obs=200, seed=7)
    other, _, _ = make_oracle_dgp(n_obs=200, seed=8)
    np.testing.assert_array_equal(first.to_numpy(), again.to_numpy())
    assert not np.allclose(first.to_numpy(), other.to_numpy())


def test_estimator_recovers_the_lag_one_cycle(benchmark):
    """DGP recovery: the three true lag-1 edges are the strongest found."""
    Y, W, truth = benchmark
    config = OracleVARXConfig(
        p_max=4, ols_window=150, block=25, validation_days=15, lag_offset=4
    )
    results = OracleVARX(config, method="oracle-varx", learner="linear", seed=0).fit(Y, W)
    mean_coef = results.coef.mean(axis=0)[0]
    for i, j in ((0, 2), (1, 0), (2, 1)):
        assert mean_coef[i, j] > 0.4, f"edge {j} -> {i} at lag 1 was not recovered"
    off_cycle = [
        mean_coef[i, j]
        for i in range(3)
        for j in range(3)
        if (i, j) not in {(0, 2), (1, 0), (2, 1)}
    ]
    assert max(np.abs(off_cycle)) < 0.4


def test_adaptive_lag_beats_the_rmse_rule_on_lag_tracking(benchmark):
    """The paper's Finding 1: ACLE lowers the lag RMSE whatever the base method."""
    Y, W, truth = benchmark
    config = OracleVARXConfig(
        p_max=5, ols_window=200, block=20, validation_days=20, lag_offset=5
    )
    plain = OracleVARX(config, method="var", seed=0).fit(Y)
    adaptive = OracleVARX(config, method="acle-var", seed=0).fit(Y)
    plain_score = score_results(plain, truth.A, truth.p_true)
    adaptive_score = score_results(adaptive, truth.A, truth.p_true)
    assert adaptive_score["lag_rmse"] < plain_score["lag_rmse"]


def test_cost_report_matches_the_paper_formula():
    """Appendix D.3: S = n p_max (p_max + 3) / 2 fits per block."""
    report = cost_report(n_obs=3000, n_series=3, p_max=5, block=20, n_train=200)
    assert report["fits_per_block"] == 3 * 5 * 8 // 2 == 60
    assert report["fits"] == report["blocks"] * 60
    assert report["ratio_fresh_split"] > 1


def test_first_stage_fit_count_matches_the_formula(benchmark):
    """The implementation performs exactly the number of fits the paper counts."""
    Y, W, _ = benchmark
    config = OracleVARXConfig(
        p_max=3, ols_window=100, block=25, validation_days=10, lag_offset=3
    )
    results = OracleVARX(config, method="or-varx", learner="linear", seed=0).fit(Y, W)
    blocks = len(
        blocked_folds(len(Y), config.n_train, config.block, p_max=config.p_max)
    )
    per_block = 3 * config.p_max * (config.p_max + 3) // 2
    assert results.n_fits == blocks * per_block


def test_admissible_window_flags_an_oversized_window():
    """Assumption (W): a window far above T^(2/3) forfeits Theorem 1."""
    assert admissible_window(3000, 60)["target"] == "Theta(u_0)"
    assert admissible_window(3000, 1200)["target"] == "Theta_bar"
    assert "above the scale" in admissible_window(3000, 1200)["message"]
