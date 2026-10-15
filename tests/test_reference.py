"""Cross-implementation checks: the paper's published numbers and statsmodels.

The authors' repository carries no licence, so none of their code or saved output
is vendored here. What this file pins down instead is:

1. the numbers **printed in the paper**, for the rows that are deterministic
   (VAR, VARX, ACLE-VAR and ACLE-VARX use no random learner, so anyone can
   reproduce them exactly);
2. equality with ``statsmodels`` wherever an established implementation of the
   same quantity exists.

The DGP was separately checked against the benchmark data distributed with the
paper and reproduces it to ``max|difference| = 9.6e-16`` across all 3,000 rows of
both ``Y`` and ``W``; that comparison is not run here because the comparison data
cannot be redistributed.
"""

from __future__ import annotations

import numpy as np
import pytest

from oracle_varx import OracleVARX, OracleVARXConfig
from oracle_varx.datasets import make_oracle_dgp
from oracle_varx.evaluation import score_results

#: Table 4 of Tan et al. (2026), the four rows with no random first stage.
#: coef_mae, fcst_mae, lag_rmse, fdr, power, f1
PAPER_TABLE_4 = {
    "var": (0.0605, 0.1100, 1.475, 0.129, 0.655, 0.734),
    "acle-var": (0.0655, 0.1094, 1.006, 0.147, 0.661, 0.730),
    "varx": (0.0627, 0.1092, 1.239, 0.076, 0.650, 0.755),
    "acle-varx": (0.0691, 0.1081, 1.099, 0.080, 0.651, 0.754),
}

#: Tolerances, stated out loud. The coefficient error of the ACLE rows is read off
#: the selected-lag fit in the paper, so it is checked separately below.
TOLERANCE = {
    "fcst_mae": 0.0005,
    "lag_rmse": 0.02,
    "fdr": 0.003,
    "power": 0.003,
    "f1": 0.003,
    "coef_mae": 0.001,
}


@pytest.fixture(scope="module")
def paper_benchmark():
    """The paper's benchmark at its published settings: T = 3000, nu = 0.1, seed 42."""
    return make_oracle_dgp()


@pytest.mark.slow
@pytest.mark.parametrize("method", sorted(PAPER_TABLE_4))
def test_reproduces_the_published_table_4_row(paper_benchmark, method):
    """Every metric of the paper's deterministic rows, within a stated tolerance."""
    Y, W, truth = paper_benchmark
    config = OracleVARXConfig.synthetic()
    confounders = None if method in ("var", "acle-var") else W
    results = OracleVARX(config, method=method, seed=42).fit(Y, confounders)
    scores = score_results(results, truth.A, truth.p_true)

    coef_mae, fcst_mae, lag_rmse, fdr, power, f1 = PAPER_TABLE_4[method]
    assert scores["fcst_mae"] == pytest.approx(fcst_mae, abs=TOLERANCE["fcst_mae"])
    assert scores["lag_rmse"] == pytest.approx(lag_rmse, abs=TOLERANCE["lag_rmse"])
    assert scores["fdr"] == pytest.approx(fdr, abs=TOLERANCE["fdr"])
    assert scores["power"] == pytest.approx(power, abs=TOLERANCE["power"])
    assert scores["f1"] == pytest.approx(f1, abs=TOLERANCE["f1"])

    if method in ("var", "varx"):
        # The non-adaptive rows match exactly on the coefficient error too.
        assert scores["coef_mae"] == pytest.approx(coef_mae, abs=TOLERANCE["coef_mae"])


@pytest.mark.slow
@pytest.mark.parametrize("method", ["acle-var", "acle-varx"])
def test_acle_coefficient_error_needs_the_selected_lag_fit(paper_benchmark, method):
    """The paper reads the ACLE coefficients off the fit at the selected lag order.

    With ``coef_source="p_hat"`` the published values come within 0.001; with the
    ``p_max`` fit that Algorithm 1 tests, they do not. Both readings are offered
    for exactly this reason.
    """
    Y, W, truth = paper_benchmark
    config = OracleVARXConfig.synthetic()
    confounders = None if method == "acle-var" else W
    published = PAPER_TABLE_4[method][0]

    selected = OracleVARX(
        config, method=method, seed=42, coef_source="p_hat"
    ).fit(Y, confounders)
    tested = OracleVARX(config, method=method, seed=42).fit(Y, confounders)

    from_selected = score_results(selected, truth.A, truth.p_true)["coef_mae"]
    from_tested = score_results(tested, truth.A, truth.p_true)["coef_mae"]
    assert abs(from_selected - published) < abs(from_tested - published)
    assert from_selected == pytest.approx(published, abs=0.001)


def test_var_path_equals_statsmodels_on_one_window():
    """The batched VAR path reproduces statsmodels exactly on a single window."""
    from statsmodels.tsa.api import VAR

    from oracle_varx._core import _ols_windows

    rng = np.random.default_rng(11)
    system = rng.normal(size=(260, 3))
    for t in range(3, 260):
        system[t] += 0.5 * system[t - 1] - 0.2 * system[t - 2]

    window, p = 200, 2
    days = np.array([window + p])
    theta, se, _ = _ols_windows(system, p=p, window=window, days=days)

    fitted = VAR(system[: window + p]).fit(maxlags=p, trend="c")
    np.testing.assert_allclose(theta[0], fitted.params[1:], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(se[0], fitted.stderr[1:], rtol=1e-8, atol=1e-11)


def test_bh_step_equals_the_papers_own_statement():
    """Appendix D.4 spelled out by hand, against the delegated implementation."""
    from oracle_varx._core import _bh_reject

    rng = np.random.default_rng(12)
    p_values = rng.uniform(0.0, 0.3, size=60)
    level = 0.05

    order = np.argsort(p_values)
    sorted_p = p_values[order]
    m = p_values.size
    passing = np.flatnonzero(sorted_p <= (np.arange(1, m + 1) / m) * level)
    expected = np.zeros(m, dtype=bool)
    if passing.size:
        expected[order[: passing.max() + 1]] = True

    np.testing.assert_array_equal(_bh_reject(p_values, level), expected)


def test_batched_solve_equals_a_per_window_loop():
    """The NumPy engine that replaces the authors' GPU tensor path."""
    from oracle_varx._core import second_stage

    rng = np.random.default_rng(13)
    treatment = rng.normal(size=(90, 3))
    outcome = rng.normal(size=(90, 2))
    window = 60

    theta, _, _ = second_stage(treatment, outcome, window=window)
    for i in (0, 7, theta.shape[0] - 1):
        expected = np.linalg.lstsq(
            treatment[i : i + window], outcome[i : i + window], rcond=None
        )[0]
        np.testing.assert_allclose(theta[i], expected, rtol=1e-9, atol=1e-12)
