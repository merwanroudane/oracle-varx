"""Input contracts, reproducibility, and the output surface."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oracle_varx import OracleVARX, OracleVARXConfig, make_learner, oracle_varx
from oracle_varx.datasets import load_treasury, make_oracle_dgp, observed_confounders
from oracle_varx.diagnostics import check_assumptions, constancy_test
from oracle_varx.portfolio import performance, strategy_returns, weights

CONFIG = OracleVARXConfig(
    p_max=3, ols_window=80, block=20, validation_days=10, lag_offset=3
)


@pytest.fixture(scope="module")
def data():
    return make_oracle_dgp(n_obs=400, seed=1)


def test_dataframe_and_ndarray_give_identical_numbers(data):
    """Input type must never change a result."""
    Y, W, _ = data
    frame = OracleVARX(CONFIG, method="oracle-varx", learner="linear", seed=0).fit(Y, W)
    array = OracleVARX(CONFIG, method="oracle-varx", learner="linear", seed=0).fit(
        Y.to_numpy(), W.to_numpy()
    )
    np.testing.assert_allclose(frame.coef, array.coef, rtol=1e-12)
    np.testing.assert_array_equal(frame.lag, array.lag)
    assert frame.names == ["X", "Y", "Z"]
    assert array.names == ["Y1", "Y2", "Y3"]


def test_names_and_index_are_preserved(data):
    """A dated frame comes back dated, so tables and figures can say when."""
    Y, W, _ = data
    dates = pd.date_range("2020-01-01", periods=len(Y), freq="B")
    results = OracleVARX(CONFIG, method="acle-var", seed=0).fit(Y.set_index(dates))
    assert results.index is not None
    assert len(results.index) == results.n_windows
    assert results.lag_series().index[0] == results.index[0]


def test_same_seed_gives_identical_results(data):
    """Reproducibility: one seed, one answer."""
    Y, W, _ = data
    first = OracleVARX(CONFIG, method="oracle-varx", learner="extra_trees", seed=3).fit(Y, W)
    again = OracleVARX(CONFIG, method="oracle-varx", learner="extra_trees", seed=3).fit(Y, W)
    np.testing.assert_array_equal(first.coef, again.coef)
    np.testing.assert_array_equal(first.edges, again.edges)


def test_results_do_not_depend_on_n_jobs(data):
    """Parallelism must not move a number: child seeds are spawned per fold."""
    Y, W, _ = data
    serial = OracleVARX(
        CONFIG, method="or-varx", learner="linear", seed=4, n_jobs=1
    ).fit(Y, W)
    parallel = OracleVARX(
        CONFIG, method="or-varx", learner="linear", seed=4, n_jobs=2
    ).fit(Y, W)
    np.testing.assert_allclose(serial.coef, parallel.coef, rtol=1e-10)


def test_missing_values_raise_a_pointed_error(data):
    """The error must name the argument at fault and say what to do."""
    Y, W, _ = data
    dirty = Y.copy()
    dirty.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match=r"Y holds 1 missing.*dropna"):
        OracleVARX(CONFIG, method="acle-var").fit(dirty)


def test_mismatched_lengths_raise(data):
    Y, W, _ = data
    with pytest.raises(ValueError, match="Y has 400 rows but W has 399"):
        OracleVARX(CONFIG, method="oracle-varx", learner="linear").fit(Y, W.iloc[:-1])


def test_dml_without_confounders_raises_with_the_alternative(data):
    """The message must point at the rung that does apply."""
    Y, _, _ = data
    with pytest.raises(ValueError, match="needs confounders"):
        OracleVARX(CONFIG, method="oracle-varx", learner="linear").fit(Y, None)


def test_too_short_a_sample_raises_with_the_knob_to_turn(data):
    Y, W, _ = data
    config = OracleVARXConfig(p_max=3, ols_window=400, block=20, validation_days=10)
    with pytest.raises(ValueError, match="ols_window"):
        OracleVARX(config, method="acle-var").fit(Y)


def test_unknown_method_and_correction_raise():
    with pytest.raises(ValueError, match="unknown method"):
        OracleVARX(CONFIG, method="oracle")
    with pytest.raises(ValueError, match="fdr_bh"):
        OracleVARX(CONFIG, correction="bonferroni")
    with pytest.raises(ValueError, match="coef_source"):
        OracleVARX(CONFIG, coef_source="both")


def test_functional_wrapper_matches_the_class(data):
    """The one-call front door must agree with the class it wraps."""
    Y, W, _ = data
    direct = oracle_varx(
        Y, W, p_max=3, method="or-varx", learner="linear",
        ols_window=80, block=20, validation_days=10, seed=0,
    )
    config = OracleVARXConfig(
        p_max=3, ols_window=80, block=20, validation_days=10, lag_offset=3
    )
    classed = OracleVARX(config, method="or-varx", learner="linear", seed=0).fit(Y, W)
    np.testing.assert_allclose(direct.coef, classed.coef, rtol=1e-12)


def test_every_method_rung_runs(data):
    Y, W, _ = data
    for method in ("var", "varx", "acle-var", "acle-varx", "or-varx", "oracle-varx"):
        confounders = None if method in ("var", "acle-var") else W
        results = OracleVARX(CONFIG, method=method, learner="linear", seed=0).fit(
            Y, confounders
        )
        assert results.n_windows > 0
        assert (results.lag >= 1).all()
        assert (results.lag <= CONFIG.p_max).all()


def test_results_surface(data):
    """Everything a user reads off a fit."""
    Y, W, _ = data
    results = OracleVARX(CONFIG, method="oracle-varx", learner="linear", seed=0).fit(Y, W)

    lower, upper = results.conf_int(0.95)
    assert (lower <= results.coef).all() and (results.coef <= upper).all()
    assert ((results.pvalues >= 0) & (results.pvalues <= 1)).all()

    tidy = results.to_frame(window=0)
    assert set(tidy.columns) >= {"lag", "from", "to", "coef", "se", "z", "qvalue"}

    graph = results.graph(-1)
    assert graph.n_edges == int(results.edges[-1].sum())
    assert graph.adjacency_frame(1).shape == (3, 3)

    persistence = results.edge_persistence()
    assert (persistence["share"] <= 1.0).all()

    text = str(results.summary())
    assert "ORACLE-VARX" in text and "Significance" in text
    assert "\\toprule" in results.to_latex()
    assert "|" in results.to_markdown()
    assert results.alpha is not None
    assert "p_hat" in str(results)


def test_confidence_level_must_be_a_probability(data):
    Y, W, _ = data
    results = OracleVARX(CONFIG, method="acle-var", seed=0).fit(Y)
    with pytest.raises(ValueError, match="level must lie"):
        results.conf_int(1.5)


def test_observability_levels_slice_the_confounders(data):
    _, W, _ = data
    assert observed_confounders(W, "all").shape[1] == 3
    assert observed_confounders(W, "partial_2").shape[1] == 2
    assert observed_confounders(W, "partial_1").shape[1] == 1
    assert observed_confounders(W, "none").shape[1] == 0
    with pytest.raises(ValueError, match="unknown observability"):
        observed_confounders(W, "some")


def test_shipped_dataset_loads_and_is_aligned():
    """The real dataset must be present in the wheel and internally consistent."""
    Y, W = load_treasury()
    assert Y.shape[1] == 9
    assert W.shape[1] == 4
    assert len(Y) > 3000
    assert (Y.index == W.index).all()
    assert Y.notna().all().all() and W.notna().all().all()
    array_Y, array_W = load_treasury(as_frame=False)
    assert isinstance(array_Y, np.ndarray) and array_W.shape[1] == 4


def test_learner_factory(data):
    assert make_learner("extra_trees", seed=0).get_params()["max_depth"] == 5
    assert make_learner("random_forest", seed=0).get_params()["n_estimators"] == 100
    assert make_learner("linear").__class__.__name__ == "LinearRegression"
    with pytest.raises(ValueError, match="unknown learner"):
        make_learner("magic")
    # Any estimator with fit/predict is accepted and passed straight through.
    from sklearn.tree import DecisionTreeRegressor

    tree = DecisionTreeRegressor(max_depth=2)
    assert make_learner(tree) is tree


def test_custom_estimator_as_first_stage(data):
    Y, W, _ = data
    from sklearn.linear_model import Ridge

    results = OracleVARX(CONFIG, method="or-varx", learner=Ridge(alpha=1.0), seed=0).fit(
        Y, W
    )
    assert results.learner == "custom"
    assert results.n_windows > 0


def test_portfolio_rules_have_unit_gross_exposure():
    forecasts = np.array([[0.02, -0.01, 0.004, 0.0]])
    for rule in ("naive", "weighted", "top25", "top50", "top75"):
        w = weights(forecasts, rule=rule)
        assert np.abs(w).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unknown rule"):
        weights(forecasts, rule="momentum")


def test_portfolio_hedge_removes_net_exposure():
    """A fully net-long book paying the hedge return nets out to zero."""
    forecasts = np.array([[1.0, 1.0]])
    realised = np.array([[0.01, 0.01]])
    hedged = strategy_returns(forecasts, realised, hedge=np.array([0.01]), rule="naive")
    assert hedged[0] == pytest.approx(0.0, abs=1e-12)


def test_performance_statistics():
    rng = np.random.default_rng(9)
    stats = performance(rng.normal(scale=0.01, size=504))
    assert stats["n_days"] == 504
    assert -1.0 <= stats["max_drawdown"] <= 0.0
    assert 0.0 <= stats["hit_rate"] <= 1.0
    with pytest.raises(ValueError, match="no finite returns"):
        performance(np.array([np.nan]))


def test_diagnostics_run(data):
    Y, W, _ = data
    results = OracleVARX(CONFIG, method="oracle-varx", learner="linear", seed=0).fit(Y, W)
    checks = check_assumptions(results, Y)
    assert len(checks) == 4
    assert set(checks.columns) == {"statistic", "pvalue", "verdict", "note"}
    constancy = constancy_test(results, blocks=3)
    assert {"lag", "from", "to", "wald", "pvalue"} <= set(constancy.columns)
    with pytest.raises(ValueError, match="blocks must be"):
        constancy_test(results, blocks=1)
