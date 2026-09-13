import numpy as np
from numpy.random import PCG64, RandomState
from scipy.stats import bootstrap
from dabest._stats_tools.confint_1group import summary_ci_1group


def test_default_single_group_interval_matches_scipy_bca():
    x = np.random.default_rng(193).normal(size=31)
    actual = summary_ci_1group(x, np.mean, resamples=4000, random_seed=42)
    reference = bootstrap((x,), np.mean, vectorized=False, n_resamples=4000,
                          confidence_level=0.95, random_state=RandomState(PCG64(42)))
    np.testing.assert_allclose(actual["bootstraps"], np.sort(reference.bootstrap_distribution))
    # DABEST selects observed bootstrap values while SciPy interpolates quantiles.
    np.testing.assert_allclose([actual["bca_ci_low"], actual["bca_ci_high"]],
                               reference.confidence_interval, atol=0.01)


def test_higher_confidence_produces_wider_interval():
    x = np.arange(1., 32.) ** 1.2
    narrow = summary_ci_1group(x, np.mean, resamples=1000, alpha=0.5)
    wide = summary_ci_1group(x, np.mean, resamples=1000, alpha=0.1)
    assert wide["bca_ci_low"] < narrow["bca_ci_low"]
    assert wide["bca_ci_high"] > narrow["bca_ci_high"]
