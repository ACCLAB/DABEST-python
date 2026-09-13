import numpy as np
import pytest
from dabest._stats_tools.confint_1group import summary_ci_1group


def test_required_statistic_keyword_is_used_for_summary_and_bias():
    x = np.array([1., 2., 4., 8., 12., 20., 24.])
    result = summary_ci_1group(x, np.quantile, resamples=200, q=0.75)
    assert result["summary"] == np.quantile(x, 0.75)


def test_statistic_keyword_matches_equivalent_closure():
    x = np.arange(1., 13.) ** 1.3
    kwargs = dict(resamples=500, random_seed=23)
    actual = summary_ci_1group(x, np.std, ddof=1, **kwargs)
    expected = summary_ci_1group(x, lambda y: np.std(y, ddof=1), **kwargs)
    assert actual["summary"] == pytest.approx(expected["summary"])
    assert actual["bca_ci_low"] == expected["bca_ci_low"]
    assert actual["bca_ci_high"] == expected["bca_ci_high"]
    np.testing.assert_array_equal(actual["bootstraps"], expected["bootstraps"])
