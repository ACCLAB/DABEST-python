import numpy as np
import pytest
from dabest import TwoGroupsEffectSize
from dabest._stats_tools import effsize


@pytest.mark.parametrize("effect_size", ["mean_diff", "median_diff", "cohens_d", "hedges_g"])
def test_public_effect_size_matches_complete_pairs(effect_size):
    control = np.array([1., np.nan, 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.])
    test = np.array([2., 40., 5., np.nan, 7., 8., 9., 10., 11., 12., 13., 14.])
    complete = np.isfinite(control) & np.isfinite(test)
    kwargs = dict(effect_size=effect_size, is_paired="baseline", resamples=100, permutation_count=30)
    actual = TwoGroupsEffectSize(control, test, **kwargs)
    expected = TwoGroupsEffectSize(control[complete], test[complete], **kwargs)
    assert actual.difference == pytest.approx(expected.difference)
    np.testing.assert_array_equal(actual.bootstraps, expected.bootstraps)
    np.testing.assert_array_equal(actual.permutations, expected.permutations)
    assert actual.bca_low == expected.bca_low
    assert actual.bca_high == expected.bca_high


@pytest.mark.parametrize("statistic", [effsize.cohens_d, effsize.hedges_g])
def test_paired_standardized_effects_drop_incomplete_pairs(statistic):
    control = np.array([1., 2., np.nan, 4., 8.])
    test = np.array([2., np.nan, 99., 7., 10.])
    complete = np.isfinite(control) & np.isfinite(test)
    expected = statistic(control[complete], test[complete], "baseline")
    assert statistic(control, test, "baseline") == pytest.approx(expected)


def test_unpaired_samples_still_drop_missing_values_independently():
    control = np.array([1., np.nan, 3., 5.])
    test = np.array([2., 4., np.nan, 8., 10.])
    expected = effsize.cohens_d(control[np.isfinite(control)], test[np.isfinite(test)])
    assert effsize.cohens_d(control, test) == pytest.approx(expected)


def test_paired_arrays_must_match_before_missing_values_are_removed():
    with pytest.raises(ValueError, match="same length"):
        effsize.cohens_d(np.array([1., 2., np.nan]), np.array([3., 5.]), "baseline")
