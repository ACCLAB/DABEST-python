import numpy as np
import pandas as pd
import pytest
import dabest


@pytest.mark.parametrize("effect_size", ["cohens_d", "hedges_g"])
def test_three_observation_standardized_effect_completes(effect_size):
    frame = pd.DataFrame({"control": [1., 2., 5.], "test": [3., 7., 11.]})
    model = dabest.load(frame, idx=("control", "test"), resamples=1000, random_seed=42)
    with pytest.warns(UserWarning, match="bootstrap.*not defined"):
        result = getattr(model, effect_size).results.iloc[0]
    assert np.isfinite(result.difference)
    assert len(result.bootstraps) == 1000
    assert np.isinf(result.bootstraps).any()
    assert np.isfinite(result.bca_low)
    assert np.isfinite(result.bca_high)


@pytest.mark.parametrize("effect_size", ["cohens_d", "hedges_g"])
@pytest.mark.parametrize("shift", [0., 1.])
def test_undefined_observed_effect_is_still_rejected(effect_size, shift):
    one = np.array([1., 1., 1.])
    two = one + shift
    with pytest.raises(ValueError, match="divisor is zero"):
        getattr(dabest.effsize, effect_size)(one, two)
    with pytest.raises(ValueError, match="divisor is zero"):
        dabest.TwoGroupsEffectSize(one, two, effect_size, resamples=30, permutation_count=10)
    with pytest.raises(ValueError, match="divisor is zero"):
        dabest.PermutationTest(one, two, effect_size, permutation_count=10)


@pytest.mark.parametrize("effect_size", ["cohens_d", "hedges_g"])
def test_boolean_observations_still_support_standardized_bootstraps(effect_size):
    control = np.array([False, True] * 8)
    test = np.array([False, True, True, True] * 4)
    expected = getattr(dabest.effsize, effect_size)(control, test)
    result = dabest.TwoGroupsEffectSize(control, test, effect_size,
                                      resamples=60, permutation_count=30)
    assert result.difference == pytest.approx(expected)
    assert np.isfinite(result.bootstraps).all()
