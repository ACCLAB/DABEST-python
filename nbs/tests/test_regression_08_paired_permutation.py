import numpy as np
from scipy.stats import permutation_test
from dabest import PermutationTest


def test_paired_median_uses_the_paired_null_statistic():
    control = np.array([1., 20., 40.])
    test = np.array([4., 28., 52.])
    reference = permutation_test((control, test), lambda a, b: np.median(b - a),
                                 permutation_type="samples", n_resamples=np.inf,
                                 vectorized=False)
    actual = PermutationTest(control, test, "median_diff", is_paired="baseline",
                             permutation_count=2000, random_seed=42)
    assert set(actual.permutations) == set(reference.null_distribution)
    for value in np.unique(reference.null_distribution):
        expected_probability = np.mean(reference.null_distribution == value)
        assert abs(np.mean(actual.permutations == value) - expected_probability) < 0.04


def test_paired_resamples_do_not_inherit_the_previous_swap_pattern():
    # These differences encode each of the eight sign patterns uniquely.
    control = np.zeros(3)
    test = np.array([1., 3., 9.])
    result = PermutationTest(control, test, "mean_diff", is_paired="baseline",
                             permutation_count=4000, random_seed=13)
    repeated = np.mean(result.permutations[1:] == result.permutations[:-1])
    # Independent uniform sign patterns coincide with probability 1/8.
    assert abs(repeated - 1 / 8) < 0.025


def test_paired_permutations_are_repeatable():
    kwargs = dict(effect_size="mean_diff", is_paired="baseline", permutation_count=50,
                  random_seed=91, ps_adjust=True)
    one = PermutationTest(np.array([1., 3., 9.]), np.array([4., 7., 11.]), **kwargs)
    two = PermutationTest(np.array([1., 3., 9.]), np.array([4., 7., 11.]), **kwargs)
    np.testing.assert_array_equal(one.permutations, two.permutations)
    assert one.pvalue == two.pvalue


def test_bias_adjustment_handles_more_than_float_range_of_sign_patterns():
    control = np.arange(1025.)
    test = control + 0.1
    result = PermutationTest(control, test, "mean_diff", is_paired="baseline",
                             permutation_count=1, random_seed=42, ps_adjust=True)
    assert np.isfinite(result.pvalue)
    assert 0 <= result.pvalue <= 1
