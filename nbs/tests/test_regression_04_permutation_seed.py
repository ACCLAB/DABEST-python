import numpy as np
from dabest import TwoGroupsEffectSize, PermutationTest


def test_effect_size_passes_seed_to_permutation_test():
    control = np.array([1., 3., 4., 9., 11., 14.])
    test = np.array([2., 5., 6., 10., 17., 19.])
    seeds = (23, 97)
    actual = []
    for seed in seeds:
        result = TwoGroupsEffectSize(control, test, "mean_diff", resamples=100,
                                     permutation_count=30, random_seed=seed)
        expected = PermutationTest(control, test, "mean_diff", permutation_count=30,
                                   random_seed=seed)
        np.testing.assert_array_equal(result.permutations, expected.permutations)
        assert result.pvalue_permutation == expected.pvalue
        actual.append(result.permutations)
    assert not np.array_equal(*actual)
