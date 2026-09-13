import numpy as np
import pandas as pd
import pytest
import dabest


@pytest.mark.parametrize("paired", ["baseline", "sequential"])
@pytest.mark.parametrize("effect_size", ["mean_diff", "median_diff"])
@pytest.mark.parametrize("subjects", [list("abcdefgh"), [1, "b", 3, "d", 5, "f", 7, "h"]])
def test_shuffling_long_data_preserves_paired_analysis(paired, effect_size, subjects):
    wide = pd.DataFrame({
        "subject": subjects,
        "before": [1., 3., 8., 10., 14., 17., 23., 31.],
        "after": [4., 6., 9., 16., 18., 19., 30., 34.],
    })
    long = wide.melt(id_vars="subject", var_name="group", value_name="score")
    shuffled = long.sample(frac=1, random_state=13)
    kwargs = dict(idx=("before", "after"), paired=paired, id_col="subject", resamples=100)
    expected = getattr(dabest.load(wide, **kwargs), effect_size).results.iloc[0]
    actual = getattr(dabest.load(shuffled, x="group", y="score", **kwargs), effect_size).results.iloc[0]
    assert actual.difference == pytest.approx(expected.difference)
    np.testing.assert_array_equal(actual.bootstraps, expected.bootstraps)
    np.testing.assert_array_equal(actual.permutations, expected.permutations)
    assert actual.bca_low == expected.bca_low
    assert actual.bca_high == expected.bca_high
    assert actual.pvalue_wilcoxon == expected.pvalue_wilcoxon
