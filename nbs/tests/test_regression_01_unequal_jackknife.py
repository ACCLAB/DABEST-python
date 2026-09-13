import numpy as np
import pytest
from dabest._stats_tools.confint_2group_diff import compute_meandiff_jackknife


@pytest.mark.parametrize("control,test", [
    ([1., 2., 8.], [3., 5., 7., 11., 19.]),
    ([3., 5., 7., 11., 19.], [1., 2., 8.]),
])
def test_unpaired_jackknife_leaves_out_every_observation(control, test):
    control, test = np.asarray(control), np.asarray(test)
    expected = [test.mean() - np.delete(control, i).mean() for i in range(len(control))]
    expected += [np.delete(test, i).mean() - control.mean() for i in range(len(test))]
    actual = compute_meandiff_jackknife(control, test, None, "mean_diff")
    np.testing.assert_allclose(actual, expected)


def test_paired_jackknife_keeps_subjects_together():
    control = np.array([1., 4., 9., 16.])
    test = np.array([3., 6., 8., 20.])
    expected = [np.delete(test - control, i).mean() for i in range(len(control))]
    np.testing.assert_allclose(compute_meandiff_jackknife(control, test, "baseline", "mean_diff"), expected)
