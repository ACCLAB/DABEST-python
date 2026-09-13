import tracemalloc
import numpy as np
import pytest
from scipy.stats import mannwhitneyu
from dabest._stats_tools.effsize import _mann_whitney_u, cliffs_delta


@pytest.mark.parametrize("control,test", [
    ([1., 2., 2., 4.], [2., 2., 3., 7., 9.]),
    ([1., 1., 1.], [1., 1., 1., 1.]),
    ([1., 2., 3.], [4., 5.]),
    ([4., 5.], [1., 2., 3.]),
])
def test_cliffs_delta_matches_scipy_rank_statistic(control, test):
    control, test = np.array(control), np.array(test)
    reference = 2 * mannwhitneyu(test, control).statistic / (len(control) * len(test)) - 1
    assert cliffs_delta(control, test) == pytest.approx(reference)


def test_rank_calculation_uses_linear_auxiliary_memory():
    x = np.arange(2000.) % 71
    y = np.arange(2000.) % 53
    # Exercise the same function without JIT compilation allocations so the
    # measurement includes only NumPy work arrays, consistently across platforms.
    tracemalloc.start()
    try:
        actual = _mann_whitney_u.py_func(x, y)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert actual == mannwhitneyu(x, y).statistic
    # Pairwise boolean arrays require at least 16 MB for these inputs;
    # the sorted ranks need well under 1 MB.
    assert peak < 2_000_000
