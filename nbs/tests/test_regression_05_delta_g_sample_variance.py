import numpy as np
import pytest
from dabest._stats_tools.confint_2group_diff import compute_delta2_bootstrapped_diff


@pytest.mark.parametrize("sizes", [(4, 4, 4, 4), (3, 5, 6, 8), (1, 3, 4, 5)])
def test_delta_g_uses_pooled_within_group_sample_variance(sizes):
    groups = [np.arange(float(n)) * scale + shift
              for n, scale, shift in zip(sizes, [1., 2., 3., 4.], [0., 1., 2., 5.])]
    # Compute pooled variance directly from residual sums of squares.
    pooled_sd = np.sqrt(sum(np.square(x - x.mean()).sum() for x in groups) /
                        sum(len(x) - 1 for x in groups))
    delta = groups[3].mean() - groups[2].mean() - groups[1].mean() + groups[0].mean()
    boot_g, g, boot_delta = compute_delta2_bootstrapped_diff(*groups, resamples=30)
    assert g == pytest.approx(delta / pooled_sd)
    np.testing.assert_allclose(boot_g * pooled_sd, boot_delta)


def test_proportional_delta_delta_is_unstandardized():
    groups = [np.array(x) for x in ([0., 1., 0.], [1., 1., 0.], [0., 0., 1.], [1., 1., 1.])]
    boot_g, g, boot_delta = compute_delta2_bootstrapped_diff(*groups, proportional=True, resamples=30)
    np.testing.assert_array_equal(boot_g, boot_delta)
    assert g == pytest.approx(1 / 3)
