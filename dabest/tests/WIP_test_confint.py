#! /usr/bin/env python
import pytest
import sys
import numpy as np
import scipy as sp

# This filters out an innocuous warning when pandas is imported,
# but the version has not been compiled against the newest numpy.
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import pandas as pd
from .._stats_tools import confint_1group as ci_1g
from .._stats_tools import confint_2group_diff as ci_md



def generate_two_groups():
    pass


# @pytest.fixture
# def does_ci_capture_difference(control, expt, paired, nreps=100, alpha=0.05):
#     if expt is None:
#         mean_diff = control.mean()
#     else:
#         if paired is True:
#             mean_diff = np.mean(expt - control)
#         elif paired is False:
#             mean_diff = expt.mean() - control.mean()
#
#     ERROR_THRESHOLD = nreps * alpha
#     error_count_bca = 0
#
#     for i in range(1, nreps):
#         results = bst.bootstrap(control, expt, paired=paired, alpha_level=alpha)
#
#         print("\n95CI BCa = {}, {}".format(results.bca_ci_low, results.bca_ci_high))
#         try:
#             test_mean_within_ci_bca(mean_diff, results)
#         except AssertionError:
#             error_count_bca += 1
#
#     print('\nNumber of BCa CIs not capturing the mean is {}'.format(error_count_bca))
#     assert error_count_bca < ERROR_THRESHOLD
