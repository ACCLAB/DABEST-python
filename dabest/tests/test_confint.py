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
from ..stats_tools import confint_1group as ci_1g
from ..stats_tools import confint_meandiff as ci_md



@pytest.fixture
def create_pop(d=0.5, m1=50, sd=10, size=10000,
               random_seed=12345):
    """
    Keywords
    --------
    d: float, default 0.5
        The desired Cohen's d.

    m1: float, default 50
        The mean of the control population.

    sd: float, default 10
        The standard deviation of both the control and test populations.
        We assume homogeneity of variance here.

    size: int, default 10000
        The size of the population.

    Returns
    -------
    control, test: arrays
        Two populations with the desired Cohen's d.
    """

    from numpy.random import normal as normdist
    from numpy.random import seed

    seed(random_seed)

    c = normdist(loc=m1, scale=sd, size=size)

    m2 = (d * sd) + m1

    t = normdist(loc=m2, scale=sd, size=size)

    # reset seed
    seed()

    # Return output
    return c, t



@pytest.fixture
def create_samples(control_pop, test_pop,
                   control_N=10, test_N=10, replace=False,
                   random_seed=54321, **kwargs):
    """
    docstring
    """
    from numpy.random import choice, seed

    # Set seed.
    seed(random_seed)

    c_out = choice(control_pop, size=control_N,
                   replace=replace, **kwargs)
    t_out = choice(test_pop, size=test_N,
                   replace=replace, **kwargs)

    # reset seed
    seed()

    # Return output
    return c_out, t_out



@pytest.fixture
def does_ci_capture_difference(control, expt, paired, nreps=100, alpha=0.05):
    if expt is None:
        mean_diff = control.mean()
    else:
        if paired is True:
            mean_diff = np.mean(expt - control)
        elif paired is False:
            mean_diff = expt.mean() - control.mean()

    ERROR_THRESHOLD = nreps * alpha
    error_count_bca = 0
    error_count_pct = 0

    for i in range(1, nreps):
        results = bst.bootstrap(control, expt, paired=paired, alpha_level=alpha)

        print("\n95CI BCa = {}, {}".format(results.bca_ci_low, results.bca_ci_high))
        try:
            test_mean_within_ci_bca(mean_diff, results)
        except AssertionError:
            error_count_bca += 1

        print("\n95CI %tage = {}, {}".format(results.pct_ci_low, results.pct_ci_high))
        try:
            test_mean_within_ci_pct(mean_diff, results)
        except AssertionError:
            error_count_pct += 1

    print('\nNumber of BCa CIs not capturing the mean is {}'.format(error_count_bca))
    assert error_count_bca < ERROR_THRESHOLD

    print('\nNumber of Pct CIs not capturing the mean is {}'.format(error_count_pct))
    assert error_count_pct < ERROR_THRESHOLD
