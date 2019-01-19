#! /usr/bin/env python
import pytest
import sys

# This filters out an innocuous warning.
import warnings
warnings.filterwarnings("ignore", message="Using or importing the ABCs")



# Functions to generate samples for testing.
def generate_two_pops(diff, diff_type,
                      populationN=10000,
                      control_mean=100, sd=10,
                      seed=12345):
    import numpy as np
    import scipy as sp
    import pandas as pd

    Ns = 20

    # Determine the central measure of the test population.
    if diff_type == "mean_diff":
        test_pop_loc = control_mean + diff

    elif diff_type == "cohens_d":
        test_pop_loc = control_mean + (diff * sd)

    # Create and return population.
    np.random.seed(seed)
    control_pop = sp.stats.norm.rvs(loc=control_mean,
                                    scale=sd, size=populationN)
    test_pop    = sp.stats.norm.rvs(loc=test_pop_loc,
                                    scale=sd, size=populationN)
    np.random.seed()

    return control_pop, test_pop

def sample_from_pops(c, t, sampleN=40, seed=12345):
    import numpy as np

    choice_kwargs = dict(size=sampleN, replace=False)

    # Sample from the populations
    np.random.seed(seed)
    control = np.random.choice(c, **choice_kwargs)
    test    = np.random.choice(t, **choice_kwargs)
    np.random.seed()

    return control, test



# Test functions.
def test_mean_diff_ci():
    md        = np.random.randint(1, 15)
    print("Testing mean difference confidence interval capture; \
           current mean difference is {}".format(md))

    c, t      = generate_two_pops(md, "mean_diff")
    con, test = sample_from_pops(c, t)

    result    = effsizes.loc['mean_diff', :]

    assert result['bca_ci_low'] < md < result['bca_ci_high']
