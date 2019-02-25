#! /usr/bin/env python

import numpy as np
from scipy.stats import norm
from ..api import load
import pandas as pd
import pytest

np.random.seed(88888)


def test_mean_diff_unpaired_ci(reps=100, ci=95):
    # Create mean_diff data
    N = 10
    CONTROL_MEAN = np.random.randint(1, 1000)
    POP_SD       = np.random.randint(1, 15)
    TRUE_DIFFERENCE = np.random.randint(-POP_SD*5, POP_SD*5)

    norm_rvs_kwargs = dict(scale=POP_SD, size=N)
    c1 = norm.rvs(loc=CONTROL_MEAN, **norm_rvs_kwargs)
    t1 = norm.rvs(loc=CONTROL_MEAN+TRUE_DIFFERENCE, **norm_rvs_kwargs)

    df = pd.DataFrame({'Control' : c1, 'Test': t1})


    # Create several CIs and see if the true population difference lies within.
    error_count = 0

    for i in range(0, reps):
        # pick a random seed
        rnd_sd = np.random.randint(0, 999999)

        two_groups_unpaired = load(data=df, idx=("Control", "Test"),
                                   ci=ci, random_seed=rnd_sd)

        md = two_groups_unpaired.mean_diff.results
        low, high = float(md.bca_low), float(md.bca_high)

        if low < TRUE_DIFFERENCE < high is False:
            error_count += 1



    assert error_count <= reps * (100 - ci) / 100






def test_standardized_diff_unpaired_ci(reps=100, ci=95):
    # Create data for hedges g and cohens d
    N = 10

    CONTROL_MEAN = np.random.randint(1, 1000)
    POP_SD       = np.random.randint(1, 15)
    POP_D        = np.round(np.random.uniform(-2, 2, 1)[0], 2)


    TRUE_DIFFERENCE = CONTROL_MEAN + (POP_D * POP_SD)

    norm_rvs_kwargs = dict(scale=POP_SD, size=N)
    c1 = norm.rvs(loc=CONTROL_MEAN, **norm_rvs_kwargs)
    t1 = norm.rvs(loc=CONTROL_MEAN+TRUE_DIFFERENCE, **norm_rvs_kwargs)

    df = pd.DataFrame({'Control' : c1, 'Test': t1})


    # Create several CIs and see if the true population difference lies within.
    error_count_cohens_d = 0
    error_count_hedges_g = 0

    for i in range(0, reps):
        # pick a random seed
        rnd_sd = np.random.randint(0, 999999)

        two_groups_unpaired = load(data=df, idx=("Control", "Test"),
                                   ci=ci, random_seed=rnd_sd)

        cd = two_groups_unpaired.cohens_d.results
        hg = two_groups_unpaired.hedges_g.results

        cd_low, cd_high = float(cd.bca_low), float(cd.bca_high)
        if cd_low < POP_D < cd_high is False:
            error_count_cohens_d += 1

        hg_low, hg_high = float(hg.bca_low), float(hg.bca_high)
        if hg_low < POP_D < hg_high is False:
            error_count_hedges_g += 1



    max_errors = reps * (100 - ci) / 100
    assert error_count_cohens_d <= max_errors
    assert error_count_hedges_g <= max_errors






def test_cliffs_delta_ci(reps=100, ci=95):
    # Create two populations with a 50% overlap.
    DIFFERENCE = np.random.randint(1, 10)
    SD = np.abs(DIFFERENCE)

    N = 10000
    pop_kwargs = dict(scale=SD, size=N)
    pop1 = norm.rvs(loc=100, **pop_kwargs)
    pop2 = norm.rvs(loc=100+DIFFERENCE, **pop_kwargs)

    n = 20
    sample_kwargs = dict(size=n, replace=False)
    sample1 = np.random.choice(pop1, **sample_kwargs)
    sample2 = np.random.choice(pop2, **sample_kwargs)

    df = pd.DataFrame({'Control' : sample1, 'Test': sample2})


    # Create several CIs and see if the true overlap of 50% lies within.
    error_count = 0

    for i in range(0, reps):
        # pick a random seed
        rnd_sd = np.random.randint(0, 999999)

        two_groups_unpaired = load(data=df, idx=("Control", "Test"),
                                   ci=ci, random_seed=rnd_sd)

        cd = two_groups_unpaired.cliffs_delta.results
        low, high = float(cd.bca_low), float(cd.bca_high)

        if low < 0.5 < high is False:
            error_count += 1



    assert error_count <= reps * (100 - ci) / 100
