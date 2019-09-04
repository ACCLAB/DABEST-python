#! /usr/bin/env python

import numpy as np
from scipy.stats import norm
from scipy.stats import skewnorm
import pandas as pd
import pytest

from .._api import load



def test_paired_mean_diff_ci():    
    # See Altman et al., Statistics with Confidence: 
    # Confidence Intervals and Statistical Guidelines (Second Edition). Wiley, 2000.
    # Pg 31.
    # Added in v0.2.5.
    blood_pressure = {"before": [148, 142, 136, 134, 138, 140, 132, 144,
                                128, 170, 162, 150, 138, 154, 126, 116],
                      "after" : [152, 152, 134, 148, 144, 136, 144, 150, 
                                146, 174, 162, 162, 146, 156, 132, 126],
                     "subject_id" : np.arange(1, 17)}
    exercise_bp = pd.DataFrame(blood_pressure)


    ex_bp = load(data=exercise_bp, idx=("before", "after"), 
                 paired=True, id_col="subject_id")
    paired_mean_diff = ex_bp.mean_diff.results
    
    assert pytest.approx(3.625) == paired_mean_diff.bca_low[0]
    assert pytest.approx(9.125) == paired_mean_diff.bca_high[0]


# def test_paired_median_diff_ci():    
#     # See Altman et al., Statistics with Confidence: 
#     # Confidence Intervals and Statistical Guidelines (Second Edition). Wiley, 2000.
#     # Pg 31.
#     endorphin = {"before": [10.6, 5.2, 8.4, 9.0, 6.6, 4.6,
#                             14.1, 5.2, 4.4, 17.4, 7.2],
#                   "after" : [14.6, 15.6, 20.2, 20.9, 24.0,
#                             25.0, 35.2, 30.2, 30.0, 46.2, 37.0],
#                  "subject_id" : np.arange(1, 12)}
#     marathon = pd.DataFrame(endorphin)
# 
# 
#     endorphin_marathon = load(data=marathon, idx=("before", "after"), 
#                                paired=True, id_col="subject_id")
#     paired_median_diff = endorphin_marathon.median_diff.results
# 
#     assert pytest.approx(10.4) == paired_median_diff.bca_low[0]
#     assert pytest.approx(25.0) == paired_median_diff.bca_high[0]


def test_unpaired_ci(reps=30, ci=95):
    # Dropped to 30 reps to save time. v0.2.5.
    POPULATION_N = 10000
    SAMPLE_N = 10

    # Create data for hedges g and cohens d.
    CONTROL_MEAN = np.random.randint(1, 1000)
    POP_SD       = np.random.randint(1, 15)
    POP_D        = np.round(np.random.uniform(-2, 2, 1)[0], 2)

    TRUE_STD_DIFFERENCE = CONTROL_MEAN + (POP_D * POP_SD)
    norm_sample_kwargs = dict(scale=POP_SD, size=SAMPLE_N)
    c1 = norm.rvs(loc=CONTROL_MEAN, **norm_sample_kwargs)
    t1 = norm.rvs(loc=CONTROL_MEAN+TRUE_STD_DIFFERENCE, **norm_sample_kwargs)

    std_diff_df = pd.DataFrame({'Control' : c1, 'Test': t1})



    # Create mean_diff data
    CONTROL_MEAN = np.random.randint(1, 1000)
    POP_SD       = np.random.randint(1, 15)
    TRUE_DIFFERENCE = np.random.randint(-POP_SD*5, POP_SD*5)

    c1 = norm.rvs(loc=CONTROL_MEAN, **norm_sample_kwargs)
    t1 = norm.rvs(loc=CONTROL_MEAN+TRUE_DIFFERENCE, **norm_sample_kwargs)

    mean_df = pd.DataFrame({'Control' : c1, 'Test': t1})



    # Create median_diff data
    MEDIAN_DIFFERENCE = np.random.randint(-5, 5)
    A = np.random.randint(-7, 7)

    skew_kwargs = dict(a=A, scale=5, size=POPULATION_N)
    skewpop1 = skewnorm.rvs(**skew_kwargs, loc=100)
    skewpop2 = skewnorm.rvs(**skew_kwargs, loc=100+MEDIAN_DIFFERENCE)

    sample_kwargs = dict(replace=False, size=SAMPLE_N)
    skewsample1 = np.random.choice(skewpop1, **sample_kwargs)
    skewsample2 = np.random.choice(skewpop2, **sample_kwargs)

    median_df = pd.DataFrame({'Control' : skewsample1, 'Test': skewsample2})



    # Create two populations with a 50% overlap.
    CD_DIFFERENCE = np.random.randint(1, 10)
    SD = np.abs(CD_DIFFERENCE)

    pop_kwargs = dict(scale=SD, size=POPULATION_N)
    pop1 = norm.rvs(loc=100, **pop_kwargs)
    pop2 = norm.rvs(loc=100+CD_DIFFERENCE, **pop_kwargs)

    sample_kwargs = dict(replace=False, size=SAMPLE_N)
    sample1 = np.random.choice(pop1, **sample_kwargs)
    sample2 = np.random.choice(pop2, **sample_kwargs)

    cd_df = pd.DataFrame({'Control' : sample1, 'Test': sample2})



    # Create several CIs and see if the true population difference lies within.
    error_count_cohens_d     = 0
    error_count_hedges_g     = 0
    error_count_mean_diff    = 0
    error_count_median_diff  = 0
    error_count_cliffs_delta = 0

    for i in range(0, reps):
        # pick a random seed
        rnd_sd = np.random.randint(0, 999999)
        load_kwargs = dict(ci=ci, random_seed=rnd_sd)



        std_diff_data = load(data=std_diff_df, idx=("Control", "Test"), **load_kwargs)

        cd = std_diff_data.cohens_d.results
        cd_low, cd_high = float(cd.bca_low), float(cd.bca_high)
        if cd_low < POP_D < cd_high is False:
            error_count_cohens_d += 1

        hg = std_diff_data.hedges_g.results
        hg_low, hg_high = float(hg.bca_low), float(hg.bca_high)
        if hg_low < POP_D < hg_high is False:
            error_count_hedges_g += 1



        mean_diff_data = load(data=mean_df, idx=("Control", "Test"), **load_kwargs)
        mean_d = mean_diff_data.mean_diff.results
        mean_d_low, mean_d_high = float(mean_d.bca_low), float(mean_d.bca_high)
        if mean_d_low < TRUE_DIFFERENCE < mean_d_high is False:
            error_count_mean_diff += 1


        median_diff_data = load(data=median_df, idx=("Control", "Test"),
                             **load_kwargs)
        median_d = median_diff_data.median_diff.results
        median_d_low, median_d_high = float(median_d.bca_low), float(median_d.bca_high)
        if median_d_low < MEDIAN_DIFFERENCE < median_d_high is False:
            error_count_median_diff += 1


        cd_data = load(data=cd_df, idx=("Control", "Test"), **load_kwargs)
        cd = cd_data.cliffs_delta.results
        low, high = float(cd.bca_low), float(cd.bca_high)
        if low < 0.5 < high is False:
            error_count_cliffs_delta += 1


    max_errors = int(np.ceil(reps * (100 - ci) / 100))

    assert error_count_cohens_d     <= max_errors
    assert error_count_hedges_g     <= max_errors
    assert error_count_mean_diff    <= max_errors
    assert error_count_median_diff  <= max_errors
    assert error_count_cliffs_delta <= max_errors
