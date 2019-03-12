# #! /usr/bin/env python
import pytest
import sys
import numpy as np
import scipy as sp

# This filters out an innocuous warning when pandas is imported,
# but the version has not been compiled against the newest numpy.
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import pandas as pd
from .. import _bootstrap_tools as bst



def create_dummy_dataset(seed=None, n=30, base_mean=0, expt_groups=6,
                         scale_means=2, scale_std=1.2):
    """
    Creates a dummy dataset for plotting.

    Returns the seed used to generate the random numbers,
    the maximum possible difference between mean differences,
    and the dataset itself.
    """

    # Set a random seed.
    if seed is None:
        random_seed = np.random.randint(low=1, high=1000, size=1)[0]
    else:
        if isinstance(seed, int):
            random_seed = seed
        else:
            raise TypeError('{} is not an integer.'.format(seed))

    # Generate a set of random means
    np.random.seed(random_seed)
    MEANS = np.repeat(base_mean, expt_groups) + np.random.random(size=expt_groups) * scale_means
    SCALES = np.random.random(size=expt_groups) * scale_std

    max_mean_diff = np.ptp(MEANS)

    dataset = list()
    for i, m in enumerate(MEANS):
        pop = sp.stats.norm.rvs(loc=m, scale=SCALES[i], size=10000)
        sample = np.random.choice(pop, size=n, replace=False)
        dataset.append(sample)

    df = pd.DataFrame(dataset).T
    df.columns = [str(c) for c in df.columns]

    return random_seed, max_mean_diff, df


def is_difference(result):
    assert result.is_difference == True

def is_paired(result):
    assert result.is_paired == True

def check_pvalue_1samp(result):
    assert result.pvalue_1samp_ttest != 'NIL'

def check_pvalue_2samp_unpaired(result):
    assert result.pvalue_2samp_ind_ttest != 'NIL'

def check_pvalue_2samp_paired(result):
    assert result.pvalue_2samp_related_ttest != 'NIL'

def check_mann_whitney(result):
    """Nonparametric unpaired"""
    assert result.pvalue_mann_whitney != 'NIL'
    assert result.pvalue_wilcoxon == 'NIL'

def check_wilcoxon(result):
    """Nonparametric Paired"""
    assert result.pvalue_wilcoxon != 'NIL'
    assert result.pvalue_mann_whitney == 'NIL'

# def test_mean_within_ci_bca(mean, result):
#     assert mean >= result.bca_ci_low
#     assert mean <= result.bca_ci_high
#
# def test_mean_within_ci_pct(mean, result):
#     assert mean >= result.pct_ci_low
#     assert mean <= result.pct_ci_high

def single_samp_stat_tests(sample, result):

    assert result.is_difference == False
    assert result.is_paired == False

    ttest_result = sp.stats.ttest_1samp(sample, 0).pvalue
    assert result.pvalue_1samp_ttest == pytest.approx(ttest_result)

def unpaired_stat_tests(control, expt, result):
    is_difference(result)
    check_pvalue_2samp_unpaired(result)
    check_mann_whitney(result)

    true_mean = expt.mean() - control.mean()
    assert result.summary == pytest.approx(true_mean)

    scipy_ttest_ind_result = sp.stats.ttest_ind(control, expt).pvalue
    assert result.pvalue_2samp_ind_ttest == pytest.approx(scipy_ttest_ind_result)

    mann_whitney_result = sp.stats.mannwhitneyu(control, expt,
                                                alternative='two-sided').pvalue
    assert result.pvalue_mann_whitney == pytest.approx(mann_whitney_result)

def paired_stat_tests(control, expt, result):
    is_difference(result)
    is_paired(result)
    check_wilcoxon(result)

    true_mean = np.mean(expt - control)
    assert result.summary == pytest.approx(true_mean)

    scipy_ttest_paired = sp.stats.ttest_rel(control, expt).pvalue
    assert result.pvalue_2samp_paired_ttest == pytest.approx(scipy_ttest_paired)

    wilcoxon_result = sp.stats.wilcoxon(control, expt).pvalue
    assert result.pvalue_wilcoxon == pytest.approx(wilcoxon_result)

def does_ci_capture_mean_diff(control, expt, paired, nreps=100, alpha=0.05):
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
            # test_mean_within_ci_bca(mean_diff, results)
            assert mean_diff >= results.bca_ci_low
            assert mean_diff <= results.bca_ci_high
        except AssertionError:
            error_count_bca += 1

        print("\n95CI %tage = {}, {}".format(results.pct_ci_low, results.pct_ci_high))
        try:
            # test_mean_within_ci_pct(mean_diff, results)
            assert mean_diff >= results.pct_ci_low
            assert mean_diff <= results.pct_ci_high
        except AssertionError:
            error_count_pct += 1

    print('\nNumber of BCa CIs not capturing the mean is {}'.format(error_count_bca))
    assert error_count_bca < ERROR_THRESHOLD

    print('\nNumber of Pct CIs not capturing the mean is {}'.format(error_count_pct))
    assert error_count_pct < ERROR_THRESHOLD




# Start tests below.
def test_single_sample_bootstrap(mean=100, sd=10, n=25, nreps=100, alpha=0.05):
    print("Testing single sample bootstrap.")

    # Set the random seed.
    random_seed = np.random.randint(low=1, high=1000, size=1)[0]
    np.random.seed(random_seed)
    print("\nRandom seed = {}".format(random_seed))

    # single sample
    pop = sp.stats.norm.rvs(loc=mean, scale=sd * np.random.random(1)[0], size=10000)
    sample = np.random.choice(pop, size=n, replace=False)
    print("\nMean = {}".format(mean))

    results = bst.bootstrap(sample, alpha_level=alpha)
    single_samp_stat_tests(sample, results)

    does_ci_capture_mean_diff(sample, None, False, nreps, alpha)



def test_unpaired_difference(mean=100, sd=10, n=25, nreps=100, alpha=0.05):
    print("Testing unpaired difference bootstrap.\n")

    rand_delta = np.random.randint(-10, 10) # randint between -10 and 10
    SCALES = sd * np.random.random(2)

    pop1 = sp.stats.norm.rvs(loc=mean, scale=SCALES[0], size=10000)
    sample1 = np.random.choice(pop1, size=n, replace=False)

    pop2 = sp.stats.norm.rvs(loc=mean+rand_delta, scale=SCALES[1], size=10000)
    sample2 = np.random.choice(pop2, size=n, replace=False)

    results = bst.bootstrap(sample1, sample2, paired=False, alpha_level=alpha)
    unpaired_stat_tests(sample1, sample2, results)

    does_ci_capture_mean_diff(sample1, sample2, False, nreps, alpha)



def test_paired_difference(mean=100, sd=10, n=25, nreps=100, alpha=0.05):
    print("Testing paired difference bootstrap.\n")

    # Assume equal variances here, given that the samples
    # are supposed to be paired.

    rand_delta = np.random.randint(-10, 10) # randint between -10 and 10
    print('difference={}'.format(rand_delta))
    SCALE = sd * np.random.random(1)[0]

    pop1 = sp.stats.norm.rvs(loc=mean, scale=SCALE, size=10000)
    sample1 = np.random.choice(pop1, size=n, replace=False)

    pop2 = sp.stats.norm.rvs(loc=mean+rand_delta, scale=SCALE, size=10000)
    sample2 = np.random.choice(pop2, size=n, replace=False)

    results = bst.bootstrap(sample1, sample2, alpha_level=alpha, paired=True)
    paired_stat_tests(sample1, sample2, results)

    does_ci_capture_mean_diff(sample1, sample2, True, nreps, alpha)
