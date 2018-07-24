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
from .. import bootstrap_tools as bst


@pytest.fixture
def create_dummy_dataset(n=50, expt_groups=6):
    # Dummy dataset
    Ns = n
    dataset = list()
    for seed in np.random.randint(low=100, high=1000, size=expt_groups):
        np.random.seed(seed)
        dataset.append(np.random.randn(Ns))
    df = pd.DataFrame(dataset).T
    # Create some upwards/downwards shifts.
    for c in df.columns:
        df.loc[:,c] =( df[c] * np.random.random()) + np.random.random()

    return df

# Unpaired tests.
@pytest.fixture
def unpaired(df, control, expt):
    result = bst.bootstrap(df[control], df[expt])

    assert(result.is_difference == True)
    assert(result.pvalue_1samp_ttest =='NIL')
    assert(result.is_paired == False)
    assert(result.pvalue_2samp_paired_ttest =='NIL')
    assert(result.pvalue_wilcoxon == 'NIL')

    true_mean = df[expt].mean() - df[control].mean()
    assert(result.summary == true_mean)

    scipy_ttest_ind_result = sp.stats.ttest_ind(df[control],
        df[expt]).pvalue
    assert(result.pvalue_2samp_ind_ttest == scipy_ttest_ind_result)

    mann_whitney_result = sp.stats.mannwhitneyu(df[control],
        df[expt],
        alternative='two-sided').pvalue
    assert(result.pvalue_mann_whitney == mann_whitney_result)

# Paired tests.
@pytest.fixture
def paired(df, control, expt):
    result = bst.bootstrap(df[control], df[expt], paired=True)

    assert(result.is_difference == True)
    assert(result.pvalue_1samp_ttest =='NIL')
    assert(result.is_paired == True)
    assert(result.pvalue_2samp_ind_ttest =='NIL')
    assert(result.pvalue_mann_whitney == 'NIL')

    true_mean = np.mean(df[expt] - df[control])
    assert(result.summary == true_mean)

    scipy_ttest_ind_result = sp.stats.ttest_rel(df[control],
        df[expt]).pvalue
    assert(result.pvalue_2samp_paired_ttest == scipy_ttest_ind_result)

    wilcoxon_result = sp.stats.wilcoxon(df[control],
        df[expt]).pvalue
    assert(result.pvalue_wilcoxon == wilcoxon_result)

@pytest.fixture
def make_test_tuples(df):
    # Create all pairs of control-expt tuples.
    con = np.repeat(df.columns[0], len(df.columns)-1)
    expt = [c for c in df.columns[1:]]
    zipped = zip(con, expt)
    test_tuples = list(zipped)

    return test_tuples

# Start tests below.
def test_unpaired(expt_groups_count=5):
    # Create dummy data for testing.
    test_data = create_dummy_dataset(expt_groups=expt_groups_count)
    # Now, create all pairs of control-expt tuples.
    test_tuples = make_test_tuples(test_data)
    # Run tests.
    print('testing unpaired with dataframe.')
    for t in test_tuples:
        sys.stdout.write('\r{}'.format(t))
        sys.stdout.flush()
        unpaired(df=test_data, control=t[0], expt=t[1])

def test_paired(expt_groups_count=5):
    # Create dummy data for testing.
    test_data = create_dummy_dataset(expt_groups=expt_groups_count)
    # Now, create all pairs of control-expt tuples.
    test_tuples = make_test_tuples(test_data)
    # Run tests.
    print('testing paired with dataframe.')
    for t in test_tuples:
        sys.stdout.write('\r{}'.format(t))
        sys.stdout.flush()
        paired(df=test_data, control=t[0], expt=t[1])

def test_single_sample_bootstrap(mean=100, sd=10, n=25,
    alpha=0.05, nreps=100):
    print("Testing single sample bootstrap.")
    # single sample
    sample = np.random.normal(loc=mean,
                              scale=sd*np.random.random(1)[0],
                              size=n)
    error_count = 0
    for i in range(1, nreps):
        try:
            results = bst.bootstrap(sample, alpha_level=alpha)
            assert mean >= results.bca_ci_low
            assert mean <= results.bca_ci_high
            assert mean >= results.pct_ci_low
            assert mean <= results.pct_ci_high

        except AssertionError:
            error_count += 1
        sys.stdout.write('\r{0} runs of {1};  errors so far={2}'.format(i+1,
            nreps, error_count))
        sys.stdout.flush()
    print('\nNumber of CIs not capturing the mean is {}'.format(error_count))
    assert error_count < nreps*alpha

def test_difference_paired_bootstrap(mean=100, sd=10, n=25,
    alpha=0.05, nreps=100):
    print("Testing paired difference bootstrap.")
    # Assume equal variances here
    # given that the samples are supposed to be paired.
    sample1 = np.random.normal(loc=mean,
                               scale=sd*(1+np.random.random(1)[0]),
                               size=n)
    rand_delta = np.random.randint(-10, 10) # randint between -10 and 10
    sample2 = np.random.normal(loc=mean+rand_delta,
                               scale=sd*(1+np.random.random(1)[0]),
                               size=n)

    error_count = 0
    for i in range(1, nreps):
        try:
            results = bst.bootstrap(sample1, sample2,
                                    alpha_level=alpha, paired=True)
            assert rand_delta >= results.bca_ci_low
            assert rand_delta <= results.bca_ci_high
            assert rand_delta >= results.pct_ci_low
            assert rand_delta <= results.pct_ci_high

        except AssertionError:
            error_count += 1
        sys.stdout.write('\r{0} runs of {1};  errors so far={2}'.format(i+1,
            nreps, error_count))
        sys.stdout.flush()
    print('\nNumber of CIs not capturing the mean is {}'.format(error_count))
    assert error_count < nreps*alpha

def test_difference_unpaired_bootstrap(mean=100, sd=10, n=25,
    alpha=0.05, nreps=100):
    print("Testing unpaired difference bootstrap.")
    rand_n1 = np.random.randint(low=-10, high=10, size=1)[0]
    sample1 = np.random.normal(loc=mean,
                               scale=sd*(1+np.random.random(1)[0]),
                               size=n+rand_n1)
    rand_n2 = np.random.randint(low=-10, high=10, size=1)[0]
    rand_delta = np.random.randint(0, 10) # randint between 0 and 10
    sample2 = np.random.normal(loc=mean+rand_delta,
                               scale=sd*(1+np.random.random(1)[0]),
                               size=n+rand_n2)
    error_count = 0
    for i in range(1, nreps):
        try:
            results = bst.bootstrap(sample1, sample2,
                alpha_level=alpha, paired=False)
            assert rand_delta >= results.bca_ci_low
            assert rand_delta <= results.bca_ci_high
            assert rand_delta >= results.pct_ci_low
            assert rand_delta <= results.pct_ci_high

        except AssertionError:
            error_count += 1
        sys.stdout.write('\r{0} runs of {1};  errors so far={2}'.format(i+1,
            nreps, error_count))
        sys.stdout.flush()
    print('\nNumber of CIs not capturing the mean is {}'.format(error_count))
    assert error_count < nreps*alpha
