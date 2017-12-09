# #! /usr/bin/env python


import pytest
import sys
import numpy as np
import scipy as sp
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

@pytest.fixture
def single_sample_bootstrap(abs_tol, mean=100, sd=10, n=20, alpha=0.05):
    sample = np.random.normal(loc=mean,
                               scale=sd*np.random.random(1)[0],
                               size=n)
    obs_mean = np.mean(sample)
    # Get the t-statistic, and use it to compute the desired CI.
    ci = sp.stats.t.ppf(1-alpha/2, n-1) * sample.std()/np.sqrt(n-1)

    results = bst.bootstrap(sample, alpha_level=alpha)

    assert results.summary == pytest.approx(obs_mean, abs=abs_tol)
    assert results.bca_ci_low == pytest.approx(obs_mean - ci, abs=abs_tol)
    assert results.bca_ci_high == pytest.approx(obs_mean + ci, abs=abs_tol)
    assert results.pct_ci_low == pytest.approx(obs_mean - ci, abs=abs_tol)
    assert results.pct_ci_high == pytest.approx(obs_mean + ci, abs=abs_tol)


@pytest.fixture
def difference_bootstrap(is_paired, abs_tol, mean=100, sd=10, n=20, alpha=0.05):
    rand_delta = np.random.random(1)
    if is_paired is True:
        # Assume equal variances here
        # given that the samples are supposed to be paired.
        sample1 = np.random.normal(loc=mean,
                                   scale=sd,#*(1+np.random.random(1)[0]),
                                   size=n)
        sample2 = np.random.normal(loc=mean+rand_delta,
                                   scale=sd,#*(1+np.random.random(1)[0]),
                                   size=n)
        diffs = sample2 - sample1
        obs_mean = np.mean(diffs)
        # Get the t-statistic, and use it to compute a desired CI.
        ci = sp.stats.t.ppf(1-alpha/2, len(diffs)-1) * diffs.std()/np.sqrt(len(diffs)-1)

    else:
        rand_n1 = np.random.randint(low=-10, high=10, size=1)[0]
        sample1 = np.random.normal(loc=mean,
                                   scale=sd*(1+np.random.random(1)[0]),
                                   size=n+rand_n1)
        rand_n2 = np.random.randint(low=-10, high=10, size=1)[0]
        sample2 = np.random.normal(loc=mean+rand_delta,
                                   scale=sd*(1+np.random.random(1)[0]),
                                   size=n+rand_n2)
        obs_mean = np.mean(sample2) - np.mean(sample1)
        total_Ns = n*2 + rand_n1 + rand_n2
        sd1 = sample1.var()/len(sample1)
        sd2 = sample2.var()/len(sample2)
        mean_diff_sd = np.sqrt(sd1 + sd2)
        # Get the t-statistic, and use it to compute a desired CI.
        ci = sp.stats.t.ppf(1-alpha/2, total_Ns-1) * mean_diff_sd

    results = bst.bootstrap(sample1, sample2,
        alpha_level=alpha, paired=is_paired)

    assert results.summary == obs_mean
    assert results.bca_ci_low == pytest.approx(obs_mean - ci, abs=abs_tol)
    assert results.bca_ci_high == pytest.approx(obs_mean + ci, abs=abs_tol)
    assert results.pct_ci_low == pytest.approx(obs_mean - ci, abs=abs_tol)
    assert results.pct_ci_high == pytest.approx(obs_mean + ci, abs=abs_tol)

@pytest.fixture
def bootstrap_difference_bootstrap(is_paired=False,
    abstol=1.75,
    alpha=0.05, nreps=500):
    error_count = 0
    for i in range(1, nreps):
        try:
            difference_bootstrap(abs_tol=abstol, alpha=alpha,
                is_paired=is_paired)
        except AssertionError:
            error_count += 1
        sys.stdout.write('\r{0} runs of {1};  errors so far={2}'.format(i,
            nreps, error_count))
        sys.stdout.flush()
    print('\nErrors with abs tol={0} is {1}'.format(abstol, error_count))
    assert error_count < nreps*alpha

# Start tests below.

def test_unpaired(expt_groups_count=5):
    # Create dummy data for testing.
    test_data = create_dummy_dataset(expt_groups=expt_groups_count)
    # Now, create all pairs of control-expt tuples.
    test_tuples = make_test_tuples(test_data)
    # Run tests.
    print('testing unpaired')
    for t in test_tuples:
        print(t)
        unpaired(df=test_data, control=t[0], expt=t[1])

def test_paired(expt_groups_count=5):
    # Create dummy data for testing.
    test_data = create_dummy_dataset(expt_groups=expt_groups_count)
    # Now, create all pairs of control-expt tuples.
    test_tuples = make_test_tuples(test_data)
    # Run tests.
    print('testing paired')
    for t in test_tuples:
        print(t)
        paired(df=test_data, control=t[0], expt=t[1])

def test_single_sample_bootstrap(abstol=1.25, alpha=0.05, nreps=1000):
    error_count = 0
    for i in range(1, nreps):
        try:
            single_sample_bootstrap(abs_tol=abstol, alpha=alpha)
        except AssertionError:
            error_count += 1
        sys.stdout.write('\r{0} runs of {1};  errors so far={2}'.format(i,
            nreps, error_count))
        sys.stdout.flush()
    print('\nErrors with abs tol={0} is {1}'.format(abstol, error_count))
    assert error_count < nreps*alpha

def test_bootstrap_difference_bootstrap_paired(is_paired=True,
    abstol=1.75,
    alpha=0.05, nreps=500):

    bootstrap_difference_bootstrap(is_paired=is_paired,
        abstol=abstol,
        alpha=alpha, nreps=nreps)


def test_bootstrap_difference_bootstrap_unpaired(is_paired=False,
    abstol=1.75,
    alpha=0.05, nreps=500):

    bootstrap_difference_bootstrap(is_paired=is_paired,
        abstol=abstol,
        alpha=alpha, nreps=nreps)
