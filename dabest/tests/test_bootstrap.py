# #! /usr/bin/env python


import pytest
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
    # Now, create all pairs of control-expt tuples.
    con = np.repeat(df.columns[0], len(df.columns)-1)
    expt = [c for c in df.columns[1:]]
    zipped = zip(con, expt)
    test_tuples = list(zipped)

    return test_tuples

# Start tests below.


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
