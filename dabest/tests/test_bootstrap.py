# #! /usr/bin/env python

# Load Libraries
from .. import bootstrap_tools as bst

# Check that pytest itself is working.
def func(x):
    return x + 1

def test_answer():
    assert func(3) == 4

def create_dummy_dataset(n=50):
    import pandas as pd
    import numpy as np

    # Dummy dataset
    Ns = n
    dataset = list()
    for seed in [10,11,12,13,14,15]:
        # fix the seed so we get the same numbers each time.
        np.random.seed(seed)
        dataset.append(np.random.randn(Ns))
    df = pd.DataFrame(dataset).T
    cols = ['Control','Group1','Group2','Group3','Group4','Group5']
    df.columns = cols
    # Create some upwards/downwards shifts.
    df['Group2'] = df['Group2'] - 0.1
    df['Group3'] = df['Group3'] + 0.2
    df['Group4'] = (df['Group4']*1.1) + 4
    df['Group5'] = (df['Group5']*1.1) - 1
    # Add gender column for color.
    df['Gender'] = np.concatenate([np.repeat('Male', Ns/2),
                                  np.repeat('Female', Ns/2)])

    return df

test_data = create_dummy_dataset()

def test_control_vs_group1_unpaired():
    res = bst.bootstrap(test_data['Control'],
                         test_data['Group1'])
    assert(res.is_difference == True)
    assert(res.is_paired == False)
    assert(res.pvalue_2samp_ind_ttest == 0.29644969216077316)
    assert(res.summary == -0.1913038699392576) 
