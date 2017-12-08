# #! /usr/bin/env python

# # Check that pytest itself is working.
# def func(x):
#     return x + 1
#
# def test_answer():
#     assert func(3) == 4

# Load Libraries
import pytest

@pytest.fixture
def create_dummy_dataset(n=50, expt_groups=6):
    import pandas as pd
    import numpy as np
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
    # Add gender column for color.
    df['Gender'] = np.concatenate([np.repeat('Male', Ns/2),
                                   np.repeat('Female', Ns/2)])

    return df
#

#
# # f.savefig('testfig.svg', **savefig_kwargs)
def test_Gardner_Altman_unpaired(df):
    from .. import api
    df = create_dummy_dataset()

    return api.plot(data=df,
                    idx=('Control','Group1'))


def Gardner_Altman_paired(df):
    from .. import api

    return api.plot(data=df,
                       idx=('Control','Group1'),
                       paired=True)

def Cumming_two_group_unpaired(df):
    from .. import api

    return api.plot(data=df,
                      idx=('Control','Group1'),
                      float_contrast=True)

def Cumming_two_group_paired(df):
    from .. import api

    return api.plot(data=df,
                       idx=('Control','Group1'),
                       paired=True,
                       float_contrast=True)

def custom_swarm_label(df):
    from .. import api

    return api.plot(data=df,
                       idx=('Control','Group1'),
                       swarm_label='my swarm',
                       contrast_label='contrast')

def custom_contrast_label(df):
    from .. import api

    return api.plot(data=df,
                       idx=('Control','Group1'),
                       contrast_label='contrast')

def with_color_col(df):
    from .. import api

    return api.plot(data=df,
                       idx=('Control','Group1'),
                       color_col='Gender')

def with_custom_palette(df):
    import dabest

    return dabest.plot(data=df,
                       idx=('Control','Group1'),
                       color_col='Gender',
                       custom_palette={'Male':'blue',
                                       'Female':'red'}
                        )
