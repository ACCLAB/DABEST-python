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
    # Turn columns into strings
    df.columns = [str(c) for c in df.columns]
    # Add gender column for color.
    df['Gender'] = np.concatenate([np.repeat('Male', Ns/2),
                                   np.repeat('Female', Ns/2)])

    return df

@pytest.fixture
def get_swarm_yspans(coll, round_result=False, decimals=12):
    """
    Given a matplotlib Collection, will obtain the y spans
    for the collection. Will return None if this fails.

    Modified from `get_swarm_spans` in plot_tools.py.
    """
    import numpy as np
    _, y = np.array(coll.get_offsets()).T
    try:
        if round_result:
            return np.around(y.min(), decimals), np.around(y.max(),decimals)
        else:
            return y.min(), y.max()
    except ValueError:
        return None

# savefig_kwargs = {'transparent': True,
#                  'frameon': False,
#                  'bbox_inches': 'tight',
#                  'format': 'svg'}

def test_Gardner_Altman_unpaired():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pytest
    from .. import api

    df = create_dummy_dataset()
    for c in df.columns[1:-1]:
        f1, swarmplt = plt.subplots(figsize=(10, 10))
        sns.swarmplot(data=df[[df.columns[0], c]],
                      ax=swarmplt)
        sns_yspans = []
        for coll in swarmplt.collections:
            sns_yspans.append(get_swarm_yspans(coll))

        f2, b = api.plot(data=df,
                           fig_size=(12.5, 11),
                           idx=(df.columns[0], c))
        dabest_yspans = []
        for coll in f2.axes[0].collections:
            dabest_yspans.append(get_swarm_yspans(coll))

        for j, span in enumerate(sns_yspans):
            assert span == pytest.approx(dabest_yspans[j])


# def Gardner_Altman_paired(df):
#     from .. import api
#
#     return api.plot(data=df,
#                        idx=('Control','Group1'),
#                        paired=True)
#
# def Cumming_two_group_unpaired(df):
#     from .. import api
#
#     return api.plot(data=df,
#                       idx=('Control','Group1'),
#                       float_contrast=True)
#
# def Cumming_two_group_paired(df):
#     from .. import api
#
#     return api.plot(data=df,
#                        idx=('Control','Group1'),
#                        paired=True,
#                        float_contrast=True)
#
# def custom_swarm_label(df):
#     from .. import api
#
#     return api.plot(data=df,
#                        idx=('Control','Group1'),
#                        swarm_label='my swarm',
#                        contrast_label='contrast')
#
# def custom_contrast_label(df):
#     from .. import api
#
#     return api.plot(data=df,
#                        idx=('Control','Group1'),
#                        contrast_label='contrast')
#
# def with_color_col(df):
#     from .. import api
#
#     return api.plot(data=df,
#                        idx=('Control','Group1'),
#                        color_col='Gender')
#
# def with_custom_palette(df):
#     import dabest
#
#     return dabest.plot(data=df,
#                        idx=('Control','Group1'),
#                        color_col='Gender',
#                        custom_palette={'Male':'blue',
#                                        'Female':'red'}
#                         )
