# #! /usr/bin/env python

# Load Libraries
import numpy as np
import scipy as sp
import matplotlib as mpl
mpl.use('Agg')


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pytest
from .. import api


# Fixtures.
@pytest.fixture
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



@pytest.fixture
def get_swarm_yspans(coll, round_result=False, decimals=12):
    """
    Given a matplotlib Collection, will obtain the y spans
    for the collection. Will return None if this fails.

    Modified from `get_swarm_spans` in plot_tools.py.
    """
    _, y = np.array(coll.get_offsets()).T
    try:
        if round_result:
            return np.around(y.min(), decimals), np.around(y.max(),decimals)
        else:
            return y.min(), y.max()
    except ValueError:
        return None



# Start tests.
def test_swarmspan():
    print('Testing swarmspan')
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



def test_ylims():
    print('Testing assignment of ylims')
    df = create_dummy_dataset()

    print('Testing assignment for Gardner-Altman plot')
    rand_swarm_ylim2 = (np.random.randint(-7, 0), np.random.randint(0, 7))
    f2, b2 = api.plot(data=df,
                   idx=(('0','1'),('2','3')),
                   float_contrast=True,
                   swarm_ylim=rand_swarm_ylim2)
    for i in range(0, int(len(f2.axes)/2)):
        assert f2.axes[i].get_ylim() == pytest.approx(rand_swarm_ylim2)

    print('Testing assignment of ylims for Cummings plot')
    rand_swarm_ylim1 = (np.random.randint(-7, 0), np.random.randint(0, 7))
    rand_contrast_ylim1 = (np.random.randint(-1, 0), np.random.randint(0, 1))
    f1, b1 = api.plot(data=df,
                   idx=(('0','1'),('2','3')),
                   float_contrast=False,
                   swarm_ylim=rand_swarm_ylim1,
                   contrast_ylim=rand_contrast_ylim1)
    for i in range(0, int(len(f1.axes)/2)):
        assert f1.axes[i].get_ylim() == pytest.approx(rand_swarm_ylim1)
    for i in range(int(len(f1.axes)/2), len(f1.axes)):
        assert f1.axes[i].get_ylim() == pytest.approx(rand_contrast_ylim1)



def test_ylabels():
    print('Testing assignment of ylabels')
    df = create_dummy_dataset()

    print('Testing ylabel assignment for Gardner-Altman plot')
    f1, _ = api.plot(data=df,
                     idx=(('0','1'),('2','3')),
                     float_contrast=True,
                     swarm_label="Hello",
                     contrast_label="World"
                    )
    assert f1.axes[0].get_ylabel() == 'Hello'

    print('Testing ylabel assignment for Cummings plot')
    f2, _ = api.plot(data=df,
                         idx=(('0','1'),('2','3')),
                         float_contrast=False,
                         swarm_label="Hello Again",
                         contrast_label="World\nFolks"
                        )
    assert f2.axes[0].get_ylabel() == "Hello Again"
    assert f2.axes[2].get_ylabel() == "World\nFolks"



def test_paired():
    print('Testing Gardner-Altman paired plotting')
    df = create_dummy_dataset()
    f, b = api.plot(data=df,
                   idx=('0','1'),
                   paired=True)
    axx = f.axes[0]
    assert df['0'].tolist() == [l.get_ydata()[0] for l in axx.lines]
    assert df['1'].tolist() == [l.get_ydata()[1] for l in axx.lines]
