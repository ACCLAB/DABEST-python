# #! /usr/bin/env python


# Load Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pytest
from .. import api

# Fixtures.
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
        df.loc[:,c] =(df[c] * np.random.random()) + np.random.random()
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
