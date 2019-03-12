# #! /usr/bin/env python

# Load Libraries

import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
from .._api import load
from .utils import create_dummy_dataset, get_swarm_yspans



def test_gardner_altman_unpaired():

    base_mean = np.random.randint(10, 101)
    seed, ptp, df = create_dummy_dataset(base_mean=base_mean)
    print('\nSeed = {}; base mean = {}'.format(seed, base_mean))

    for c in df.columns[1:-1]:
        print('{}...'.format(c))

        # Create Gardner-Altman plot.
        rand_swarm_ylim = (np.random.uniform(base_mean-10, base_mean, 1),
                           np.random.uniform(base_mean, base_mean+10, 1))
        two_group_unpaired = load(df, idx=(df.columns[0], c))
        f1 = two_group_unpaired.mean_diff.plot(swarm_ylim=rand_swarm_ylim,
                                               swarm_label="Raw swarmplot...",
                                               contrast_label="Contrast!")

        rawswarm_axes = f1.axes[0]
        contrast_axes = f1.axes[1]

        # Check ylims match the desired ones.
        assert rawswarm_axes.get_ylim()[0] == pytest.approx(rand_swarm_ylim[0])
        assert rawswarm_axes.get_ylim()[1] == pytest.approx(rand_swarm_ylim[1])


        # Check each swarmplot group matches canonical seaborn swarmplot.
        _, swarmplt = plt.subplots(1)
        swarmplt.set_ylim(rand_swarm_ylim)
        sns.swarmplot(data=df[[df.columns[0], c]], ax=swarmplt)
        sns_yspans = []
        for coll in swarmplt.collections:
            sns_yspans.append(get_swarm_yspans(coll))
        dabest_yspans = [get_swarm_yspans(coll)
                        for coll in rawswarm_axes.collections]
        for j, span in enumerate(sns_yspans):
            assert span == pytest.approx(dabest_yspans[j])

        # Check xtick labels.
        swarm_xticks = [a.get_text() for a in rawswarm_axes.get_xticklabels()]
        assert swarm_xticks[0] == "{}\nN = 30".format(df.columns[0])
        assert swarm_xticks[1] == "{}\nN = 30".format(c)

        contrast_xticks = [a.get_text() for a in contrast_axes.get_xticklabels()]
        assert contrast_xticks[1] == "{}\nminus\n{}".format(c, df.columns[0])

        # Check ylabels.
        assert rawswarm_axes.get_ylabel() == "Raw swarmplot..."
        assert contrast_axes.get_ylabel() == "Contrast!"





def test_cummings_unpaired():
    base_mean = np.random.randint(-5, 5)
    seed, ptp, df = create_dummy_dataset(base_mean=base_mean, expt_groups=7)
    print('\nSeed = {}; base mean = {}'.format(seed, base_mean))

    IDX = (('0','5'), ('3','2'), ('4', '1', '6'))
    multi_2group_unpaired = load(df, idx=IDX)

    rand_swarm_ylim = (np.random.uniform(base_mean-10, base_mean, 1),
                       np.random.uniform(base_mean, base_mean+10, 1))
    rand_contrast_ylim = (-base_mean/3, base_mean/3)

    f1 = multi_2group_unpaired.mean_diff.plot(swarm_ylim=rand_swarm_ylim,
                                              contrast_ylim=rand_contrast_ylim,
                                              swarm_label="Raw swarmplot!",
                                              contrast_label="Contrast...")

    rawswarm_axes = f1.axes[0]
    contrast_axes = f1.axes[1]

    # Check ylims match the desired ones.
    assert rawswarm_axes.get_ylim()[0] == pytest.approx(rand_swarm_ylim[0])
    assert rawswarm_axes.get_ylim()[1] == pytest.approx(rand_swarm_ylim[1])
    
    # This needs to be rounded, because if the base mean is 0,
    # the ylim might be -0.001, which will not match 0.
    if base_mean == 0:
        ylim_low = np.round(contrast_axes.get_ylim()[0])
    else:
        ylim_low = contrast_axes.get_ylim()[0]
    assert ylim_low == pytest.approx(rand_contrast_ylim[0])
    
    assert contrast_axes.get_ylim()[1] == pytest.approx(rand_contrast_ylim[1])

    # Check xtick labels.
    idx_flat = [g for t in IDX for g in t]
    swarm_xticks = [a.get_text() for a in rawswarm_axes.get_xticklabels()]
    for j, xtick in enumerate(swarm_xticks):
        assert xtick == "{}\nN = 30".format(idx_flat[j])

    contrast_xticks = [a.get_text() for a in contrast_axes.get_xticklabels()]
    assert contrast_xticks[1] == "5\nminus\n0"
    assert contrast_xticks[3] == "2\nminus\n3"
    assert contrast_xticks[5] == "1\nminus\n4"
    assert contrast_xticks[6] == "6\nminus\n4"

    # Check ylabels.
    assert rawswarm_axes.get_ylabel() == "Raw swarmplot!"
    assert contrast_axes.get_ylabel() == "Contrast..."





def test_gardner_altman_paired():
    base_mean = np.random.randint(-5, 5)
    seed, ptp, df = create_dummy_dataset(base_mean=base_mean)


    # Check that the plot data matches the raw data.
    two_group_paired = load(df, idx=("1", "2"), id_col="idcol", paired=True)
    f1 = two_group_paired.mean_diff.plot()
    rawswarm_axes = f1.axes[0]
    contrast_axes = f1.axes[1]
    assert df['1'].tolist() == [l.get_ydata()[0] for l in rawswarm_axes.lines]
    assert df['2'].tolist() == [l.get_ydata()[1] for l in rawswarm_axes.lines]


    # Check that id_col must be specified.
    err_to_catch = "`id_col` must be specified if `is_paired` is set to True."
    with pytest.raises(IndexError, match=err_to_catch):
        this_will_not_work = load(df, idx=("1", "2"), paired=True)
