#!/usr/bin/env python3

# -*- coding: utf-8 -*-


import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as Ticker


from .._api import load
from .utils import create_demo_dataset_delta



df = create_demo_dataset_delta()

unpaired = load(data = df, x = ["Light", "Genotype"], y = "Y", delta2 = True, 
                experiment = "Experiment")

baseline = load(data = df, x = ["Light", "Genotype"], y = "Y", delta2 = True, 
                experiment = "Experiment",
                paired="baseline", id_col="ID")

sequential = load(data = df, x = ["Light", "Genotype"], y = "Y", delta2 = True, 
                experiment = "Experiment",
                paired="sequential", id_col="ID")


@pytest.mark.mpl_image_compare(tolerance=10)
def test_47_cummings_unpaired_delta_delta_meandiff():
    return unpaired.mean_diff.plot(fig_size=(12, 8), raw_marker_size=4);


@pytest.mark.mpl_image_compare(tolerance=10)
def test_48_cummings_sequential_delta_delta_meandiff():
    return sequential.mean_diff.plot();


@pytest.mark.mpl_image_compare(tolerance=10)
def test_49_cummings_baseline_delta_delta_meandiff():
    return baseline.mean_diff.plot();


@pytest.mark.mpl_image_compare(tolerance=10)
def test_50_delta_plot_ylabel():
    return baseline.mean_diff.plot(swarm_label="This is my\nrawdata",
                                   contrast_label="The bootstrap\ndistribtions!", 
                                   delta2_label="This is delta!");


@pytest.mark.mpl_image_compare(tolerance=10)
def test_51_delta_plot_change_palette_a():
    return sequential.mean_diff.plot(custom_palette="Dark2");


@pytest.mark.mpl_image_compare(tolerance=10)
def test_52_delta_dot_sizes():
    return sequential.mean_diff.plot(show_pairs=False,raw_marker_size=3,
                                       es_marker_size=12);


@pytest.mark.mpl_image_compare(tolerance=10)
def test_53_delta_change_ylims():
    return sequential.mean_diff.plot(swarm_ylim=(0, 5),
                                       contrast_ylim=(-2, 2),
                                       fig_size=(15,6));


@pytest.mark.mpl_image_compare(tolerance=10)
def test_54_delta_invert_ylim():
    return sequential.mean_diff.plot(contrast_ylim=(2, -2),
                                       contrast_label="More negative is better!");


@pytest.mark.mpl_image_compare(tolerance=10)
def test_55_delta_median_diff():
    return sequential.median_diff.plot();


@pytest.mark.mpl_image_compare(tolerance=10)
def test_56_delta_cohens_d():
    return unpaired.cohens_d.plot();


@pytest.mark.mpl_image_compare(tolerance=10)
def test_57_delta_show_delta2():
    return unpaired.mean_diff.plot(show_delta2=False);


@pytest.mark.mpl_image_compare(tolerance=10)
def test_58_delta_axes_invert_ylim():
    return unpaired.mean_diff.plot(delta2_ylim=(2, -2),
                                   delta2_label="More negative is better!");

                            
@pytest.mark.mpl_image_compare(tolerance=10)
def test_59_delta_axes_invert_ylim_not_showing_delta2():
    return unpaired.mean_diff.plot(delta2_ylim=(2, -2),
                                   delta2_label="More negative is better!",
                                   show_delta2=False);