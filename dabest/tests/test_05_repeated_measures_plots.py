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
from .utils import create_demo_dataset



df = create_demo_dataset()

sequential = load(df, idx=("Control 1", "Test 1", "Test 2", "Test 3"),
                    paired = "sequential", id_col = "ID")

baseline   = load(df, idx=("Control 1", "Test 1", "Test 2", "Test 3"),
                    paired= "baseline", id_col="ID")

shared_control = load(df, idx=("Control 1", "Test 1", "Test 2", "Test 3"))


multi_sequential = load(df, idx=(("Control 1", "Test 1", "Test 2", "Test 3"),
                                 ("Control 2", "Test 4", "Test 5", "Test 6")),
                    paired = "sequential", id_col = "ID")


multi_baseline = load(df, idx=(("Control 1", "Test 1", "Test 2", "Test 3"),
                                 ("Control 2", "Test 4", "Test 5", "Test 6")),
                            paired="baseline", id_col="ID")


multi_shared_control = load(df, idx=(("Control 1", "Test 1", "Test 2", "Test 3"),
                                 ("Control 2", "Test 4", "Test 5", "Test 6")))


@pytest.mark.mpl_image_compare(tolerance=10)
def test_29_cummings_sequential_repeated_measures_meandiff():
    return sequential.mean_diff.plot(fig_size=(8,6));



@pytest.mark.mpl_image_compare(tolerance=10)
def test_30_cummings_sequential_repeated_measures_mediandiff():
    return sequential.median_diff.plot(fig_size=(8,6));



@pytest.mark.mpl_image_compare(tolerance=10)
def test_31_cummings_sequential_repeated_measures_hedges_g():
    return sequential.hedges_g.plot(fig_size=(8,6));



@pytest.mark.mpl_image_compare(tolerance=10)
def test_32_cummings_baseline_repeated_measures_meandiff():
    return baseline.mean_diff.plot(fig_size=(8,6));



@pytest.mark.mpl_image_compare(tolerance=10)
def test_33_cummings_baseline_repeated_measures_mediandiff():
    return baseline.median_diff.plot(fig_size=(8,6));



@pytest.mark.mpl_image_compare(tolerance=10)
def test_34_cummings_baseline_repeated_measures_hedges_g():
    return baseline.hedges_g.plot(fig_size=(8,6));



@pytest.mark.mpl_image_compare(tolerance=10)
def test_35_cummings_shared_control_meandiff():
    return shared_control.mean_diff.plot(fig_size=(8,6));



@pytest.mark.mpl_image_compare(tolerance=10)
def test_36_cummings_shared_control_mediandiff():
    return shared_control.median_diff.plot(fig_size=(8,6));



@pytest.mark.mpl_image_compare(tolerance=10)
def test_37_cummings_shared_control_hedges_g():
    return shared_control.hedges_g.plot(fig_size=(8,6));



@pytest.mark.mpl_image_compare(tolerance=10)
def test_38_cummings_multi_group_sequential():
    return multi_sequential.mean_diff.plot(fig_size=(15,6));



@pytest.mark.mpl_image_compare(tolerance=10)
def test_39_cummings_multi_group_baseline():
    return multi_baseline.mean_diff.plot(fig_size=(15, 6));



@pytest.mark.mpl_image_compare(tolerance=10)
def test_40_cummings_multi_group_shared_control():
    return multi_shared_control.mean_diff.plot(fig_size=(15,6));



@pytest.mark.mpl_image_compare(tolerance=10)
def test_41_cummings_ylabel():
    return multi_baseline.mean_diff.plot(swarm_label="This is my\nrawdata",
                                   contrast_label="The bootstrap\ndistribtions!");



@pytest.mark.mpl_image_compare(tolerance=10)
def test_42_repeated_measures_color():
    return multi_sequential.mean_diff.plot(color_col="Gender", fig_size=(15,6));



@pytest.mark.mpl_image_compare(tolerance=10)
def test_43_repeated_measures_change_palette_a():
    return multi_sequential.mean_diff.plot(fig_size=(15, 6),
                                       color_col="Gender",
                                       custom_palette="Dark2");



@pytest.mark.mpl_image_compare(tolerance=10)
def test_44_repeated_measures_dot_sizes():
    return multi_sequential.mean_diff.plot(show_pairs=False,raw_marker_size=3,
                                       es_marker_size=12);



@pytest.mark.mpl_image_compare(tolerance=10)
def test_45_repeated_measures_change_ylims():
    return multi_sequential.mean_diff.plot(swarm_ylim=(0, 5),
                                       contrast_ylim=(-2, 2));



@pytest.mark.mpl_image_compare(tolerance=10)
def test_46_repeated_measures_invert_ylim():
    return multi_sequential.mean_diff.plot(contrast_ylim=(2, -2),
                                       contrast_label="More negative is better!");


