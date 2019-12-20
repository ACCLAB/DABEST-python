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

two_groups_unpaired = load(df, idx=("Control 1", "Test 1"))

two_groups_paired   = load(df, idx=("Control 1", "Test 1"), 
                           paired=True, id_col="ID")
                           
multi_2group = load(df, idx=(("Control 1", "Test 1",),
                                     ("Control 2", "Test 2"))
                            )
                                   
multi_2group_paired = load(df, idx=(("Control 1", "Test 1"),
                                           ("Control 2", "Test 2")),
                                  paired=True, id_col="ID")

shared_control = load(df, idx=("Control 1", "Test 1",
                                      "Test 2", "Test 3",
                                      "Test 4", "Test 5", "Test 6")
                             )
                             
multi_groups = load(df, idx=(("Control 1", "Test 1",),
                                     ("Control 2", "Test 2","Test 3"),
                                     ("Control 3", "Test 4","Test 5", "Test 6")
                                   ))



@pytest.mark.mpl_image_compare
def test_01_gardner_altman_unpaired_meandiff():
    return two_groups_unpaired.mean_diff.plot();



@pytest.mark.mpl_image_compare
def test_02_gardner_altman_unpaired_mediandiff():
    return two_groups_unpaired.median_diff.plot();



@pytest.mark.mpl_image_compare
def test_03_gardner_altman_unpaired_hedges_g():
    return two_groups_unpaired.hedges_g.plot();



@pytest.mark.mpl_image_compare
def test_04_gardner_altman_paired_meandiff():
    return two_groups_paired.mean_diff.plot();



@pytest.mark.mpl_image_compare
def test_04_gardner_altman_paired_hedges_g():
    return two_groups_paired.hedges_g.plot();


    
@pytest.mark.mpl_image_compare
def test_05_cummings_two_group_unpaired_meandiff():
    return two_groups_unpaired.mean_diff.plot(fig_size=(4, 6),
                                              float_contrast=False);



@pytest.mark.mpl_image_compare
def test_06_cummings_two_group_paired_meandiff():
    return two_groups_paired.mean_diff.plot(fig_size=(6, 6),
                                            float_contrast=False);



@pytest.mark.mpl_image_compare
def test_07_cummings_multi_group_unpaired():
    return multi_2group.mean_diff.plot();



@pytest.mark.mpl_image_compare
def test_08_cummings_multi_group_paired():
    return multi_2group_paired.mean_diff.plot(fig_size=(6, 6));



@pytest.mark.mpl_image_compare
def test_09_cummings_shared_control():
    return shared_control.mean_diff.plot();



@pytest.mark.mpl_image_compare
def test_10_cummings_multi_groups():
    return multi_groups.mean_diff.plot();



@pytest.mark.mpl_image_compare(tolerance=20)
def test_11_inset_plots():
    
    # Load the iris dataset. Requires internet access.
    iris = pd.read_csv("https://github.com/mwaskom/seaborn-data/raw/master/iris.csv")
    iris_melt = pd.melt(iris.reset_index(), 
                        id_vars=["species", "index"], var_name="metric")
                        
                        
                        
    # Load the above data into `dabest`.
    iris_dabest1 = load(data=iris, x="species", y="petal_width",
                              idx=("setosa", "versicolor", "virginica"))

    iris_dabest2 = load(data=iris, x="species", y="sepal_width",
                              idx=("setosa", "versicolor"))

    iris_dabest3 = load(data=iris_melt[iris_melt.species=="setosa"], 
                        x="metric", y="value",
                        idx=("sepal_length", "sepal_width"),
                        paired=True, id_col="index")
                        
                        
                        
    # Create Figure.
    fig, ax = plt.subplots(nrows=2, ncols=2, 
                           figsize=(15, 15), 
                           gridspec_kw={"wspace":0.5})

    iris_dabest1.mean_diff.plot(ax=ax.flat[0]);

    iris_dabest2.mean_diff.plot(ax=ax.flat[1]);

    iris_dabest3.mean_diff.plot(ax=ax.flat[2]);

    iris_dabest3.mean_diff.plot(ax=ax.flat[3], float_contrast=False);
    
    return fig
    
    
    
@pytest.mark.mpl_image_compare
def test_12_gardner_altman_ylabel():
    return two_groups_unpaired.mean_diff.plot(swarm_label="This is my\nrawdata",  
                                   contrast_label="The bootstrap\ndistribtions!");
                                   


@pytest.mark.mpl_image_compare
def test_13_multi_2group_color():
    return multi_2group.mean_diff.plot(color_col="Gender");



@pytest.mark.mpl_image_compare
def test_14_gardner_altman_paired_color():
    return two_groups_paired.mean_diff.plot(fig_size=(6, 6),
                                            color_col="Gender");



@pytest.mark.mpl_image_compare
def test_15_change_palette_a():
    return multi_2group.mean_diff.plot(fig_size=(8, 6),
                                       color_col="Gender", 
                                       custom_palette="Dark2");



@pytest.mark.mpl_image_compare
def test_16_change_palette_b():
    return multi_2group.mean_diff.plot(custom_palette="Paired");



my_color_palette = {"Control 1" : "blue",    
                "Test 1"    : "purple",
                "Control 2" : "#cb4b16",     # This is a hex string.
                "Test 2"    : (0., 0.7, 0.2) # This is a RGB tuple.
               }
               
@pytest.mark.mpl_image_compare
def test_17_change_palette_c():
    return multi_2group.mean_diff.plot(custom_palette=my_color_palette);



@pytest.mark.mpl_image_compare
def test_18_desat():
    return multi_2group.mean_diff.plot(custom_palette=my_color_palette, 
                            swarm_desat=0.75, 
                            halfviolin_desat=0.25);



@pytest.mark.mpl_image_compare                            
def test_19_dot_sizes():
    return multi_2group.mean_diff.plot(raw_marker_size=3, 
                                       es_marker_size=12);



@pytest.mark.mpl_image_compare
def test_20_change_ylims():
    return multi_2group.mean_diff.plot(swarm_ylim=(0, 5), 
                                       contrast_ylim=(-2, 2));



@pytest.mark.mpl_image_compare
def test_21_invert_ylim():
    return multi_2group.mean_diff.plot(contrast_ylim=(2, -2), 
                                       contrast_label="More negative is better!");



@pytest.mark.mpl_image_compare
def test_22_ticker_gardner_altman():

    f = two_groups_unpaired.mean_diff.plot()

    rawswarm_axes = f.axes[0]
    contrast_axes = f.axes[1]

    rawswarm_axes.yaxis.set_major_locator(Ticker.MultipleLocator(1))
    rawswarm_axes.yaxis.set_minor_locator(Ticker.MultipleLocator(0.5))

    contrast_axes.yaxis.set_major_locator(Ticker.MultipleLocator(0.5))
    contrast_axes.yaxis.set_minor_locator(Ticker.MultipleLocator(0.25))
    
    return f



@pytest.mark.mpl_image_compare
def test_23_ticker_cumming():
    f = multi_2group.mean_diff.plot(swarm_ylim=(0,6),
                               contrast_ylim=(-3, 1))

    rawswarm_axes = f.axes[0]
    contrast_axes = f.axes[1]

    rawswarm_axes.yaxis.set_major_locator(Ticker.MultipleLocator(2))
    rawswarm_axes.yaxis.set_minor_locator(Ticker.MultipleLocator(1))

    contrast_axes.yaxis.set_major_locator(Ticker.MultipleLocator(0.5))
    contrast_axes.yaxis.set_minor_locator(Ticker.MultipleLocator(0.25))
    
    return f



np.random.seed(9999)
Ns = [20, 10, 21, 20]
c1 = pd.DataFrame({'Control':norm.rvs(loc=3, scale=0.4, size=Ns[0])})
t1 = pd.DataFrame({'Test 1': norm.rvs(loc=3.5, scale=0.5, size=Ns[1])})
t2 = pd.DataFrame({'Test 2': norm.rvs(loc=2.5, scale=0.6, size=Ns[2])})
t3 = pd.DataFrame({'Test 3': norm.rvs(loc=3, scale=0.75, size=Ns[3])})
wide_df = pd.concat([c1, t1, t2, t3],axis=1)


long_df = pd.melt(wide_df, 
              value_vars=["Control", "Test 1", "Test 2", "Test 3"],
                value_name="value",
                var_name="group")
long_df['dummy'] = np.repeat(np.nan, len(long_df))



@pytest.mark.mpl_image_compare
def test_24_wide_df_nan():

    wide_df_dabest = load(wide_df, 
                          idx=("Control", "Test 1", "Test 2", "Test 3")
                          )

    return wide_df_dabest.mean_diff.plot(); 



@pytest.mark.mpl_image_compare
def test_25_long_df_nan():

    long_df_dabest = load(long_df, x="group", y="value",
                          idx=("Control", "Test 1", "Test 2", "Test 3")
                          )

    return long_df_dabest.mean_diff.plot(); 
    
    
    
@pytest.mark.mpl_image_compare
def test_26_slopegraph_kwargs():
    
    return two_groups_paired.mean_diff.plot(
            slopegraph_kwargs=dict(linestyle='dotted')
            );
    


@pytest.mark.mpl_image_compare
def test_27_gardner_altman_reflines_kwargs():
    
    return two_groups_unpaired.mean_diff.plot(
            reflines_kwargs=dict(linestyle='dotted')
            );



@pytest.mark.mpl_image_compare
def test_28_unpaired_cumming_reflines_kwargs():
    
    return two_groups_unpaired.mean_diff.plot(
            fig_size=(12,10),
            float_contrast=False,
            reflines_kwargs=dict(linestyle='dotted', 
                                 linewidth=2),
            contrast_ylim=(-1, 1)
            );



@pytest.mark.mpl_image_compare
def test_28_paired_cumming_slopegraph_reflines_kwargs():
    
    return two_groups_paired.mean_diff.plot(float_contrast=False,
                                 color_col="Gender",
                                 slopegraph_kwargs=dict(linestyle='dotted'),
                                 reflines_kwargs=dict(linestyle='dashed', 
                                                      linewidth=2),
                                 contrast_ylim=(-1, 1)
                                 );
    
    
@pytest.mark.mpl_image_compare
def test_99_style_sheets():
    # Perform this test last so we don't have to reset the plot style.
    plt.style.use("dark_background")
    
    return multi_2group.mean_diff.plot();
    
