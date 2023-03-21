import pytest
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.ticker as Ticker
import matplotlib.pyplot as plt

import os
import sys
cur_dir = os.getcwd()
pkg_rootdir = os.path.dirname(cur_dir)
if pkg_rootdir not in sys.path:
    sys.path.append(pkg_rootdir)
from dabest._api import load


def create_demo_dataset(seed=9999, N=20):
    
    import numpy as np
    import pandas as pd
    from scipy.stats import norm # Used in generation of populations.

    np.random.seed(9999) # Fix the seed so the results are replicable.
    # pop_size = 10000 # Size of each population.

    # Create samples
    c1 = norm.rvs(loc=3, scale=0.4, size=N)
    c2 = norm.rvs(loc=3.5, scale=0.75, size=N)
    c3 = norm.rvs(loc=3.25, scale=0.4, size=N)

    t1 = norm.rvs(loc=3.5, scale=0.5, size=N)
    t2 = norm.rvs(loc=2.5, scale=0.6, size=N)
    t3 = norm.rvs(loc=3, scale=0.75, size=N)
    t4 = norm.rvs(loc=3.5, scale=0.75, size=N)
    t5 = norm.rvs(loc=3.25, scale=0.4, size=N)
    t6 = norm.rvs(loc=3.25, scale=0.4, size=N)


    # Add a `gender` column for coloring the data.
    females = np.repeat('Female', N/2).tolist()
    males = np.repeat('Male', N/2).tolist()
    gender = females + males

    # Add an `id` column for paired data plotting. 
    id_col = pd.Series(range(1, N+1))

    # Combine samples and gender into a DataFrame.
    df = pd.DataFrame({'Control 1' : c1,     'Test 1' : t1,
                       'Control 2' : c2,     'Test 2' : t2,
                       'Control 3' : c3,     'Test 3' : t3,
                       'Test 4'    : t4,     'Test 5' : t5, 'Test 6' : t6,
                       'Gender'    : gender, 'ID'  : id_col
                      })
                      
    return df
df = create_demo_dataset()

two_groups_unpaired = load(df, idx=("Control 1", "Test 1"))

two_groups_paired   = load(df, idx=("Control 1", "Test 1"),
                           paired="baseline", id_col="ID")

multi_2group = load(df, idx=(("Control 1", "Test 1",),
                             ("Control 2", "Test 2"))
                    )

multi_2group_paired = load(df,
                            idx=(("Control 1", "Test 1"),
                                 ("Control 2", "Test 2")),
                            paired="baseline", id_col="ID")

shared_control = load(df, idx=("Control 1", "Test 1",
                                "Test 2", "Test 3",
                                "Test 4", "Test 5", "Test 6")
                    )

multi_groups = load(df, idx=(("Control 1", "Test 1",),
                             ("Control 2", "Test 2","Test 3"),
                             ("Control 3", "Test 4","Test 5", "Test 6")
                             )
                    )

multi_groups_baseline = load(df, idx=(("Control 1", "Test 1",),
                             ("Control 2", "Test 2","Test 3"),
                             ("Control 3", "Test 4","Test 5", "Test 6")
                             ), paired="baseline", id_col="ID"
                    )

multi_groups_sequential = load(df, idx=(("Control 1", "Test 1",),
                             ("Control 2", "Test 2","Test 3"),
                             ("Control 3", "Test 4","Test 5", "Test 6")
                             ), paired="sequential", id_col="ID"
                    )



@pytest.mark.mpl_image_compare(tolerance=10)
def test_01_gardner_altman_unpaired_meandiff():
    return two_groups_unpaired.mean_diff.plot();



@pytest.mark.mpl_image_compare(tolerance=10)
def test_02_gardner_altman_unpaired_mediandiff():
    return two_groups_unpaired.median_diff.plot();



@pytest.mark.mpl_image_compare(tolerance=10)
def test_03_gardner_altman_unpaired_hedges_g():
    return two_groups_unpaired.hedges_g.plot();



@pytest.mark.mpl_image_compare(tolerance=10)
def test_04_gardner_altman_paired_meandiff():
    return two_groups_paired.mean_diff.plot();



@pytest.mark.mpl_image_compare(tolerance=10)
def test_04_gardner_altman_paired_hedges_g():
    return two_groups_paired.hedges_g.plot();



@pytest.mark.mpl_image_compare(tolerance=10)
def test_05_cummings_two_group_unpaired_meandiff():
    return two_groups_unpaired.mean_diff.plot(fig_size=(4, 6),
                                              float_contrast=False);



@pytest.mark.mpl_image_compare(tolerance=10)
def test_06_cummings_two_group_paired_meandiff():
    return two_groups_paired.mean_diff.plot(fig_size=(6, 6),
                                            float_contrast=False);



@pytest.mark.mpl_image_compare(tolerance=10)
def test_07_cummings_multi_group_unpaired():
    return multi_2group.mean_diff.plot();



@pytest.mark.mpl_image_compare(tolerance=10)
def test_08_cummings_multi_group_paired():
    return multi_2group_paired.mean_diff.plot(fig_size=(6, 6));



@pytest.mark.mpl_image_compare(tolerance=10)
def test_09_cummings_shared_control():
    return shared_control.mean_diff.plot();



@pytest.mark.mpl_image_compare(tolerance=10)
def test_10_cummings_multi_groups():
    return multi_groups.mean_diff.plot();



@pytest.mark.mpl_image_compare(tolerance=10)
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
                        paired="baseline", id_col="index")



    # Create Figure.
    fig, ax = plt.subplots(nrows=2, ncols=2,
                           figsize=(15, 15),
                           gridspec_kw={"wspace":0.5})

    iris_dabest1.mean_diff.plot(ax=ax.flat[0]);

    iris_dabest2.mean_diff.plot(ax=ax.flat[1]);

    iris_dabest3.mean_diff.plot(ax=ax.flat[2]);

    iris_dabest3.mean_diff.plot(ax=ax.flat[3], float_contrast=False);

    return fig



@pytest.mark.mpl_image_compare(tolerance=10)
def test_12_gardner_altman_ylabel():
    return two_groups_unpaired.mean_diff.plot(swarm_label="This is my\nrawdata",
                                   contrast_label="The bootstrap\ndistribtions!");



@pytest.mark.mpl_image_compare(tolerance=10)
def test_13_multi_2group_color():
    return multi_2group.mean_diff.plot(color_col="Gender");



@pytest.mark.mpl_image_compare(tolerance=10)
def test_14_gardner_altman_paired_color():
    return two_groups_paired.mean_diff.plot(fig_size=(6, 6),
                                            color_col="Gender");



@pytest.mark.mpl_image_compare(tolerance=10)
def test_15_change_palette_a():
    return multi_2group.mean_diff.plot(fig_size=(8, 6),
                                       color_col="Gender",
                                       custom_palette="Dark2");



@pytest.mark.mpl_image_compare(tolerance=10)
def test_16_change_palette_b():
    return multi_2group.mean_diff.plot(custom_palette="Paired");



my_color_palette = {"Control 1" : "blue",
                "Test 1"    : "purple",
                "Control 2" : "#cb4b16",     # This is a hex string.
                "Test 2"    : (0., 0.7, 0.2) # This is a RGB tuple.
               }

@pytest.mark.mpl_image_compare(tolerance=10)
def test_17_change_palette_c():
    return multi_2group.mean_diff.plot(custom_palette=my_color_palette);



@pytest.mark.mpl_image_compare(tolerance=10)
def test_18_desat():
    return multi_2group.mean_diff.plot(custom_palette=my_color_palette,
                            swarm_desat=0.75,
                            halfviolin_desat=0.25);



@pytest.mark.mpl_image_compare(tolerance=10)
def test_19_dot_sizes():
    return multi_2group.mean_diff.plot(raw_marker_size=3,
                                       es_marker_size=12);



@pytest.mark.mpl_image_compare(tolerance=10)
def test_20_change_ylims():
    return multi_2group.mean_diff.plot(swarm_ylim=(0, 5),
                                       contrast_ylim=(-2, 2));



@pytest.mark.mpl_image_compare(tolerance=10)
def test_21_invert_ylim():
    return multi_2group.mean_diff.plot(contrast_ylim=(2, -2),
                                       contrast_label="More negative is better!");



@pytest.mark.mpl_image_compare(tolerance=10)
def test_22_ticker_gardner_altman():

    f = two_groups_unpaired.mean_diff.plot()

    rawswarm_axes = f.axes[0]
    contrast_axes = f.axes[1]

    rawswarm_axes.yaxis.set_major_locator(Ticker.MultipleLocator(1))
    rawswarm_axes.yaxis.set_minor_locator(Ticker.MultipleLocator(0.5))

    contrast_axes.yaxis.set_major_locator(Ticker.MultipleLocator(0.5))
    contrast_axes.yaxis.set_minor_locator(Ticker.MultipleLocator(0.25))

    return f



@pytest.mark.mpl_image_compare(tolerance=10)
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



@pytest.mark.mpl_image_compare(tolerance=10)
def test_24_wide_df_nan():

    wide_df_dabest = load(wide_df,
                          idx=("Control", "Test 1", "Test 2", "Test 3")
                          )

    return wide_df_dabest.mean_diff.plot();



@pytest.mark.mpl_image_compare(tolerance=10)
def test_25_long_df_nan():

    long_df_dabest = load(long_df, x="group", y="value",
                          idx=("Control", "Test 1", "Test 2", "Test 3")
                          )

    return long_df_dabest.mean_diff.plot();



@pytest.mark.mpl_image_compare(tolerance=10)
def test_26_slopegraph_kwargs():

    return two_groups_paired.mean_diff.plot(
            slopegraph_kwargs=dict(linestyle='dotted')
            );



@pytest.mark.mpl_image_compare(tolerance=10)
def test_27_gardner_altman_reflines_kwargs():

    return two_groups_unpaired.mean_diff.plot(
            reflines_kwargs=dict(linestyle='dotted')
            );



@pytest.mark.mpl_image_compare(tolerance=10)
def test_28_unpaired_cumming_reflines_kwargs():

    return two_groups_unpaired.mean_diff.plot(
            fig_size=(12,10),
            float_contrast=False,
            reflines_kwargs=dict(linestyle='dotted',
                                 linewidth=2),
            contrast_ylim=(-1, 1)
            );



@pytest.mark.mpl_image_compare(tolerance=10)
def test_29_paired_cumming_slopegraph_reflines_kwargs():

    return two_groups_paired.mean_diff.plot(float_contrast=False,
                                 color_col="Gender",
                                 slopegraph_kwargs=dict(linestyle='dotted'),
                                 reflines_kwargs=dict(linestyle='dashed',
                                                      linewidth=2),
                                 contrast_ylim=(-1, 1)
                                 );

@pytest.mark.mpl_image_compare(tolerance=10)
def test_30_sequential_cumming_slopegraph():
    return multi_groups_sequential.mean_diff.plot();

@pytest.mark.mpl_image_compare(tolerance=10)
def test_31_baseline_cumming_slopegraph():
    return multi_groups_baseline.mean_diff.plot();


@pytest.mark.mpl_image_compare(tolerance=10)
def test_99_style_sheets():
    # Perform this test last so we don't have to reset the plot style.
    plt.style.use("dark_background")

    return multi_2group.mean_diff.plot();