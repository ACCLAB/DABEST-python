import pytest
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.ticker as Ticker
import matplotlib.pyplot as plt
from .._api import load
from .utils import create_demo_prop_dataset

df = create_demo_prop_dataset()

two_groups_unpaired = load(df, idx=("Control 1", "Test 1"), proportional=True)

multi_2group = load(df, idx=(("Control 1", "Test 1",),
                             ("Control 2", "Test 2")),
                    proportional=True)

shared_control = load(df, idx=("Control 1", "Test 1",
                                "Test 2", "Test 3",
                                "Test 4", "Test 5", "Test 6"),
                    proportional=True)

multi_groups = load(df, idx=(("Control 1", "Test 1",),
                             ("Control 2", "Test 2","Test 3"),
                             ("Control 3", "Test 4","Test 5", "Test 6")
                             ),proportional=True)

two_groups_paired   = load(df, idx=("Control 1", "Test 1"),
                           paired="baseline", id_col="ID",proportional=True)

multi_2group_paired = load(df, idx=(("Control 1", "Test 1"),
                                 ("Control 2", "Test 2")),
                            paired="baseline", id_col="ID", proportional=True)

multi_groups_paired = load(df, idx=(("Control 1", "Test 1",),
                             ("Control 2", "Test 2","Test 3"),
                             ("Control 3", "Test 4","Test 5", "Test 6")
                             ),paired="baseline", id_col="ID", proportional=True)

two_groups_sequential   = load(df, idx=("Control 1", "Test 1"),
                           paired="sequential", id_col="ID",proportional=True)

multi_2group_sequential = load(df, idx=(("Control 1", "Test 1"),
                                 ("Control 2", "Test 2")),
                            paired="sequential", id_col="ID", proportional=True)

multi_groups_sequential = load(df, idx=(("Control 1", "Test 1",),
                             ("Control 2", "Test 2","Test 3"),
                             ("Control 3", "Test 4","Test 5", "Test 6")
                             ),paired="sequential", id_col="ID", proportional=True)


@pytest.mark.mpl_image_compare
def test_101_gardner_altman_unpaired_propdiff():
    return two_groups_unpaired.mean_diff.plot();

@pytest.mark.mpl_image_compare
def test_102_gardner_altman_paired_propdiff():
    return two_groups_paired.mean_diff.plot();

@pytest.mark.mpl_image_compare
def test_103_cummings_two_group_unpaired_propdiff():
    return two_groups_unpaired.mean_diff.plot(fig_size=(4, 6),
                                              float_contrast=False);

@pytest.mark.mpl_image_compare
def test_104_cummings_two_group_paired_propdiff():
    return two_groups_paired.mean_diff.plot(fig_size=(6, 6),
                                            float_contrast=False);

@pytest.mark.mpl_image_compare
def test_105_cummings_multi_group_unpaired__propdiff():
    return multi_2group.mean_diff.plot();

@pytest.mark.mpl_image_compare
def test_106_cummings_shared_control_propdiff():
    return shared_control.mean_diff.plot();

@pytest.mark.mpl_image_compare
def test_107_cummings_multi_groups_propdiff():
    return multi_groups.mean_diff.plot();

@pytest.mark.mpl_image_compare
def test_108_inset_plots_propdiff():

    # Load the titanic dataset. Requires internet access.
    titanic = pd.read_csv("https://github.com/mwaskom/seaborn-data/raw/master/titanic.csv")
    titanic['alone'][titanic['alone'] == True] = 1
    titanic['alone'][titanic['alone'] == False] = 0
    titanic["alone"] = titanic["alone"].astype("int")
    titanic_melt = pd.melt(titanic.reset_index(),
                           id_vars=["sex", "index"], var_name="metric")

    titanic_dabest1 = load(data=titanic, x="sex", y="survived",
                              idx=("female","male"), proportional=True)
    titanic_dabest2 = load(data=titanic, x="sex", y="alone",
                           idx=("female", "male"), proportional=True)
    # Create Figure.
    fig, ax = plt.subplots(nrows=2, ncols=2,
                           figsize=(15, 15),
                           gridspec_kw={"wspace": 0.5})
    titanic_dabest1.mean_diff.plot(ax=ax.flat[0]);
    titanic_dabest2.mean_diff.plot(ax=ax.flat[1]);
    titanic_dabest1.mean_diff.plot(ax=ax.flat[2], float_contrast=False);
    titanic_dabest2.mean_diff.plot(ax=ax.flat[3], float_contrast=False);
    return fig

@pytest.mark.mpl_image_compare
def test_109_gardner_altman_ylabel():
    return two_groups_unpaired.mean_diff.plot(bar_label="This is my\nrawdata",
                                   contrast_label="The bootstrap\ndistribtions!");

@pytest.mark.mpl_image_compare
def test_110_change_fig_size():
    return two_groups_unpaired.mean_diff.plot(fig_size=(6, 6),
                                            custom_palette="Dark2");

@pytest.mark.mpl_image_compare
def test_111_change_palette_b():
    return multi_2group.mean_diff.plot(custom_palette="Paired");


my_color_palette = {"Control 1" : "blue",
                "Test 1"    : "purple",
                "Control 2" : "#cb4b16",     # This is a hex string.
                "Test 2"    : (0., 0.7, 0.2) # This is a RGB tuple.
                }

@pytest.mark.mpl_image_compare
def test_112_change_palette_c():
    return multi_2group.mean_diff.plot(custom_palette=my_color_palette);

@pytest.mark.mpl_image_compare
def test_113_desat():
    return multi_2group.mean_diff.plot(custom_palette=my_color_palette,
                            bar_desat=0.1,
                            halfviolin_desat=0.25);

@pytest.mark.mpl_image_compare
def test_114_change_ylims():
    return multi_2group.mean_diff.plot(contrast_ylim=(-2, 2));

@pytest.mark.mpl_image_compare
def test_115_invert_ylim():
    return multi_2group.mean_diff.plot(contrast_ylim=(2, -2),
                                       contrast_label="More negative is better!");

@pytest.mark.mpl_image_compare
def test_116_ticker_gardner_altman():

    fig = two_groups_unpaired.mean_diff.plot()

    rawswarm_axes = fig.axes[0]
    contrast_axes = fig.axes[1]

    rawswarm_axes.yaxis.set_major_locator(Ticker.MultipleLocator(1))
    rawswarm_axes.yaxis.set_minor_locator(Ticker.MultipleLocator(0.5))

    contrast_axes.yaxis.set_major_locator(Ticker.MultipleLocator(0.5))
    contrast_axes.yaxis.set_minor_locator(Ticker.MultipleLocator(0.25))
    return fig

@pytest.mark.mpl_image_compare
def test_117_err_color():
    return two_groups_unpaired.mean_diff.plot(err_color="purple");

@pytest.mark.mpl_image_compare
def test_118_cummings_two_group_unpaired_meandiff_bar_width():
    return two_groups_unpaired.mean_diff.plot(bar_width=0.4,float_contrast=False);

np.random.seed(9999)
Ns = [20, 10, 21, 20]
n=1
c1 = pd.DataFrame({'Control':np.random.binomial(n, 0.2, size=Ns[0])})
t1 = pd.DataFrame({'Test 1': np.random.binomial(n, 0.5, size=Ns[1])})
t2 = pd.DataFrame({'Test 2': np.random.binomial(n, 0.4, size=Ns[2])})
t3 = pd.DataFrame({'Test 3': np.random.binomial(n, 0.7, size=Ns[3])})
wide_df = pd.concat([c1, t1, t2, t3],axis=1)


long_df = pd.melt(wide_df,
              value_vars=["Control", "Test 1", "Test 2", "Test 3"],
                value_name="value",
                var_name="group")
long_df['dummy'] = np.repeat(np.nan, len(long_df))

@pytest.mark.mpl_image_compare
def test_119_wide_df_nan():

    wide_df_dabest = load(wide_df,
                          idx=("Control", "Test 1", "Test 2", "Test 3"),
                          proportional=True
                          )

    return wide_df_dabest.mean_diff.plot();

@pytest.mark.mpl_image_compare
def test_120_long_df_nan():

    long_df_dabest = load(long_df, x="group", y="value",
                          idx=("Control", "Test 1", "Test 2", "Test 3"),
                          proportional=True
                          )

    return long_df_dabest.mean_diff.plot();

@pytest.mark.mpl_image_compare
def test_121_cohens_h_gardner_altman():
    return two_groups_unpaired.cohens_h.plot();

@pytest.mark.mpl_image_compare
def test_122_cohens_h_cummings():
    return two_groups_unpaired.cohens_h.plot(float_contrast=False);

@pytest.mark.mpl_image_compare
def test_123_style_sheets():
    # Perform this test last so we don't have to reset the plot style.
    plt.style.use("dark_background")
    return multi_2group.mean_diff.plot(face_color="black");


