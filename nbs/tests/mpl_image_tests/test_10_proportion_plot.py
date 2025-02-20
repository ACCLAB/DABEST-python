import pytest
import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.ticker as Ticker
import matplotlib.pyplot as plt
from dabest._api import load


def create_demo_prop_dataset(seed=9999, N=40):
    np.random.seed(9999)  # Fix the seed so the results are replicable.
    # Create samples
    n = 1
    c1 = np.random.binomial(n, 0.2, size=N)
    c2 = np.random.binomial(n, 0.2, size=N)
    c3 = np.random.binomial(n, 0.8, size=N)

    t1 = np.random.binomial(n, 0.5, size=N)
    t2 = np.random.binomial(n, 0.2, size=N)
    t3 = np.random.binomial(n, 0.3, size=N)
    t4 = np.random.binomial(n, 0.4, size=N)
    t5 = np.random.binomial(n, 0.5, size=N)
    t6 = np.random.binomial(n, 0.6, size=N)
    t7 = np.zeros(N)
    t8 = np.ones(N)
    t9 = np.zeros(N)

    # Add a `gender` column for coloring the data.
    females = np.repeat("Female", N / 2).tolist()
    males = np.repeat("Male", N / 2).tolist()
    gender = females + males

    # Add an `id` column for paired data plotting.
    id_col = pd.Series(range(1, N + 1))

    # Combine samples and gender into a DataFrame.
    df = pd.DataFrame(
        {
            "Control 1": c1,
            "Test 1": t1,
            "Control 2": c2,
            "Test 2": t2,
            "Control 3": c3,
            "Test 3": t3,
            "Test 4": t4,
            "Test 5": t5,
            "Test 6": t6,
            "Test 7": t7,
            "Test 8": t8,
            "Test 9": t9,
            "Gender": gender,
            "ID": id_col,
        }
    )

    return df


df = create_demo_prop_dataset()

two_groups_unpaired = load(df, idx=("Control 1", "Test 1"), proportional=True)

multi_2group = load(
    df,
    idx=(
        (
            "Control 1",
            "Test 1",
        ),
        ("Control 2", "Test 2"),
    ),
    proportional=True,
)

shared_control = load(
    df,
    idx=("Control 1", "Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Test 6"),
    proportional=True,
)

multi_groups = load(
    df,
    idx=(
        (
            "Control 1",
            "Test 1",
        ),
        ("Control 2", "Test 2", "Test 3"),
        ("Control 3", "Test 4", "Test 5", "Test 6"),
    ),
    proportional=True,
)

two_groups_paired = load(
    df, idx=("Control 1", "Test 1"), paired="baseline", id_col="ID", proportional=True
)

multi_2group_paired = load(
    df,
    idx=(("Control 1", "Test 1"), ("Control 2", "Test 2")),
    paired="baseline",
    id_col="ID",
    proportional=True,
)

multi_groups_paired = load(
    df,
    idx=(
        (
            "Control 1",
            "Test 1",
        ),
        ("Control 2", "Test 2", "Test 3"),
        ("Control 3", "Test 4", "Test 5", "Test 6"),
    ),
    paired="baseline",
    id_col="ID",
    proportional=True,
)

two_groups_sequential = load(
    df, idx=("Control 1", "Test 1"), paired="sequential", id_col="ID", proportional=True
)

multi_2group_sequential = load(
    df,
    idx=(("Control 1", "Test 1"), ("Control 2", "Test 2")),
    paired="sequential",
    id_col="ID",
    proportional=True,
)

multi_groups_sequential = load(
    df,
    idx=(
        (
            "Control 1",
            "Test 1",
        ),
        ("Control 2", "Test 2", "Test 3"),
        ("Control 3", "Test 4", "Test 5", "Test 6"),
    ),
    paired="sequential",
    id_col="ID",
    proportional=True,
)
shared_control_paired = load(
    df,
    idx=("Control 1", "Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Test 6"),
    paired="sequential",
    id_col="ID",
    proportional=True,
)

zero_to_zero = load(
    df, idx=("Test 7", "Test 9"), proportional=True, paired="sequential", id_col="ID"
)
zero_to_one = load(
    df, idx=("Test 7", "Test 8"), proportional=True, paired="sequential", id_col="ID"
)
one_to_zero = load(
    df, idx=("Test 8", "Test 7"), proportional=True, paired="sequential", id_col="ID"
)

one_in_separate_control = load(
    df,
    idx=(
        (("Control 1", "Test 1"), ("Test 2", "Test 3"), ("Test 4", "Test 8", "Test 6"))
    ),
    proportional=True,
    paired="sequential",
    id_col="ID",
)


@pytest.mark.mpl_image_compare(tolerance=8)
def test_101_gardner_altman_unpaired_propdiff():
    plt.rcdefaults()
    return two_groups_unpaired.mean_diff.plot()


@pytest.mark.mpl_image_compare(tolerance=8)
def test_103_cummings_two_group_unpaired_propdiff():
    plt.rcdefaults()
    return two_groups_unpaired.mean_diff.plot(fig_size=(4, 6), float_contrast=False)


@pytest.mark.mpl_image_compare(tolerance=8)
def test_105_cummings_multi_group_unpaired_propdiff():
    plt.rcdefaults()
    return multi_2group.mean_diff.plot()


@pytest.mark.mpl_image_compare(tolerance=8)
def test_106_cummings_shared_control_propdiff():
    plt.rcdefaults()
    return shared_control.mean_diff.plot()


@pytest.mark.mpl_image_compare(tolerance=8)
def test_107_cummings_multi_groups_propdiff():
    plt.rcdefaults()
    return multi_groups.mean_diff.plot()


@pytest.mark.mpl_image_compare(tolerance=8)
def test_109_gardner_altman_ylabel():
    plt.rcdefaults()
    return two_groups_unpaired.mean_diff.plot(
        bar_label="This is my\nrawdata", contrast_label="The bootstrap\ndistribtions!"
    )


@pytest.mark.mpl_image_compare(tolerance=8)
def test_110_change_fig_size():
    plt.rcdefaults()
    return two_groups_unpaired.mean_diff.plot(fig_size=(6, 6), custom_palette="Dark2")


@pytest.mark.mpl_image_compare(tolerance=8)
def test_111_change_palette_b():
    plt.rcdefaults()
    return multi_2group.mean_diff.plot(custom_palette="Paired")


my_color_palette = {
    "Control 1": "blue",
    "Test 1": "purple",
    "Control 2": "#cb4b16",  # This is a hex string.
    "Test 2": (0.0, 0.7, 0.2),  # This is a RGB tuple.
}


@pytest.mark.mpl_image_compare(tolerance=8)
def test_112_change_palette_c():
    plt.rcdefaults()
    return multi_2group.mean_diff.plot(custom_palette=my_color_palette)


@pytest.mark.mpl_image_compare(tolerance=8)
def test_113_desat():
    plt.rcdefaults()
    return multi_2group.mean_diff.plot(
        custom_palette=my_color_palette, bar_desat=0.1, halfviolin_desat=0.25
    )


@pytest.mark.mpl_image_compare(tolerance=8)
def test_114_change_ylims():
    plt.rcdefaults()
    return multi_2group.mean_diff.plot(contrast_ylim=(-2, 2))


@pytest.mark.mpl_image_compare(tolerance=8)
def test_115_invert_ylim():
    plt.rcdefaults()
    return multi_2group.mean_diff.plot(
        contrast_ylim=(2, -2), contrast_label="More negative is better!"
    )


@pytest.mark.mpl_image_compare(tolerance=8)
def test_116_ticker_gardner_altman():
    plt.rcdefaults()
    fig = two_groups_unpaired.mean_diff.plot()

    rawswarm_axes = fig.axes[0]
    contrast_axes = fig.axes[1]

    rawswarm_axes.yaxis.set_major_locator(Ticker.MultipleLocator(1))
    rawswarm_axes.yaxis.set_minor_locator(Ticker.MultipleLocator(0.5))

    contrast_axes.yaxis.set_major_locator(Ticker.MultipleLocator(0.5))
    contrast_axes.yaxis.set_minor_locator(Ticker.MultipleLocator(0.25))
    return fig


@pytest.mark.mpl_image_compare(tolerance=8)
def test_117_err_color():
    plt.rcdefaults()
    return two_groups_unpaired.mean_diff.plot(err_color="purple")


@pytest.mark.mpl_image_compare(tolerance=8)
def test_118_cummings_two_group_unpaired_meandiff_bar_width():
    plt.rcdefaults()
    return two_groups_unpaired.mean_diff.plot(bar_width=0.4, float_contrast=False)


np.random.seed(9999)
Ns = [20, 10, 21, 20]
n = 1
c1 = pd.DataFrame({"Control": np.random.binomial(n, 0.2, size=Ns[0])})
t1 = pd.DataFrame({"Test 1": np.random.binomial(n, 0.5, size=Ns[1])})
t2 = pd.DataFrame({"Test 2": np.random.binomial(n, 0.4, size=Ns[2])})
t3 = pd.DataFrame({"Test 3": np.random.binomial(n, 0.7, size=Ns[3])})
wide_df = pd.concat([c1, t1, t2, t3], axis=1)


long_df = pd.melt(
    wide_df,
    value_vars=["Control", "Test 1", "Test 2", "Test 3"],
    value_name="value",
    var_name="group",
)
long_df["dummy"] = np.repeat(np.nan, len(long_df))


@pytest.mark.mpl_image_compare(tolerance=8)
def test_119_wide_df_nan():
    plt.rcdefaults()
    wide_df_dabest = load(
        wide_df, idx=("Control", "Test 1", "Test 2", "Test 3"), proportional=True
    )

    return wide_df_dabest.mean_diff.plot()


@pytest.mark.mpl_image_compare(tolerance=8)
def test_120_long_df_nan():
    plt.rcdefaults()
    long_df_dabest = load(
        long_df,
        x="group",
        y="value",
        idx=("Control", "Test 1", "Test 2", "Test 3"),
        proportional=True,
    )

    return long_df_dabest.mean_diff.plot()


@pytest.mark.mpl_image_compare(tolerance=8)
def test_121_cohens_h_gardner_altman():
    plt.rcdefaults()
    return two_groups_unpaired.cohens_h.plot()


@pytest.mark.mpl_image_compare(tolerance=8)
def test_122_cohens_h_cummings():
    plt.rcdefaults()
    return two_groups_unpaired.cohens_h.plot(float_contrast=False)


@pytest.mark.mpl_image_compare(tolerance=8)
def test_123_sankey_gardner_altman():
    plt.rcdefaults()
    return two_groups_paired.mean_diff.plot()


@pytest.mark.mpl_image_compare(tolerance=8)
def test_124_sankey_cummings():
    plt.rcdefaults()
    return two_groups_paired.mean_diff.plot(float_contrast=False)


@pytest.mark.mpl_image_compare(tolerance=8)
def test_125_sankey_2paired_groups():
    plt.rcdefaults()
    return multi_2group_paired.mean_diff.plot()


@pytest.mark.mpl_image_compare(tolerance=8)
def test_126_sankey_2sequential_groups():
    plt.rcdefaults()
    return multi_2group_sequential.mean_diff.plot()


@pytest.mark.mpl_image_compare(tolerance=8)
def test_127_sankey_multi_group_paired():
    plt.rcdefaults()
    return multi_groups_paired.mean_diff.plot()


@pytest.mark.mpl_image_compare(tolerance=8)
def test_128_sankey_transparency():
    plt.rcdefaults()
    return two_groups_paired.mean_diff.plot(sankey_kwargs={"alpha": 0.2})


@pytest.mark.mpl_image_compare(tolerance=8)
def test_129_zero_to_zero():
    plt.rcdefaults()
    return zero_to_zero.mean_diff.plot()


@pytest.mark.mpl_image_compare(tolerance=8)
def test_130_zero_to_one():
    plt.rcdefaults()
    return zero_to_one.mean_diff.plot()


@pytest.mark.mpl_image_compare(tolerance=8)
def test_131_one_to_zero():
    plt.rcdefaults()
    return one_to_zero.mean_diff.plot()


@pytest.mark.mpl_image_compare(tolerance=8)
def test_132_shared_control_sankey_off():
    plt.rcdefaults()
    return shared_control_paired.mean_diff.plot(sankey_kwargs={"sankey": False})


@pytest.mark.mpl_image_compare(tolerance=8)
def test_133_shared_control_flow_off():
    plt.rcdefaults()
    return shared_control_paired.mean_diff.plot(sankey_kwargs={"flow": False})


@pytest.mark.mpl_image_compare(tolerance=8)
def test_134_separate_control_sankey_off():
    plt.rcdefaults()
    return multi_groups_sequential.mean_diff.plot(sankey_kwargs={"sankey": False})


@pytest.mark.mpl_image_compare(tolerance=8)
def test_135_separate_control_flow_off():
    plt.rcdefaults()
    return multi_groups_sequential.mean_diff.plot(sankey_kwargs={"flow": False})

# Show sample counts
@pytest.mark.mpl_image_compare(tolerance=8)
def test_137_multi_2group_show_sample_counts():
    plt.rcdefaults()
    return multi_2group.mean_diff.plot(prop_sample_counts=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_138_multi_groups_paired_show_sample_counts():
    plt.rcdefaults()
    return multi_groups_paired.mean_diff.plot(prop_sample_counts=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_139_multi_2group_show_sample_counts_and_kwargs():
    plt.rcdefaults()
    return multi_2group.mean_diff.plot(prop_sample_counts=True, prop_sample_counts_kwargs={
        "color": "red", "fontsize": 12, "fontweight": "bold"})

@pytest.mark.mpl_image_compare(tolerance=8)
def test_140_multi_groups_paired_show_sample_counts_with_sankey_off():
    plt.rcdefaults()
    return multi_groups_paired.mean_diff.plot(prop_sample_counts=True, sankey_kwargs={"sankey": False})


@pytest.mark.mpl_image_compare(tolerance=8)
def test_141_sankey_change_palette_a():
    plt.rcdefaults()
    return multi_groups_paired.mean_diff.plot(custom_palette="Dark2")

@pytest.mark.mpl_image_compare(tolerance=8)
def test_142_sankey_change_palette_b():
    plt.rcdefaults()
    return multi_groups_paired.mean_diff.plot(custom_palette={1: 'red', 0: 'blue'})

@pytest.mark.mpl_image_compare(tolerance=8)
def test_143_sankey_change_palette_c():
    plt.rcdefaults()
    return multi_groups_paired.mean_diff.plot(custom_palette=['red', 'blue'])

@pytest.mark.mpl_image_compare(tolerance=8)
def test_136_style_sheets():
    # Perform this test last so we don't have to reset the plot style.
    plt.rcdefaults()
    plt.style.use("dark_background")
    return multi_2group.mean_diff.plot(face_color="black")
