import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm


import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as Ticker
import seaborn as sns

from dabest._api import load

def create_demo_dataset(seed=9999, N=20):
    import numpy as np
    import pandas as pd
    from scipy.stats import norm  # Used in generation of populations.

    np.random.seed(9999)  # Fix the seed so the results are replicable.
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
            "Gender": gender,
            "ID": id_col,
        }
    )

    return df

def create_demo_dataset_delta(seed=9999, N=20):

    import numpy as np
    import pandas as pd
    from scipy.stats import norm  # Used in generation of populations.

    np.random.seed(seed)  # Fix the seed so the results are replicable.
    # pop_size = 10000 # Size of each population.

    from scipy.stats import norm  # Used in generation of populations.

    # Create samples
    y = norm.rvs(loc=3, scale=0.4, size=N * 4)
    y[N : 2 * N] = y[N : 2 * N] + 1
    y[2 * N : 3 * N] = y[2 * N : 3 * N] - 0.5

    # Add drug column
    t1 = np.repeat("Placebo", N * 2).tolist()
    t2 = np.repeat("Drug", N * 2).tolist()
    treatment = t1 + t2

    # Add a `rep` column as the first variable for the 2 replicates of experiments done
    rep = []
    for i in range(N * 2):
        rep.append("Rep1")
        rep.append("Rep2")

    # Add a `genotype` column as the second variable
    wt = np.repeat("W", N).tolist()
    mt = np.repeat("M", N).tolist()
    wt2 = np.repeat("W", N).tolist()
    mt2 = np.repeat("M", N).tolist()

    genotype = wt + mt + wt2 + mt2

    # Add an `id` column for paired data plotting.
    id = list(range(0, N * 2))
    id_col = id + id

    # Combine all columns into a DataFrame.
    df = pd.DataFrame(
        {"ID": id_col, "Rep": rep, "Genotype": genotype, "Treatment": treatment, "Y": y}
    )
    return df

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

df = create_demo_dataset()
df_delta = create_demo_dataset_delta()
df_prop = create_demo_prop_dataset()

# Two group
two_groups_unpaired = load(df, idx=("Control 1", "Test 1"))
two_groups_paired = load(df, idx=("Control 1", "Test 1"), paired='baseline', id_col='ID')

# Multi two group
multi_2group_unpaired = load(df, idx=(("Control 1","Test 1",),("Control 2", "Test 2"),),)
multi_2group_paired = load(df, idx=(("Control 1","Test 1",),("Control 2", "Test 2"),), paired='baseline', id_col='ID')

# Multi-group
shared_control = load(df, idx=("Control 1", "Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Test 6"))
repeated_measures = load(df, idx=("Control 1", "Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Test 6"), paired='baseline', id_col='ID')


# Mixed multi group and two group
multi_groups_unpaired = load(df,idx=(("Control 1","Test 1",),("Control 2", "Test 2", "Test 3"),("Control 3", "Test 4", "Test 5", "Test 6"),),)
multi_groups_paired_baseline = load(df,idx=(("Control 1","Test 1",),("Control 2", "Test 2", "Test 3"),("Control 3", "Test 4", "Test 5", "Test 6"),),
                                    paired='baseline', id_col='ID')
multi_groups_paired_sequential = load(df,idx=(("Control 1","Test 1",),("Control 2", "Test 2", "Test 3"),("Control 3", "Test 4", "Test 5", "Test 6"),),
                                                paired='sequential', id_col='ID')

# Proportion plots
df_prop = create_demo_prop_dataset()

two_groups_unpaired_prop = load(df_prop, idx=("Control 1", "Test 1"), proportional=True)

two_groups_paired_baseline_prop = load(df_prop, idx=("Control 1", "Test 1"), paired="baseline", id_col="ID", proportional=True)

two_groups_paired_sequential_prop = load(df_prop, idx=("Control 1", "Test 1"), paired="sequential", id_col="ID", proportional=True)

multi_2group_unpaired_prop = load(df_prop, idx=(("Control 1","Test 1",),("Control 2", "Test 2"),), proportional=True,)

multi_2group_paired_baseline_prop = load(df_prop, idx=(("Control 1", "Test 1"), ("Control 2", "Test 2")), paired="baseline", id_col="ID", proportional=True,)

multi_2group_paired_sequential_prop = load(df_prop, idx=(("Control 1", "Test 1"), ("Control 2", "Test 2")), paired="sequential", id_col="ID", proportional=True,)

shared_control_prop = load(df_prop, idx=("Control 1", "Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Test 6"), proportional=True,)

repeated_measures_baseline_prop = load(df_prop, idx=("Control 1", "Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Test 6"), 
                              paired="baseline", id_col="ID", proportional=True,)

repeated_measures_sequential_prop = load(df_prop, idx=("Control 1", "Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Test 6"), 
                              paired="sequential", id_col="ID", proportional=True,)

multi_groups_unpaired_prop = load(df_prop, idx=(("Control 1","Test 1",),("Control 2", "Test 2", "Test 3"),("Control 3", "Test 4", "Test 5", "Test 6"),),
                                    proportional=True,)

multi_groups_paired_baseline_prop = load(df_prop, idx=(("Control 1","Test 1",),("Control 2", "Test 2", "Test 3"),("Control 3", "Test 4", "Test 5", "Test 6"),),
                            paired="baseline", id_col="ID", proportional=True,)

multi_groups_paired_sequential_prop = load(df_prop, idx=(("Control 1","Test 1",),("Control 2", "Test 2", "Test 3"),
                                                        ("Control 3", "Test 4", "Test 5", "Test 6"),), paired="sequential",
                                                        id_col="ID", proportional=True,)

zero_to_zero_prop = load(df_prop, idx=("Test 7", "Test 9"), proportional=True, paired="sequential", id_col="ID")
zero_to_one_prop = load(df_prop, idx=("Test 7", "Test 8"), proportional=True, paired="sequential", id_col="ID")
one_to_zero_prop = load(df_prop, idx=("Test 8", "Test 7"), proportional=True, paired="sequential", id_col="ID")
one_in_separate_control_prop = load(df_prop,idx=((("Control 1", "Test 1"), ("Test 2", "Test 3"), ("Test 4", "Test 8", "Test 6"))),
                                proportional=True, paired="sequential", id_col="ID",)



# delta-delta
delta_delta_unpaired = load(data = df_delta, x = ["Genotype", "Genotype"], y = "Y", delta2 = True, experiment = "Treatment")
delta_delta_unpaired_specified = load(data = df_delta, x = ["Genotype", "Genotype"], y = "Y", 
                          delta2 = True, experiment = "Treatment",
                                        experiment_label = ["Drug", "Placebo"],
                                        x1_level = ["M", "W"])
delta_delta_paired_baseline = load(data = df_delta, x = ["Treatment", "Rep"], y = "Y", delta2 = True, experiment = "Genotype", paired="baseline", id_col="ID")
delta_delta_paired_sequential = load(data = df_delta, x = ["Treatment", "Rep"], y = "Y", delta2 = True, experiment = "Genotype", paired="sequential", id_col="ID")

# mini_meta
mini_meta_unpaired = load(df, idx=(("Control 1", "Test 1"), ("Control 2", "Test 2"), ("Control 3", "Test 3")), mini_meta=True)
mini_meta_paired_baseline = load(df, id_col = "ID", idx=(("Control 1", "Test 1"), ("Control 2", "Test 2"), ("Control 3", "Test 3")), 
                                 paired = "baseline", mini_meta=True)
mini_meta_paired_sequential = load(df, id_col = "ID", idx=(("Control 1", "Test 1"), ("Control 2", "Test 2"), ("Control 3", "Test 3")), 
                                   paired = "sequential", mini_meta=True)


# Tests
# Two Group

@pytest.mark.mpl_image_compare(tolerance=8)
def test_300_2group_unpaired_meandiff():
    return two_groups_unpaired.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_301_2group_unpaired_mediandiff():
    return two_groups_unpaired.median_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_302_2group_unpaired_hedges_g():
    return two_groups_unpaired.hedges_g.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_303_2group_paired_meandiff():
    return two_groups_paired.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_304_2group_paired_hedges_g():
    return two_groups_paired.hedges_g.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_305_2group_cummings_unpaired_meandiff():
    return two_groups_unpaired.mean_diff.plot(horizontal=True, fig_size=(6, 4), float_contrast=False)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_306_2group_cummings_paired_meandiff():
    return two_groups_paired.mean_diff.plot(horizontal=True, float_contrast=False)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_307_multi2group_unpaired():
    return multi_2group_unpaired.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_308_multi2group_paired():
    return multi_2group_paired.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_309_sharedcontrol():
    return shared_control.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_310_repeatedmeasure():
    return repeated_measures.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_311_multigroups_unpaired():
    return multi_groups_unpaired.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_312_multigroups_paired_baseline():
    return multi_groups_paired_baseline.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_313_multigroups_paired_sequential():
    return multi_groups_paired_sequential.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_314_2group_unpaired_ylabel():
    return two_groups_unpaired.mean_diff.plot(horizontal=True,
        swarm_label="This is my\nrawdata", contrast_label="The bootstrap\ndistribtions!"
    )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_315_multi2group_color():
    return multi_2group_unpaired.mean_diff.plot(horizontal=True, color_col="Gender")

@pytest.mark.mpl_image_compare(tolerance=8)
def test_316_2group_paired_color():
    return two_groups_paired.mean_diff.plot(horizontal=True, color_col="Gender")

@pytest.mark.mpl_image_compare(tolerance=8)
def test_317_multi2group_unpaired_change_palette_a():
    return multi_2group_unpaired.mean_diff.plot(horizontal=True, color_col="Gender", custom_palette="Dark2")

@pytest.mark.mpl_image_compare(tolerance=8)
def test_318_multi2group_unpaired_change_palette_b():
    return multi_2group_unpaired.mean_diff.plot(horizontal=True, custom_palette="Paired")

my_color_palette = {
    "Control 1": "blue",
    "Test 1": "purple",
    "Control 2": "#cb4b16",  # This is a hex string.
    "Test 2": (0.0, 0.7, 0.2),  # This is a RGB tuple.
}

@pytest.mark.mpl_image_compare(tolerance=8)
def test_319_multi2group_unpaired_change_palette_c():
    return multi_2group_unpaired.mean_diff.plot(horizontal=True, custom_palette=my_color_palette)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_320_multi2group_unpaired_desat():
    return multi_2group_unpaired.mean_diff.plot(horizontal=True, 
        custom_palette=my_color_palette, swarm_desat=0.75, halfviolin_desat=0.25
    )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_321_multi2group_unpaired_dot_sizes():
    return multi_2group_unpaired.mean_diff.plot(horizontal=True, raw_marker_size=3, es_marker_size=12)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_322_multi2group_unpaired_change_ylims():
    return multi_2group_unpaired.mean_diff.plot(horizontal=True, swarm_ylim=(0, 5), contrast_ylim=(-2, 2))

@pytest.mark.mpl_image_compare(tolerance=8)
def test_323_2group_unpaired_ticker():
    f = two_groups_unpaired.mean_diff.plot(horizontal=True)

    rawswarm_axes = f.axes[0]
    contrast_axes = f.axes[1]

    rawswarm_axes.xaxis.set_major_locator(Ticker.MultipleLocator(1))
    rawswarm_axes.xaxis.set_minor_locator(Ticker.MultipleLocator(0.5))

    contrast_axes.xaxis.set_major_locator(Ticker.MultipleLocator(0.5))
    contrast_axes.xaxis.set_minor_locator(Ticker.MultipleLocator(0.25))

    return f

@pytest.mark.mpl_image_compare(tolerance=8)
def test_324_multi2group_unpaired_ticker():
    f = multi_2group_unpaired.mean_diff.plot(horizontal=True, swarm_ylim=(0, 6), contrast_ylim=(-3, 1))

    rawswarm_axes = f.axes[0]
    contrast_axes = f.axes[1]

    rawswarm_axes.xaxis.set_major_locator(Ticker.MultipleLocator(2))
    rawswarm_axes.xaxis.set_minor_locator(Ticker.MultipleLocator(1))

    contrast_axes.xaxis.set_major_locator(Ticker.MultipleLocator(0.5))
    contrast_axes.xaxis.set_minor_locator(Ticker.MultipleLocator(0.25))

    return f

np.random.seed(9999)
Ns = [20, 10, 21, 20]
c1 = pd.DataFrame({"Control": norm.rvs(loc=3, scale=0.4, size=Ns[0])})
t1 = pd.DataFrame({"Test 1": norm.rvs(loc=3.5, scale=0.5, size=Ns[1])})
t2 = pd.DataFrame({"Test 2": norm.rvs(loc=2.5, scale=0.6, size=Ns[2])})
t3 = pd.DataFrame({"Test 3": norm.rvs(loc=3, scale=0.75, size=Ns[3])})
wide_df = pd.concat([c1, t1, t2, t3], axis=1)

long_df = pd.melt(
    wide_df,
    value_vars=["Control", "Test 1", "Test 2", "Test 3"],
    value_name="value",
    var_name="group",
)
long_df["dummy"] = np.repeat(np.nan, len(long_df))

@pytest.mark.mpl_image_compare(tolerance=8)
def test_325_wide_df_nan():
    wide_df_dabest = load(wide_df, idx=("Control", "Test 1", "Test 2", "Test 3"))

    return wide_df_dabest.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_326_long_df_nan():
    long_df_dabest = load(
        long_df, x="group", y="value", idx=("Control", "Test 1", "Test 2", "Test 3")
    )

    return long_df_dabest.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_327_2group_paired_slopegraph_kwargs():
    return two_groups_paired.mean_diff.plot(horizontal=True, slopegraph_kwargs=dict(linestyle="dotted"))

@pytest.mark.mpl_image_compare(tolerance=8)
def test_328_2group_unpaired_reflines_kwargs():
    return two_groups_unpaired.mean_diff.plot(horizontal=True, reflines_kwargs=dict(linestyle="dotted"))

@pytest.mark.mpl_image_compare(tolerance=8)
def test_329_2group_unpaired_cumming_reflines_kwargs():
    return two_groups_unpaired.mean_diff.plot(horizontal=True,
        fig_size=(12, 10),
        float_contrast=False,
        reflines_kwargs=dict(linestyle="dotted", linewidth=2),
        contrast_ylim=(-1, 1),
    )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_330_2group_paired_cumming_slopegraph_reflines_kwargs():
    return two_groups_paired.mean_diff.plot(horizontal=True,
        float_contrast=False,
        color_col="Gender",
        slopegraph_kwargs=dict(linestyle="dotted"),
        reflines_kwargs=dict(linestyle="dashed", linewidth=2),
        contrast_ylim=(-1, 1),
    )



# Proportion plots

@pytest.mark.mpl_image_compare(tolerance=8)
def test_331_2group_unpaired_propdiff():
    return two_groups_unpaired_prop.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_332_2group_unpaired_cummings_propdiff():
    return two_groups_unpaired_prop.mean_diff.plot(horizontal=True, fig_size=(6,4), float_contrast=False)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_333_multi2group_unpaired_propdiff():
    return multi_2group_unpaired_prop.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_334_shared_control_propdiff():
    return shared_control_prop.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_335_repeated_measures_baseline_propdiff():
    return repeated_measures_baseline_prop.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_336_repeated_measures_sequential_propdiff():
    return repeated_measures_sequential_prop.mean_diff.plot(horizontal=True)


@pytest.mark.mpl_image_compare(tolerance=8)
def test_337_multi_groups_unpaired_propdiff():
    return multi_groups_unpaired_prop.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_338_multi_groups_paired_baseline_propdiff():
    return multi_groups_paired_baseline_prop.mean_diff.plot(horizontal=True)


@pytest.mark.mpl_image_compare(tolerance=8)
def test_339_multi_groups_paired_sequential_propdiff():
    return multi_groups_paired_sequential_prop.mean_diff.plot(horizontal=True)


@pytest.mark.mpl_image_compare(tolerance=8)
def test_340_2group_unpaired_prop_change_fig_size_and_palette_a():
    return two_groups_unpaired_prop.mean_diff.plot(horizontal=True, fig_size=(6, 6), custom_palette="Dark2")


@pytest.mark.mpl_image_compare(tolerance=8)
def test_341_multi2group_unpaired_prop_change_palette_b():
    return multi_2group_unpaired_prop.mean_diff.plot(horizontal=True, custom_palette="Paired")

my_color_palette = {
    "Control 1": "blue",
    "Test 1": "purple",
    "Control 2": "#cb4b16",  # This is a hex string.
    "Test 2": (0.0, 0.7, 0.2),  # This is a RGB tuple.
}

@pytest.mark.mpl_image_compare(tolerance=8)
def test_342_multi2group_unpaired_prop_change_palette_c():
    return multi_2group_unpaired_prop.mean_diff.plot(horizontal=True, custom_palette=my_color_palette)


@pytest.mark.mpl_image_compare(tolerance=8)
def test_343_multi2group_unpaired_prop_desat():
    return multi_2group_unpaired_prop.mean_diff.plot(horizontal=True,
        custom_palette=my_color_palette, bar_desat=0.1, halfviolin_desat=0.25
    )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_344_2group_unpaired_prop_err_color():
    return two_groups_unpaired_prop.mean_diff.plot(horizontal=True, err_color="purple")

@pytest.mark.mpl_image_compare(tolerance=8)
def test_345_2group_unpaired_cummings_meandiff_bar_width():
    return two_groups_unpaired_prop.mean_diff.plot(horizontal=True, bar_width=0.4, float_contrast=False)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_346_2group_unpaired_prop_cohens_h():
    return two_groups_unpaired_prop.cohens_h.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_347_2group_unpaired_prop_cummings_cohens_h():
    return two_groups_unpaired_prop.cohens_h.plot(horizontal=True, float_contrast=False)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_348_2group_sankey():
    return two_groups_paired_baseline_prop.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_349_2group_sankey_cummings():
    return two_groups_paired_baseline_prop.mean_diff.plot(horizontal=True, float_contrast=False)


@pytest.mark.mpl_image_compare(tolerance=8)
def test_350_multi2group_sankey_baseline():
    return multi_2group_paired_baseline_prop.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_351_multi2group_sankey_sequential():
    return multi_2group_paired_sequential_prop.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_352_multigroups_sankey_baseline():
    return multi_groups_paired_baseline_prop.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_353_multigroups_sankey_sequential():
    return multi_groups_paired_sequential_prop.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_354_2group_sankey_transparency():
    return two_groups_paired_baseline_prop.mean_diff.plot(horizontal=True, sankey_kwargs={"alpha": 0.2})

@pytest.mark.mpl_image_compare(tolerance=8)
def test_355_zero_to_zero():
    return zero_to_zero_prop.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_356_zero_to_one_prop():
    return zero_to_one_prop.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_357_one_to_zero():
    return one_to_zero_prop.mean_diff.plot(horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_358_repeated_measures_baseline_sankey_off():
    return repeated_measures_baseline_prop.mean_diff.plot(horizontal=True, sankey_kwargs={"sankey": False})

@pytest.mark.mpl_image_compare(tolerance=8)
def test_359_repeated_measures_baseline_flow_off():
    return repeated_measures_baseline_prop.mean_diff.plot(horizontal=True, sankey_kwargs={"flow": False})

@pytest.mark.mpl_image_compare(tolerance=8)
def test_360_multigroups_paired_sequential_sankey_off():
    return multi_groups_paired_sequential_prop.mean_diff.plot(horizontal=True, sankey_kwargs={"sankey": False})

@pytest.mark.mpl_image_compare(tolerance=8)
def test_361_multigroups_paired_sequential_flow_off():
    return multi_groups_paired_sequential_prop.mean_diff.plot(horizontal=True, sankey_kwargs={"flow": False})



# delta-delta
@pytest.mark.mpl_image_compare(tolerance=8)
def test_362_cummings_unpaired_delta_delta_meandiff():
    return delta_delta_unpaired.mean_diff.plot(horizontal=True);


@pytest.mark.mpl_image_compare(tolerance=8)
def test_363_cummings_sequential_delta_delta_meandiff():
    return delta_delta_paired_sequential.mean_diff.plot(horizontal=True);


@pytest.mark.mpl_image_compare(tolerance=8)
def test_364_cummings_baseline_delta_delta_meandiff():
    return delta_delta_paired_baseline.mean_diff.plot(horizontal=True);


@pytest.mark.mpl_image_compare(tolerance=8)
def test_365_delta_plot_ylabel():
    return delta_delta_paired_baseline.mean_diff.plot(horizontal=True,
                                                    swarm_label="This is my\nrawdata",
                                                    contrast_label="The bootstrap\ndistribtions!", 
                                                    delta2_label="This is delta!");

@pytest.mark.mpl_image_compare(tolerance=8)
def test_366_delta_plot_change_palette_a():
    return delta_delta_paired_sequential.mean_diff.plot(horizontal=True, custom_palette="Dark2");

@pytest.mark.mpl_image_compare(tolerance=8)
def test_367_delta_specified():
    return delta_delta_unpaired_specified.mean_diff.plot(horizontal=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_368_delta_change_ylims():
    return delta_delta_paired_sequential.mean_diff.plot(horizontal=True, swarm_ylim=(0, 9),
                                                        contrast_ylim=(-2, 2),
                                                        fig_size=(15,6));

@pytest.mark.mpl_image_compare(tolerance=8)
def test_369_delta_invert_ylim():
    return delta_delta_paired_sequential.mean_diff.plot(horizontal=True, 
                                                        contrast_ylim=(2, -2),
                                                        contrast_label="More negative is better!");

@pytest.mark.mpl_image_compare(tolerance=8)
def test_370_delta_median_diff():
    return delta_delta_paired_sequential.median_diff.plot(horizontal=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_371_delta_cohens_d():
    return delta_delta_unpaired.cohens_d.plot(horizontal=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_372_delta_show_delta2():
    return delta_delta_unpaired.mean_diff.plot(horizontal=True, show_delta2=False);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_373_delta_axes_invert_ylim():
    return delta_delta_unpaired.mean_diff.plot(horizontal=True, delta2_ylim=(2, -2),
                                   delta2_label="More negative is better!");
                
@pytest.mark.mpl_image_compare(tolerance=8)
def test_374_delta_axes_invert_ylim_not_showing_delta2():
    return delta_delta_unpaired.mean_diff.plot(horizontal=True, delta2_ylim=(2, -2),
                                                delta2_label="More negative is better!",
                                                show_delta2=False);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_375_unpaired_delta_g():
    return delta_delta_unpaired.delta_g.plot(horizontal=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_376_sequential_delta_g():
    return delta_delta_paired_sequential.mean_diff.plot(horizontal=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_377_baseline_delta_g():
    return delta_delta_paired_baseline.mean_diff.plot(horizontal=True);


# mini_meta
@pytest.mark.mpl_image_compare(tolerance=8)
def test_378_cummings_unpaired_mini_meta_meandiff():
    return mini_meta_unpaired.mean_diff.plot(horizontal=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_379_cummings_sequential_mini_meta_meandiff():
    return mini_meta_paired_sequential.mean_diff.plot(horizontal=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_380_cummings_baseline_mini_meta_meandiff():
    return mini_meta_paired_baseline.mean_diff.plot(horizontal=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_381_mini_meta_plot_ylabel():
    return mini_meta_paired_baseline.mean_diff.plot(horizontal=True, swarm_label="This is my\nrawdata",
                                                    contrast_label="The bootstrap\ndistribtions!");

@pytest.mark.mpl_image_compare(tolerance=8)
def test_382_mini_meta_plot_change_palette_a():
    return mini_meta_unpaired.mean_diff.plot(horizontal=True, custom_palette="Dark2");

@pytest.mark.mpl_image_compare(tolerance=8)
def test_383_mini_meta_dot_sizes():
    return mini_meta_paired_sequential.mean_diff.plot(horizontal=True, show_pairs=False,raw_marker_size=3,
                                                    es_marker_size=12);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_384_mini_meta_change_ylims():
    return mini_meta_paired_sequential.mean_diff.plot(horizontal=True, swarm_ylim=(0, 5),
                                                        contrast_ylim=(-2, 2),
                                                        fig_size=(15,6));

@pytest.mark.mpl_image_compare(tolerance=8)
def test_385_mini_meta_invert_ylim():
    return mini_meta_paired_sequential.mean_diff.plot(horizontal=True, contrast_ylim=(2, -2),
                                                        contrast_label="More negative is better!");

@pytest.mark.mpl_image_compare(tolerance=8)
def test_386_mini_meta_median_diff():
    return mini_meta_paired_sequential.median_diff.plot(horizontal=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_387_mini_meta_cohens_d():
    return mini_meta_unpaired.cohens_d.plot(horizontal=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_388_mini_meta_not_show():
    return mini_meta_unpaired.mean_diff.plot(horizontal=True, show_mini_meta=False);


# Aesthetic kwargs
# Swarm_Side
@pytest.mark.mpl_image_compare(tolerance=8)
def test_389_Swarm_Side_Center():
    return two_groups_unpaired.mean_diff.plot(horizontal=True, swarm_side='center');

@pytest.mark.mpl_image_compare(tolerance=8)
def test_390_Swarm_Side_Right():
    return two_groups_unpaired.mean_diff.plot(horizontal=True, swarm_side='right');

@pytest.mark.mpl_image_compare(tolerance=8)
def test_391_Swarm_Side_Left():
    return two_groups_unpaired.mean_diff.plot(horizontal=True, swarm_side='left');

# Empty Circle
@pytest.mark.mpl_image_compare(tolerance=8)
def test_392_Empty_Circle():
    return multi_2group_unpaired.mean_diff.plot(horizontal=True, empty_circle=True);

# Table kwargs
@pytest.mark.mpl_image_compare(tolerance=8)
def test_393_Horizontal_Table_Kwargs():
    return multi_2group_unpaired.mean_diff.plot(horizontal=True, horizontal_table_kwargs={'color': 'red', 'alpha': 0.5, 'text_color': 'white',
                                                                       'text_units':'mm', 'label': 'delta mm', 'control_marker': 'o',});

# Gridkey
# Two Group
@pytest.mark.mpl_image_compare(tolerance=8)
def test_394_2group_unpaired_meandiff_gridkey_autoparser():
    return two_groups_unpaired.mean_diff.plot(horizontal=True, gridkey_rows='auto');

@pytest.mark.mpl_image_compare(tolerance=8)
def test_395_2group_paired_meandiff_gridkey_autoparser():
    return two_groups_paired.mean_diff.plot(horizontal=True, gridkey_rows='auto');

# Multi 2 Group
@pytest.mark.mpl_image_compare(tolerance=8)
def test_396_multi_2group_unpaired_meandiff_gridkey_userdefinedrows():
    return multi_groups_unpaired.mean_diff.plot(horizontal=True, gridkey_rows=['Control', 'Test']);

# Shared Control and Repeated Measures
@pytest.mark.mpl_image_compare(tolerance=8)
def test_397_shared_control_meandiff_gridkey_autoparser():
    return shared_control.mean_diff.plot(horizontal=True, gridkey_rows='auto');

@pytest.mark.mpl_image_compare(tolerance=8)
def test_398_repeated_measures_meandiff_gridkey_autoparser():
    return repeated_measures.mean_diff.plot(horizontal=True, gridkey_rows='auto');


# Multi groups
@pytest.mark.mpl_image_compare(tolerance=8)
def test_399_multigroups_unpaired_meandiff_gridkey_userdefinedrows():
    return multi_groups_unpaired.mean_diff.plot(horizontal=True, gridkey_rows=['Control', 'Test']);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_400_multigroups_unpaired_meandiff_gridkey_autoparser():
    return multi_groups_unpaired.mean_diff.plot(horizontal=True, gridkey_rows='auto');

@pytest.mark.mpl_image_compare(tolerance=8)
def test_401_multigroups_paired_meandiff_gridkey_userdefinedrows():
    return multi_groups_paired_baseline.mean_diff.plot(horizontal=True, gridkey_rows=['Control', 'Test']);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_402_multigroups_paired_meandiff_gridkey_autoparser():
    return multi_groups_paired_baseline.mean_diff.plot(horizontal=True, gridkey_rows='auto');

# Proportions
@pytest.mark.mpl_image_compare(tolerance=8)
def test_403_multigroups_prop_unpaired_meandiff_gridkey_userdefinedrows():
    return multi_groups_unpaired_prop.mean_diff.plot(horizontal=True, gridkey_rows=['Control', 'Test']);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_404_multigroups_prop_unpaired_meandiff_gridkey_autoparser():
    return multi_groups_unpaired_prop.mean_diff.plot(horizontal=True, gridkey_rows='auto');

@pytest.mark.mpl_image_compare(tolerance=8)
def test_405_multigroups_prop_paired_meandiff_gridkey_userdefinedrows():
    return multi_groups_paired_baseline_prop.mean_diff.plot(horizontal=True, gridkey_rows=['Control', 'Test']);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_406_multigroups_prop_paired_meandiff_gridkey_autoparser():
    return multi_groups_paired_baseline_prop.mean_diff.plot(horizontal=True, gridkey_rows='auto');


# delta-delta
@pytest.mark.mpl_image_compare(tolerance=8)
def test_407_delta_delta_unpaired_meandiff_gridkey_autoparser():
    return delta_delta_unpaired.mean_diff.plot(horizontal=True, gridkey_rows='auto');

@pytest.mark.mpl_image_compare(tolerance=8)
def test_408_delta_delta_paired_meandiff_gridkey_autoparser():
    return delta_delta_paired_baseline.mean_diff.plot(horizontal=True, gridkey_rows='auto');


# mini-meta
@pytest.mark.mpl_image_compare(tolerance=8)
def test_409_mini_meta_unpaired_meandiff_gridkey_userdefinedrows():
    return mini_meta_unpaired.mean_diff.plot(horizontal=True, gridkey_rows=['Control', 'Test']);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_410_mini_meta_unpaired_meandiff_gridkey_autoparser():
    return mini_meta_unpaired.mean_diff.plot(horizontal=True, gridkey_rows='auto');

@pytest.mark.mpl_image_compare(tolerance=8)
def test_411_mini_meta_paired_meandiff_gridkey_userdefinedrows():
    return mini_meta_paired_baseline.mean_diff.plot(horizontal=True, gridkey_rows=['Control', 'Test']);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_412_mini_meta_paired_meandiff_gridkey_autoparser():
    return mini_meta_paired_baseline.mean_diff.plot(horizontal=True, gridkey_rows='auto');

# Gridkey kwargs
multi_2group_paired_test = load(df, idx=(("Control 1","Control 2",),("Test 1", "Test 2"),), paired='baseline', id_col='ID')
@pytest.mark.mpl_image_compare(tolerance=8)
def test_413_gridkey_merge_pairs_and_autoparser():
    return multi_2group_paired_test.mean_diff.plot(horizontal=True, gridkey_rows=['Control', 'Test'], gridkey_kwargs={'merge_pairs': True});

gridkey_kwargs = {'show_es': False, 'show_Ns': False, 'marker': '√'}
@pytest.mark.mpl_image_compare(tolerance=8)
def test_414_gridkey_kwargs_and_autoparser():
    return two_groups_unpaired.mean_diff.plot(horizontal=True, gridkey_rows='auto', gridkey_kwargs=gridkey_kwargs);

# Table hide
@pytest.mark.mpl_image_compare(tolerance=8)
def test_415_Horizontal_Table_hide():
    return multi_2group_unpaired.mean_diff.plot(horizontal=True, horizontal_table_kwargs={'show': False});

# Delta-dots
@pytest.mark.mpl_image_compare(tolerance=8)
def test_416_delta_dot_hide():
    return multi_2group_paired.mean_diff.plot(horizontal=True, delta_dot=False);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_417_delta_dot_kwargs():
    return multi_2group_paired.mean_diff.plot(horizontal=True, delta_dot_kwargs={"color":'red', "alpha":0.1, 'zorder': 2, 'size': 5, 'side': 'left'});

# Contrast bars
@pytest.mark.mpl_image_compare(tolerance=8)
def test_418_shared_control_meandiff_showcontrastbars():
    return shared_control.mean_diff.plot(horizontal=True, contrast_bars=True, swarm_bars=False);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_419_shared_control_meandiff_hidecontrastbars():
    return shared_control.mean_diff.plot(horizontal=True, contrast_bars=False, swarm_bars=False);

contrast_kwargs = {'color': "red", 'alpha': 0.2}
@pytest.mark.mpl_image_compare(tolerance=8)
def test_420_shared_control_meandiff_contrastbars_kwargs():
    return shared_control.mean_diff.plot(horizontal=True, contrast_bars=True, contrast_bars_kwargs = contrast_kwargs, swarm_bars=False);

# Summary bars
summary_bars=[0, 1]
@pytest.mark.mpl_image_compare(tolerance=8)
def test_421_shared_control_meandiff_summarybars():
    return shared_control.mean_diff.plot(horizontal=True, summary_bars=[0, 1], swarm_bars=False, contrast_bars=False,);

summary_bars_kwargs = {'color': "black", 'alpha': 0.2, 'span_ax': True}
@pytest.mark.mpl_image_compare(tolerance=8)
def test_422_shared_control_meandiff_summarybars_kwargs():
    return shared_control.mean_diff.plot(horizontal=True, summary_bars=[0, 1], summary_bars_kwargs = summary_bars_kwargs,
                                         contrast_bars=False, swarm_bars=False);

# Add counts to prop plots
@pytest.mark.mpl_image_compare(tolerance=8)
def test_423_shared_control_propdiff_show_counts():
    return shared_control_prop.mean_diff.plot(horizontal=True, prop_sample_counts=True,)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_424_repeated_measures_baseline_propdiff_show_counts():
    return repeated_measures_baseline_prop.mean_diff.plot(horizontal=True, prop_sample_counts=True,)

@pytest.mark.mpl_image_compare(tolerance=8)
def test_425_repeated_measures_baseline_propdiff_show_counts_and_kwargs():
    return repeated_measures_baseline_prop.mean_diff.plot(horizontal=True,
            prop_sample_counts=True, prop_sample_counts_kwargs={"color": "red", "fontsize": 12, "fontweight": "bold"})

# Effect size paired lines
@pytest.mark.mpl_image_compare(tolerance=8)
def test_426_repeatedmeasures_meandiff_show_es_paired_lines():
    return repeated_measures.mean_diff.plot(horizontal=True, es_paired_lines=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_427_repeatedmeasures_meandiff_hide_es_paired_lines():
    return repeated_measures.mean_diff.plot(horizontal=True, es_paired_lines=False);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_428_multigroups_paired_meandiff_es_paired_lines_kwargs():
    return multi_groups_paired_baseline.mean_diff.plot(horizontal=True, es_paired_lines=True, es_paired_lines_kwargs={'color':'red', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.5});
