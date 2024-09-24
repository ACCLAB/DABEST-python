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


df = create_demo_dataset()
df_delta = create_demo_dataset_delta()

two_groups_unpaired = load(df, idx=("Control 1", "Test 1"))

multi_2group = load(
    df,
    idx=(
        (
            "Control 1",
            "Test 1",
        ),
        ("Control 2", "Test 2"),
    ),
)

shared_control = load(
    df, idx=("Control 1", "Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Test 6")
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
)

unpaired_delta_delta = load(
    data=df_delta, x=["Genotype", "Genotype"], y="Y", delta2=True, experiment="Treatment"
)

unpaired_mini_meta = load(df, idx=(("Control 1", "Test 1"), ("Control 2", "Test 2"), ("Control 3", "Test 3")),
                    mini_meta=True)

multi_groups_change_idx_original = load(
    df,
    idx=(
        ("Control 1", "Test 1", "Test 2"),
        ("Control 2", "Test 3", "Test 4"),
        ("Control 3", "Test 5", "Test 6"),
    ),
)
multi_groups_change_idx_new = load(
    df,
    idx=(
        ("Control 1", "Control 2", "Control 3"),
        ("Test 1", "Test 3", "Test 5"),
        ("Test 2", "Test 4", "Test 6"),
    ),
)
palette = {"Control 1": sns.color_palette("magma")[5],
           "Test 1": sns.color_palette("magma")[3],
           "Test 2": sns.color_palette("magma")[1],
           "Control 2": sns.color_palette("magma")[5],
           "Test 3": sns.color_palette("magma")[3],
           "Test 4": sns.color_palette("magma")[1],
           "Control 3": sns.color_palette("magma")[5],
           "Test 5": sns.color_palette("magma")[3],
           "Test 6": sns.color_palette("magma")[1]}

@pytest.mark.mpl_image_compare(tolerance=8)
def test_207_gardner_altman_meandiff_empty_circle():
    return two_groups_unpaired.mean_diff.plot(empty_circle=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_208_cummings_two_group_unpaired_meandiff_empty_circle():
    return two_groups_unpaired.mean_diff.plot(empty_circle=True, float_contrast=False);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_209_cummings_shared_control_meandiff_empty_circle():
    return shared_control.mean_diff.plot(empty_circle=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_210_cummings_multi_groups_meandiff_empty_circle():
    return multi_groups.mean_diff.plot(empty_circle=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_211_cummings_multi_2_group_meandiff_empty_circle():
    return multi_2group.mean_diff.plot(empty_circle=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_212_cummings_unpaired_delta_delta_meandiff_empty_circle():
    return unpaired_delta_delta.mean_diff.plot(empty_circle=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_213_cummings_unpaired_mini_meta_meandiff_empty_circle():
    return unpaired_mini_meta.mean_diff.plot(empty_circle=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_214_change_idx_order_custom_palette_original():
    return multi_groups_change_idx_original.mean_diff.plot(custom_palette=palette);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_215_change_idx_order_custom_palette_new():
    return multi_groups_change_idx_new.mean_diff.plot(custom_palette=palette);
