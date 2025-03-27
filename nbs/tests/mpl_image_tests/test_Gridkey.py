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
multi_groups_paired = load(df,idx=(("Control 1","Test 1",),("Control 2", "Test 2", "Test 3"),("Control 3", "Test 4", "Test 5", "Test 6"),),
                           paired='baseline', id_col='ID')


# Proportions
multi_groups_unpaired_prop = load(df_prop, idx=(("Control 1","Test 1",),("Control 2", "Test 2", "Test 3"),("Control 3", "Test 4", "Test 5", "Test 6"),),
                                    proportional=True,)

multi_groups_paired_baseline_prop = load(df_prop, idx=(("Control 1","Test 1",),("Control 2", "Test 2", "Test 3"),("Control 3", "Test 4", "Test 5", "Test 6"),),
                            paired="baseline", id_col="ID", proportional=True,)

# delta-delta
delta_delta_unpaired = load(data=df_delta, x=["Genotype", "Genotype"], y="Y", delta2=True, experiment="Treatment")
delta_delta_paired = load(data = df_delta, x = ["Treatment", "Rep"], y = "Y", delta2 = True, experiment = "Genotype", paired="baseline", id_col="ID")

# mini_meta
mini_meta_unpaired = load(df, idx=(("Control 1", "Test 1"), ("Control 2", "Test 2"), ("Control 3", "Test 3")),mini_meta=True)
mini_meta_paired = load(df, idx=(("Control 1", "Test 1"), ("Control 2", "Test 2"), ("Control 3", "Test 3")),mini_meta=True, paired='baseline', id_col='ID')


# Two Group
@pytest.mark.mpl_image_compare(tolerance=8)
def test_250_2group_unpaired_meandiff_gridkey_autoparser():
    plt.rcdefaults()
    return two_groups_unpaired.mean_diff.plot(gridkey='auto');

@pytest.mark.mpl_image_compare(tolerance=8)
def test_251_2group_unpaired_meandiff_gridkey_userdefinedrows():
    plt.rcdefaults()
    return two_groups_unpaired.mean_diff.plot(gridkey=['Control', 'Test']);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_252_2group_paired_meandiff_gridkey_autoparser():
    plt.rcdefaults()
    return two_groups_paired.mean_diff.plot(gridkey='auto');

@pytest.mark.mpl_image_compare(tolerance=8)
def test_253_2group_paired_meandiff_gridkey_userdefinedrows():
    plt.rcdefaults()
    return two_groups_paired.mean_diff.plot(gridkey=['Control', 'Test']);

# Multi 2 Group
@pytest.mark.mpl_image_compare(tolerance=8)
def test_254_multi_2group_unpaired_meandiff_gridkey_autoparser():
    plt.rcdefaults()
    return multi_groups_unpaired.mean_diff.plot(gridkey='auto');

@pytest.mark.mpl_image_compare(tolerance=8)
def test_255_multi_2group_unpaired_meandiff_gridkey_userdefinedrows():
    plt.rcdefaults()
    return multi_groups_unpaired.mean_diff.plot(gridkey=['Control', 'Test']);

# Shared Control and Repeated Measures
@pytest.mark.mpl_image_compare(tolerance=8)
def test_256_shared_control_meandiff_gridkey_autoparser():
    plt.rcdefaults()
    return shared_control.mean_diff.plot(gridkey='auto');

@pytest.mark.mpl_image_compare(tolerance=8)
def test_257_shared_control_meandiff_gridkey_userdefinedrows():
    plt.rcdefaults()
    return shared_control.mean_diff.plot(gridkey=['Control', 'Test']);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_258_repeated_measures_meandiff_gridkey_autoparser():
    plt.rcdefaults()
    return repeated_measures.mean_diff.plot(gridkey='auto');

@pytest.mark.mpl_image_compare(tolerance=8)
def test_259_repeated_measures_meandiff_gridkey_userdefinedrows():
    plt.rcdefaults()
    return repeated_measures.mean_diff.plot(gridkey=['Control', 'Test']);


# Multi groups
@pytest.mark.mpl_image_compare(tolerance=8)
def test_260_multigroups_unpaired_meandiff_gridkey_userdefinedrows():
    plt.rcdefaults()
    return multi_groups_unpaired.mean_diff.plot(gridkey=['Control', 'Test']);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_261_multigroups_unpaired_meandiff_gridkey_autoparser():
    plt.rcdefaults()
    return multi_groups_unpaired.mean_diff.plot(gridkey='auto');

@pytest.mark.mpl_image_compare(tolerance=8)
def test_262_multigroups_paired_meandiff_gridkey_userdefinedrows():
    plt.rcdefaults()
    return multi_groups_paired.mean_diff.plot(gridkey=['Control', 'Test']);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_263_multigroups_paired_meandiff_gridkey_autoparser():
    plt.rcdefaults()
    return multi_groups_paired.mean_diff.plot(gridkey='auto');

# Proportions
@pytest.mark.mpl_image_compare(tolerance=8)
def test_264_multigroups_prop_unpaired_meandiff_gridkey_userdefinedrows():
    plt.rcdefaults()
    return multi_groups_unpaired_prop.mean_diff.plot(gridkey=['Control', 'Test']);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_265_multigroups_prop_unpaired_meandiff_gridkey_autoparser():
    plt.rcdefaults()
    return multi_groups_unpaired_prop.mean_diff.plot(gridkey='auto');

@pytest.mark.mpl_image_compare(tolerance=8)
def test_266_multigroups_prop_paired_meandiff_gridkey_userdefinedrows():
    plt.rcdefaults()
    return multi_groups_paired_baseline_prop.mean_diff.plot(gridkey=['Control', 'Test']);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_267_multigroups_prop_paired_meandiff_gridkey_autoparser():
    plt.rcdefaults()
    return multi_groups_paired_baseline_prop.mean_diff.plot(gridkey='auto');


# delta-delta
@pytest.mark.mpl_image_compare(tolerance=8)
def test_268_delta_delta_unpaired_meandiff_gridkey_autoparser():
    plt.rcdefaults()
    return delta_delta_unpaired.mean_diff.plot(gridkey='auto');

@pytest.mark.mpl_image_compare(tolerance=8)
def test_269_delta_delta_paired_meandiff_gridkey_autoparser():
    plt.rcdefaults()
    return delta_delta_paired.mean_diff.plot(gridkey='auto');


# mini-meta
@pytest.mark.mpl_image_compare(tolerance=8)
def test_270_mini_meta_unpaired_meandiff_gridkey_userdefinedrows():
    plt.rcdefaults()
    return mini_meta_unpaired.mean_diff.plot(gridkey=['Control', 'Test']);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_271_mini_meta_unpaired_meandiff_gridkey_autoparser():
    plt.rcdefaults()
    return mini_meta_unpaired.mean_diff.plot(gridkey='auto');

@pytest.mark.mpl_image_compare(tolerance=8)
def test_272_mini_meta_paired_meandiff_gridkey_userdefinedrows():
    plt.rcdefaults()
    return mini_meta_paired.mean_diff.plot(gridkey=['Control', 'Test']);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_273_mini_meta_paired_meandiff_gridkey_autoparser():
    plt.rcdefaults()
    return mini_meta_paired.mean_diff.plot(gridkey='auto');


# Gridkey kwargs
multi_2group_paired_test = load(df, idx=(("Control 1","Control 2",),("Test 1", "Test 2"),), paired='baseline', id_col='ID')
@pytest.mark.mpl_image_compare(tolerance=8)
def test_274_gridkey_merge_pairs_and_autoparser():
    plt.rcdefaults()
    return multi_2group_paired_test.mean_diff.plot(gridkey=['Control', 'Test'], gridkey_kwargs={'merge_pairs': True});

gridkey_kwargs = {'show_es': False, 'show_Ns': False, 'marker': '√'}
@pytest.mark.mpl_image_compare(tolerance=8)
def test_275_gridkey_kwargs_and_autoparser():
    plt.rcdefaults()
    return two_groups_unpaired.mean_diff.plot(gridkey='auto', gridkey_kwargs=gridkey_kwargs);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_276_gridkey_fontsize_and_autoparser():
    plt.rcdefaults()
    return two_groups_unpaired.mean_diff.plot(gridkey='auto', gridkey_kwargs={'fontsize': 15});

@pytest.mark.mpl_image_compare(tolerance=8)
def test_277_gridkey_labels_fontsize_and_autoparser():
    plt.rcdefaults()
    return two_groups_unpaired.mean_diff.plot(gridkey='auto', gridkey_kwargs={'labels_fontsize': 15});

@pytest.mark.mpl_image_compare(tolerance=8)
def test_278_gridkey_labels_fontsize_and_fontsize_and_autoparser():
    plt.rcdefaults()
    return two_groups_unpaired.mean_diff.plot(gridkey='auto', 
                                              gridkey_kwargs={'fontsize': 8, 'labels_fontsize': 15});