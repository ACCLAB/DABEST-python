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

multi_2group = load(df, idx=(("Control 1", "Test 1",), ("Control 2", "Test 2"),),)

multi_2group_paired = load(df, idx=(("Control 1", "Test 1"),
                                         ("Control 2", "Test 2")),paired='baseline', id_col='ID')

shared_control = load(df, idx=("Control 1", "Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Test 6"))

repeated_measures = load(df, idx=("Control 1", "Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Test 6"), paired='baseline', id_col='ID')

multi_groups_paired_baseline = load(df,idx=(("Control 1","Test 1", "Test 2"),("Control 2", "Test 3"),("Control 3", "Test 4", "Test 5", "Test 6"),),
                                    paired='baseline', id_col='ID')

multi_groups = load(df, idx=(("Control 1", "Test 1",), ("Control 2", "Test 2", "Test 3"),
                              ("Control 3", "Test 4", "Test 5", "Test 6"),),)

unpaired_delta_delta = load(data=df_delta, x=["Genotype", "Genotype"], y="Y", delta2=True, experiment="Treatment")

unpaired_mini_meta = load(df, idx=(("Control 1", "Test 1"), ("Control 2", "Test 2"), ("Control 3", "Test 3")), mini_meta=True)

multi_groups_change_idx_original = load(df,
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

# Jitter tests
np.random.seed(9999) # Fix the seed to ensure reproducibility of results.
Ns = 20 # The number of samples taken from each population
# Create samples
c1 = [0.5]*Ns + [1.5]*Ns
c2 = [2]*Ns + [1]*Ns
t1 = [1]*Ns + [2]*Ns
t2 = [1.5]*Ns + [2.5]*Ns
t3 = [2]*Ns + [1]*Ns
t4 = [1]*Ns + [2]*Ns
t5 = [1.5]*Ns + [2.5]*Ns
id_col = pd.Series(range(1, 2*Ns+1))
df_jittertest= pd.DataFrame({'Control 1' : c1,     'Test 1' : t1,
                 'Control 2' : c2,     'Test 2' : t2, 'Test 3' : t3,
                    'Test 4'    : t4,     'Test 5' : t5, 'ID'  : id_col})
multi_2group_jitter = load(df, idx=(("Control 1","Test 1",), ("Control 2", "Test 2"), ), paired='baseline', id_col = 'ID')


# Tests

# Empty circle
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


# Change palette 
@pytest.mark.mpl_image_compare(tolerance=8)
def test_214_change_idx_order_custom_palette_original():
    return multi_groups_change_idx_original.mean_diff.plot(custom_palette=palette);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_215_change_idx_order_custom_palette_new():
    return multi_groups_change_idx_new.mean_diff.plot(custom_palette=palette);

# Swarm bars
@pytest.mark.mpl_image_compare(tolerance=8)
def test_216_cummings_shared_control_meandiff_showswarmbars():
    return shared_control.mean_diff.plot(swarm_bars=True, contrast_bars=False);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_217_cummings_shared_control_meandiff_hideswarmbars():
    return shared_control.mean_diff.plot(swarm_bars=False, contrast_bars=False);

swarm_kwargs = {'color': "red", 'alpha': 0.2}
@pytest.mark.mpl_image_compare(tolerance=8)
def test_218_cummings_shared_control_meandiff_swarmbars_kwargs():
    return shared_control.mean_diff.plot(swarm_bars=True, swarm_bars_kwargs = swarm_kwargs, contrast_bars=False);


# Contrast bars
@pytest.mark.mpl_image_compare(tolerance=8)
def test_219_cummings_shared_control_meandiff_showcontrastbars():
    return shared_control.mean_diff.plot(contrast_bars=True, swarm_bars=False);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_220_cummings_shared_control_meandiff_hidecontrastbars():
    return shared_control.mean_diff.plot(contrast_bars=False, swarm_bars=False);

contrast_kwargs = {'color': "red", 'alpha': 0.2}
@pytest.mark.mpl_image_compare(tolerance=8)
def test_221_cummings_shared_control_meandiff_contrastbars_kwargs():
    return shared_control.mean_diff.plot(contrast_bars=True, contrast_bars_kwargs = contrast_kwargs, swarm_bars=False);


# Summary bars
summary_bars=[0, 1]
@pytest.mark.mpl_image_compare(tolerance=8)
def test_222_cummings_shared_control_meandiff_summarybars():
    return shared_control.mean_diff.plot(summary_bars=[0, 1], swarm_bars=False, contrast_bars=False,);

summary_bars_kwargs = {'color': "black", 'alpha': 0.2, 'span_ax': True}
@pytest.mark.mpl_image_compare(tolerance=8)
def test_223_cummings_shared_control_meandiff_summarybars_kwargs():
    return shared_control.mean_diff.plot(summary_bars=[0, 1], summary_bars_kwargs = summary_bars_kwargs,
                                         contrast_bars=False, swarm_bars=False);


# Delta text
@pytest.mark.mpl_image_compare(tolerance=8)
def test_224_multi_2group_meandiff_showdeltatext():
    return multi_2group.mean_diff.plot(delta_text=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_225_multi_2group_meandiff_hidedeltatext():
    return multi_2group.mean_diff.plot(delta_text=False);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_226_multi_2group_meandiff_deltatext_kwargs():
    return multi_2group.mean_diff.plot(delta_text=True, delta_text_kwargs={"color":"red", "rotation":45, "va":"bottom", "alpha":0.7});

@pytest.mark.mpl_image_compare(tolerance=8)
def test_227_multi_2group_meandiff_deltatext_kwargs_specificy_coordinates():
    return multi_2group.mean_diff.plot(delta_text=True, delta_text_kwargs={"x_coordinates":(0.5, 2.75), "y_coordinates":(0.5, -1.7)});

@pytest.mark.mpl_image_compare(tolerance=8)
def test_228_multi_2group_meandiff_deltatext_kwargs_x_adjust():
    return multi_2group.mean_diff.plot(delta_text=True, delta_text_kwargs={"offset":0.1});

# Jitter
@pytest.mark.mpl_image_compare(tolerance=8)
def test_229_samevalues_jitter():
    return multi_2group_jitter.mean_diff.plot(slopegraph_kwargs={'jitter': 1});

# Delta-dots
@pytest.mark.mpl_image_compare(tolerance=8)
def test_230_delta_dot_hide():
    return multi_2group_paired.mean_diff.plot(delta_dot=False);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_231_delta_dot_kwargs():
    return multi_2group_paired.mean_diff.plot(delta_dot_kwargs={"color":'red', "alpha":0.1, 'zorder': 2, 'size': 5, 'side': 'left'});

# Effect size paired lines
@pytest.mark.mpl_image_compare(tolerance=8)
def test_232_repeatedmeasures_meandiff_show_es_paired_lines():
    return repeated_measures.mean_diff.plot(es_paired_lines=True);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_233_repeatedmeasures_meandiff_hide_es_paired_lines():
    return repeated_measures.mean_diff.plot(es_paired_lines=False);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_234_multigroups_paired_meandiff_es_paired_lines_kwargs():
    return multi_groups_paired_baseline.mean_diff.plot(es_paired_lines=True, es_paired_lines_kwargs={'color':'red', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.5});
