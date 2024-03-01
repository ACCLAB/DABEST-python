import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as Ticker

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


unpaired = load(df, idx=(("Control 1", "Test 1"), ("Control 2", "Test 2"), ("Control 3", "Test 3")),
                    mini_meta=True)


baseline = load(df, id_col = "ID",
                       idx=(("Control 1", "Test 1"), ("Control 2", "Test 2"), ("Control 3", "Test 3")),
                       paired = "baseline", mini_meta=True)


sequential = load(df, id_col = "ID",
                       idx=(("Control 1", "Test 1"), ("Control 2", "Test 2"), ("Control 3", "Test 3")),
                       paired = "sequential", mini_meta=True)




@pytest.mark.mpl_image_compare(tolerance=8)
def test_60_cummings_unpaired_mini_meta_meandiff():
    return unpaired.mean_diff.plot();


@pytest.mark.mpl_image_compare(tolerance=8)
def test_61_cummings_sequential_mini_meta_meandiff():
    return sequential.mean_diff.plot();


@pytest.mark.mpl_image_compare(tolerance=8)
def test_62_cummings_baseline_mini_meta_meandiff():
    return baseline.mean_diff.plot();


@pytest.mark.mpl_image_compare(tolerance=8)
def test_63_mini_meta_plot_ylabel():
    return baseline.mean_diff.plot(swarm_label="This is my\nrawdata",
                                   contrast_label="The bootstrap\ndistribtions!");


@pytest.mark.mpl_image_compare(tolerance=8)
def test_64_mini_meta_plot_change_palette_a():
    return unpaired.mean_diff.plot(custom_palette="Dark2");


@pytest.mark.mpl_image_compare(tolerance=8)
def test_65_mini_meta_dot_sizes():
    return sequential.mean_diff.plot(show_pairs=False,raw_marker_size=3,
                                       es_marker_size=12);


@pytest.mark.mpl_image_compare(tolerance=8)
def test_66_mini_meta_change_ylims():
    return sequential.mean_diff.plot(swarm_ylim=(0, 5),
                                    contrast_ylim=(-2, 2),
                                    fig_size=(15,6));


@pytest.mark.mpl_image_compare(tolerance=8)
def test_67_mini_meta_invert_ylim():
    return sequential.mean_diff.plot(contrast_ylim=(2, -2),
                                       contrast_label="More negative is better!");


@pytest.mark.mpl_image_compare(tolerance=8)
def test_68_mini_meta_median_diff():
    return sequential.median_diff.plot();


@pytest.mark.mpl_image_compare(tolerance=8)
def test_69_mini_meta_cohens_d():
    return unpaired.cohens_d.plot();


@pytest.mark.mpl_image_compare(tolerance=8)
def test_70_mini_meta_not_show():
    return unpaired.mean_diff.plot(show_mini_meta=False);
