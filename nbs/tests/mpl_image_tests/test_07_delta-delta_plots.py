import pytest
import numpy as np
import pandas as pd
from scipy.stats import norm


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as Ticker

from dabest._api import load

def create_demo_dataset_delta(seed=9999, N=20):
    
    import numpy as np
    import pandas as pd
    from scipy.stats import norm # Used in generation of populations.

    np.random.seed(seed) # Fix the seed so the results are replicable.
    # pop_size = 10000 # Size of each population.

    from scipy.stats import norm # Used in generation of populations.

    # Create samples
    y = norm.rvs(loc=3, scale=0.4, size=N*4)
    y[N:2*N] = y[N:2*N]+1
    y[2*N:3*N] = y[2*N:3*N]-0.5
    ind = np.random.binomial(1, 0.5, size=N*4)
    ind[N:2*N] = np.random.binomial(1, 0.2, size=N)
    ind[2*N:3*N] = np.random.binomial(1, 0.7, size=N)

    # Add drug column
    t1 = np.repeat('Placebo', N*2).tolist()
    t2 = np.repeat('Drug', N*2).tolist()
    treatment = t1 + t2 

    # Add a `rep` column as the first variable for the 2 replicates of experiments done
    rep = []
    for i in range(N*2):
        rep.append('Rep1')
        rep.append('Rep2')

    # Add a `genotype` column as the second variable
    wt = np.repeat('W', N).tolist()
    mt = np.repeat('M', N).tolist()
    wt2 = np.repeat('W', N).tolist()
    mt2 = np.repeat('M', N).tolist()


    genotype = wt + mt + wt2 + mt2

    # Add an `id` column for paired data plotting.
    id = list(range(0, N*2))
    id_col = id + id 


    # Combine all columns into a DataFrame.
    df = pd.DataFrame({'ID'        : id_col,
                      'Rep'        : rep,
                       'Genotype'  : genotype, 
                       'Treatment' : treatment,
                       'Y'         : y,
                       'Cat'       :ind
                    })
    return df


df = create_demo_dataset_delta()


unpaired = load(data = df, x = ["Genotype", "Genotype"], y = "Y", delta2 = True, 
                experiment = "Treatment")

unpaired_specified = load(data = df, x = ["Genotype", "Genotype"], y = "Y", 
                          delta2 = True, experiment = "Treatment",
                                        experiment_label = ["Drug", "Placebo"],
                                        x1_level = ["M", "W"])

baseline = load(data = df, x = ["Treatment", "Rep"], y = "Y", delta2 = True, 
                experiment = "Genotype",
                paired="baseline", id_col="ID")

sequential = load(data = df, x = ["Treatment", "Rep"], y = "Y", delta2 = True, 
                experiment = "Genotype",
                paired="sequential", id_col="ID")

unpaired_prop = load(data = df, proportional=True,
                            # id_col="index", paired='baseline', 
                            x = ["Genotype", "Genotype"], 
                            y = "Cat", delta2=True,
                            experiment="Treatment",)

unpaired_specified_prop = load(data = df, proportional=True,
                            # id_col="index", paired='baseline', 
                            x = ["Genotype", "Genotype"], 
                            y = "Cat", delta2=True,
                            experiment="Treatment",
                            experiment_label = ["Drug", "Placebo"],
                            x1_level = ["M", "W"])

paired_prop = load(data = df, proportional=True,
                            id_col="ID", paired='baseline', 
                            x = ["Genotype", "Genotype"], 
                            y = "Cat", delta2=True,
                            experiment="Treatment",)

paired_specified_prop = load(data = df, proportional=True,
                            id_col="ID", paired='baseline', 
                            x = ["Genotype", "Genotype"], 
                            y = "Cat", delta2=True,
                            experiment="Treatment",
                            experiment_label = ["Drug", "Placebo"],
                            x1_level = ["M", "W"])


@pytest.mark.mpl_image_compare(tolerance=8)
def test_47_cummings_unpaired_delta_delta_meandiff():
    plt.rcdefaults()
    return unpaired.mean_diff.plot();


@pytest.mark.mpl_image_compare(tolerance=8)
def test_48_cummings_sequential_delta_delta_meandiff():
    plt.rcdefaults()
    return sequential.mean_diff.plot();


@pytest.mark.mpl_image_compare(tolerance=8)
def test_49_cummings_baseline_delta_delta_meandiff():
    plt.rcdefaults()
    return baseline.mean_diff.plot();


@pytest.mark.mpl_image_compare(tolerance=8)
def test_50_delta_plot_ylabel():
    plt.rcdefaults()
    return baseline.mean_diff.plot(swarm_label="This is my\nrawdata",
                                   contrast_label="The bootstrap\ndistribtions!", 
                                   delta2_label="This is delta!");


@pytest.mark.mpl_image_compare(tolerance=8)
def test_51_delta_plot_change_palette_a():
    plt.rcdefaults()
    return sequential.mean_diff.plot(custom_palette="Dark2");


@pytest.mark.mpl_image_compare(tolerance=8)
def test_52_delta_specified():
    plt.rcdefaults()
    return unpaired_specified.mean_diff.plot();


@pytest.mark.mpl_image_compare(tolerance=8)
def test_53_delta_change_ylims():
    plt.rcdefaults()
    return sequential.mean_diff.plot(swarm_ylim=(0, 9),
                                       contrast_ylim=(-2, 2),
                                       fig_size=(15,6));


@pytest.mark.mpl_image_compare(tolerance=8)
def test_54_delta_invert_ylim():
    plt.rcdefaults()
    return sequential.mean_diff.plot(contrast_ylim=(2, -2),
                                       contrast_label="More negative is better!");


@pytest.mark.mpl_image_compare(tolerance=8)
def test_55_delta_median_diff():
    plt.rcdefaults()
    return sequential.median_diff.plot();


@pytest.mark.mpl_image_compare(tolerance=8)
def test_56_delta_cohens_d():
    plt.rcdefaults()
    return unpaired.cohens_d.plot();


@pytest.mark.mpl_image_compare(tolerance=8)
def test_57_delta_show_delta2():
    plt.rcdefaults()
    return unpaired.mean_diff.plot(show_delta2=False);


@pytest.mark.mpl_image_compare(tolerance=8)
def test_58_delta_axes_invert_ylim():
    plt.rcdefaults()
    return unpaired.mean_diff.plot(delta2_ylim=(2, -2),
                                   delta2_label="More negative is better!");

                            
@pytest.mark.mpl_image_compare(tolerance=8)
def test_59_delta_axes_invert_ylim_not_showing_delta2():
    plt.rcdefaults()
    return unpaired.mean_diff.plot(delta2_ylim=(2, -2),
                                   delta2_label="More negative is better!",
                                   show_delta2=False);

@pytest.mark.mpl_image_compare(tolerance=8)
def test_71_unpaired_delta_g():
    plt.rcdefaults()
    return unpaired.hedges_g.plot();

@pytest.mark.mpl_image_compare(tolerance=8)
def test_72_sequential_delta_g():
    plt.rcdefaults()
    return sequential.hedges_g.plot();

@pytest.mark.mpl_image_compare(tolerance=8)
def test_73_baseline_delta_g():
    plt.rcdefaults()
    return baseline.hedges_g.plot();

@pytest.mark.mpl_image_compare(tolerance=8)
def test_74_unpaired_prop_delta2():
    plt.rcdefaults()
    return unpaired_prop.mean_diff.plot()

@pytest.mark.mpl_image_compare(tolerance=8)
def test_75_unpaired_specified_prop_delta2():
    plt.rcdefaults()
    return unpaired_specified_prop.mean_diff.plot()

@pytest.mark.mpl_image_compare(tolerance=8)
def test_76_paired_prop_delta2():
    plt.rcdefaults()
    return paired_prop.mean_diff.plot()

@pytest.mark.mpl_image_compare(tolerance=8)
def test_77_paired_specified_prop_delta2():
    plt.rcdefaults()
    return paired_specified_prop.mean_diff.plot()