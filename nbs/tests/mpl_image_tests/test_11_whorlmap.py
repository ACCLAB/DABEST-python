import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import dabest
from dabest.multi import combine, whorlmap

def create_delta_dataset(N=50, 
                        seed=9999, 
                        second_quarter_adjustment=3, 
                        third_quarter_adjustment= -0.5,
                        fourth_quarter_adjustment= -3, 
                        scale4=1, initial_loc = 10):
    """Create a sample dataset for delta-delta analysis."""
    np.random.seed(seed)

    # Create samples
    y = norm.rvs(loc=initial_loc, scale=0.4, size=N*4)
    y[N:2*N] = norm.rvs(loc=initial_loc + second_quarter_adjustment, scale= 1, size=N) 
    y[2*N:3*N] = norm.rvs(loc=initial_loc + third_quarter_adjustment, scale=0.4, size=N)
    y[3*N:4*N] = norm.rvs(loc=initial_loc + fourth_quarter_adjustment, scale=scale4, size=N)

    # Treatment, Rep, Genotype, and ID columns
    treatment = np.repeat(['Placebo', 'Drug'], N*2).tolist()
    genotype = np.repeat(['W', 'M', 'W', 'M'], N).tolist()
    id_col = list(range(0, N*2)) * 2

    # Combine all columns into a DataFrame
    df = pd.DataFrame({
        'ID': id_col,
        'Genotype': genotype,
        'Treatment': treatment,
        'Transcript Level': y
    })
    return df

dabest_objects_2d = [[None for _ in range(2)] for _ in range(2)]
labels_2d = ["Transcript 1", "Transcript 2"]
row_labels_2d = ["Drug A", "Drug B"]
drug_effect_2d = [[.9, 2], 
             [0.1, -.3],
                                ]
drug_effect_scale_2d = [[5, 10], 
             [7, .2],
             ]
seeds = [1, 1000]

for i in range(len(row_labels_2d)):
    for j in range(len(labels_2d)):
        df = create_delta_dataset(seed=seeds[i], 
                                  fourth_quarter_adjustment=drug_effect_2d[i][j],
                                  scale4=drug_effect_scale_2d[i][j],
                                 initial_loc = 20)
        dabest_objects_2d[i][j] = dabest.load(data=df, 
                       x=["Genotype", "Genotype"], 
                       y="Transcript Level", 
                       delta2=True, 
                       experiment="Treatment")

multi_2d_mean_diff = combine(dabest_objects_2d, labels_2d, row_labels=row_labels_2d, effect_size="mean_diff")
multi_2d_delta_g = combine(dabest_objects_2d, labels_2d, row_labels=row_labels_2d, effect_size="delta_g")
multi_1d = combine(dabest_objects_2d[0], labels_2d, row_labels="Drug A", effect_size="mean_diff")

dabest_objects_2d_two_group_delta = [[None for _ in range(2)] for _ in range(2)]
for i in range(len(row_labels_2d)):
    for j in range(len(labels_2d)):
        df = create_delta_dataset(seed=seeds[i], 
                                  fourth_quarter_adjustment=drug_effect_2d[i][j],
                                  scale4=drug_effect_scale_2d[i][j],
                                 initial_loc = 20)
        dabest_objects_2d_two_group_delta[i][j] = dabest.load(data=df, 
                       x="Treatment", 
                       y="Transcript Level", 
                       idx = ("Placebo", "Drug"))
multi_2d_two_group_delta_mean_diff = combine(dabest_objects_2d_two_group_delta, labels_2d, row_labels=row_labels_2d, effect_size="mean_diff")

@pytest.mark.mpl_image_compare(tolerance=8)
def test_550_forest_plot_2d_mean_diff():
    plt.rcdefaults()
    f, a = multi_2d_mean_diff.forest_plot(
                forest_plot_title="Forest Plot",
                forest_plot_kwargs={'marker_size': 6}
            )
    return f

@pytest.mark.mpl_image_compare(tolerance=8)
def test_551_whorlmap_2d_mean_diff():
    plt.rcdefaults()
    f, a, m = multi_2d_mean_diff.whorlmap(
                title="Whorlmap",
                chop_tail=2.5,  # Remove 5% extreme values
                fig_size=(2, 2)
            )
    return f

@pytest.mark.mpl_image_compare(tolerance=8)
def test_552_whorlmap_2d_delta_g():
    plt.rcdefaults()
    f, a, m = multi_2d_delta_g.whorlmap(
                title="Delta g Whorlmap",
                chop_tail=2.5,  # Remove 5% extreme values
                fig_size=(2, 2)
            )
    return f

@pytest.mark.mpl_image_compare(tolerance=8)
def test_553_whorlmap_1d():
    plt.rcdefaults()
    f, a, m = multi_1d.whorlmap(
                chop_tail=2.5,  # Remove 5% extreme values
                fig_size=(2, 1)

            )
    return f

@pytest.mark.mpl_image_compare(tolerance=8)
def test_554_whorlmap_2d_two_group_delta_mean_diff():
    plt.rcdefaults()
    f, a, m = multi_2d_two_group_delta_mean_diff.whorlmap(
                chop_tail=2.5,  # Remove 5% extreme values
                fig_size=(2, 2)
            )
    return f