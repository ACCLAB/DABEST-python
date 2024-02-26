import pytest
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib as mpl
import os
from pathlib import Path

mpl.use("Agg")
import matplotlib.ticker as Ticker
import matplotlib.pyplot as plt

from dabest._api import load

import numpy as np
import pandas as pd
from scipy.stats import norm

def create_delta_dataset(N=20, 
                        seed=9999, 
                        second_quarter_adjustment=3, 
                        third_quarter_adjustment=-0.1):
    np.random.seed(seed)  # Set the seed for reproducibility

    # Create samples
    y = norm.rvs(loc=3, scale=0.4, size=N*4)
    y[N:2*N] += second_quarter_adjustment
    y[2*N:3*N] += third_quarter_adjustment

    # Treatment, Rep, Genotype, and ID columns
    treatment = np.repeat(['Placebo', 'Drug'], N*2).tolist()
    rep = ['Rep1', 'Rep2'] * (N*2)
    genotype = np.repeat(['W', 'M', 'W', 'M'], N).tolist()
    id_col = list(range(0, N*2)) * 2

    # Combine all columns into a DataFrame
    df = pd.DataFrame({
        'ID': id_col,
        'Rep': rep,
        'Genotype': genotype,
        'Treatment': treatment,
        'Y': y
    })

    return df

def create_mini_meta_dataset(N=20, seed=9999, control_locs=[3, 3.5, 3.25], control_scales=[0.4, 0.75, 0.4], 
                             test_locs=[3.5, 2.5, 3], test_scales=[0.5, 0.6, 0.75]):
    np.random.seed(seed)  # Set the seed for reproducibility

    # Create samples for controls and tests
    controls_tests = []
    for loc, scale in zip(control_locs + test_locs, control_scales + test_scales):
        controls_tests.append(norm.rvs(loc=loc, scale=scale, size=N))

    # Add a `Gender` column for coloring the data
    gender = ['Female'] * (N // 2) + ['Male'] * (N // 2)

    # Add an `ID` column for paired data plotting
    id_col = list(range(1, N + 1))

    # Combine samples and gender into a DataFrame
    df_columns = {f'Control {i+1}': controls_tests[i] for i in range(len(control_locs))}
    df_columns.update({f'Test {i+1}': controls_tests[i + len(control_locs)] for i in range(len(test_locs))})
    df_columns['Gender'] = gender
    df_columns['ID'] = id_col

    df = pd.DataFrame(df_columns)

    return df

# Generate the first dataset with a different seed and adjustments
df_delta2_drug1 = create_delta_dataset(seed=9999, 
                                      second_quarter_adjustment=1, 
                                      third_quarter_adjustment=-0.5)

# Generate the second dataset with a different seed and adjustments
df_delta2_drug2 = create_delta_dataset(seed=9999, 
                                      second_quarter_adjustment=0.1, 
                                      third_quarter_adjustment=-1)

# Generate the third dataset with the same seed as the first but different adjustments
df_delta2_drug3 = create_delta_dataset(seed=9999, 
                                      second_quarter_adjustment=3, 
                                      third_quarter_adjustment=-0.1)


unpaired_delta_01 = load(data = df_delta2_drug1, 
                         x = ["Genotype", "Genotype"], 
                         y = "Y", delta2 = True, 
                         experiment = "Treatment")

unpaired_delta_02 = load(data = df_delta2_drug2, 
                         x = ["Genotype", "Genotype"], 
                         y = "Y", delta2 = True, 
                         experiment = "Treatment")

unpaired_delta_03 = load(data = df_delta2_drug3, 
                         x = ["Genotype", "Genotype"], 
                         y = "Y", 
                         delta2 = True, 
                         experiment = "Treatment")

paired_delta_01 = load(data = df_delta2_drug1,
                       paired = "baseline", id_col="ID",
                       x = ["Treatment", "Rep"], y = "Y", 
                       delta2 = True, experiment = "Genotype")

paired_delta_02 = load(data = df_delta2_drug2,
                       paired = "baseline", id_col="ID",
                       x = ["Treatment", "Rep"], y = "Y", 
                       delta2 = True, experiment = "Genotype")
paired_delta_03 = load(data = df_delta2_drug3,
                       paired = "baseline", id_col="ID",
                       x = ["Treatment", "Rep"], y = "Y", 
                       delta2 = True, experiment = "Genotype")

contrasts = [unpaired_delta_01, unpaired_delta_02, unpaired_delta_03]

paired_contrasts = [paired_delta_01, paired_delta_02, paired_delta_03]

# Customizable dataset creation with different arguments
df_mini_meta01 = create_mini_meta_dataset(seed=9999, 
                                          control_locs=[3, 3.5, 3.25], 
                                          control_scales=[0.4, 0.75, 0.4], 
                                          test_locs=[3.5, 2.5, 3], 
                                          test_scales=[0.5, 0.6, 0.75])

df_mini_meta02 = create_mini_meta_dataset(seed=9999, 
                                          control_locs=[4, 2, 3.25], 
                                          control_scales=[0.3, 0.75, 0.45], 
                                          test_locs=[2, 1.5, 2.75], 
                                          test_scales=[0.5, 0.6, 0.4])

df_mini_meta03 = create_mini_meta_dataset(seed=9999, 
                                          control_locs=[6, 5.5, 4.25], 
                                          control_scales=[0.4, 0.75, 0.45], 
                                          test_locs=[4.5, 3.5, 3], 
                                          test_scales=[0.5, 0.6, 0.9])

contrast_mini_meta01 = load(data = df_mini_meta01,
                            idx=(("Control 1", "Test 1"), ("Control 2", "Test 2"), ("Control 3", "Test 3")), 
                            mini_meta=True)

contrast_mini_meta02 = load(data = df_mini_meta02,
                            idx=(("Control 1", "Test 1"), ("Control 2", "Test 2"), ("Control 3", "Test 3")), 
                            mini_meta=True)

contrast_mini_meta03 = load(data = df_mini_meta03,
                            idx=(("Control 1", "Test 1"), ("Control 2", "Test 2"), ("Control 3", "Test 3")),
                            mini_meta=True)

contrasts_mini_meta = [contrast_mini_meta01, contrast_mini_meta02, contrast_mini_meta03]    


# Import your forest_plot function here
from dabest.forest_plot import forest_plot

@pytest.mark.mpl_image_compare(tolerance=10)
def test_201_forest_plot_no_colorpalette():
    return forest_plot(contrasts, 
                       contrast_labels=['Drug1', 'Drug2', 'Drug3'])

@pytest.mark.mpl_image_compare(tolerance=10)
def test_202_forest_plot_with_colorpalette():
    return forest_plot(contrasts, 
                       contrast_labels=['Drug1', 'Drug2', 'Drug3'],
                       custom_palette=['gray', 'blue', 'green']) 

@pytest.mark.mpl_image_compare(tolerance=10)
def test_203_horizontal_forest_plot_no_colorpalette():
    return forest_plot(contrasts, 
                       contrast_labels=['Drug1', 'Drug2', 'Drug3'],
                       horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=10)
def test_204_horizontal_forest_plot_with_colorpalette():
    return forest_plot(contrasts, 
                       contrast_labels=['Drug1', 'Drug2', 'Drug3'], 
                       custom_palette=['gray', 'blue', 'green'], 
                       horizontal=True)

@pytest.mark.mpl_image_compare(tolerance=10)
def test_206_forest_mini_meta():
    return forest_plot(contrasts_mini_meta, 
                       contrast_type='mini_meta', 
                       contrast_labels=['mini_meta1', 'mini_meta2', 'mini_meta3'])

@pytest.mark.mpl_image_compare(tolerance=10)
def test_205_forest_mini_meta_horizontal():
    return forest_plot(contrasts_mini_meta, 
                       contrast_type='mini_meta', 
                       contrast_labels=['mini_meta1', 'mini_meta2', 'mini_meta3'], 
                       horizontal=True)




