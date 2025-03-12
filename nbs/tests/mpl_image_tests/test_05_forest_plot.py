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


delta1 = load(data = df_mini_meta01,
                                   idx=(("Control 1", "Test 1"), ("Control 2", "Test 2"), ("Control 3", "Test 3")))
delta2 = load(data = df_mini_meta02,
                                    idx=(("Control 1", "Test 1"), ("Control 2", "Test 2"), ("Control 3", "Test 3")))
delta3 = load(data = df_mini_meta03,
                                   idx=(("Control 1", "Test 1"), ("Control 2", "Test 2"), ("Control 3", "Test 3")))
contrasts_deltas = [delta1, delta2, delta3]  

# Import your forest_plot function here
from dabest.forest_plot import forest_plot

@pytest.mark.mpl_image_compare(tolerance=8)
def test_500_deltadelta_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts, 
                labels=['Drug1', 'Drug2', 'Drug3']
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_501_deltadelta_with_deltas_idx_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts, 
                labels=['Drug1 Delta', 'Drug1 Delta-Delta', 
                        'Drug2 Delta', 'Drug2 Delta-Delta',
                        'Drug3 Delta', 'Drug3 Delta-Delta'
                        ],
                idx = [(0, 2), (0, 2), (0, 2)]
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_502_minimeta_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts_mini_meta, 
                labels=['mini_meta1', 'mini_meta2', 'mini_meta3']
            )



@pytest.mark.mpl_image_compare(tolerance=8)
def test_503_deltadelta_custompalette_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts, 
                labels=['Drug1', 'Drug2', 'Drug3'],
                custom_palette=['gray', 'blue', 'green']
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_504_deltadelta_horizontal_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts, 
                labels=['Drug1', 'Drug2', 'Drug3'],
                horizontal=True
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_505_deltadelta_insert_ax_forest():
    plt.rcdefaults()
    f_forest_drug_profiles, axes  = plt.subplots(2, 2, figsize=[15, 14])
    f_forest_drug_profiles.subplots_adjust(hspace=0.3, wspace=0.3)

    for ax, contrast in zip(axes.flatten(), [unpaired_delta_01, unpaired_delta_02, unpaired_delta_03]):
        contrast.mean_diff.plot(                  
                        contrast_label='Mean Diff',
                        raw_marker_size = 1,
                        contrast_marker_size = 5,
                        color_col='Genotype',
                        ax = ax
        )
        forest_plot(
                data = contrasts, 
                labels = ['Drug1', 'Drug2', 'Drug3'], 
                ax = axes[1,1], 
                )
    
    for ax, title in zip(axes.flatten(), ['Drug 1', 'Drug 2', 'Drug 3', 'Forest plot']):
        ax.set_title(title, fontsize = 12)

    return f_forest_drug_profiles


@pytest.mark.mpl_image_compare(tolerance=8)
def test_506a_deltadelta_delta_g_using_hedges_g_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts, 
                labels=['Drug1', 'Drug2', 'Drug3'],
                effect_size='hedges_g'
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_506b_deltadelta_delta_g_using_delta_g_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts, 
                labels=['Drug1', 'Drug2', 'Drug3'],
                effect_size='delta_g'
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_507_deltadelta_fig_size_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts, 
                labels=['Drug1', 'Drug2', 'Drug3'],
                fig_size=[6, 6]
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_508_deltadelta_fig_size_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts, 
                labels=['Drug1', 'Drug2', 'Drug3'],
                fig_size=[6, 6]
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_509_deltadelta_halfviolin_aesthetics_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts, 
                labels=['Drug1', 'Drug2', 'Drug3'],
                contrast_alpha=0.2,
                contrast_desat=0.2
            )


@pytest.mark.mpl_image_compare(tolerance=8)
def test_510_deltadelta_labels_and_title_aesthetics_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts, 
                labels=['Drug1', 'Drug2', 'Drug3'],
                labels_fontsize=12,
                labels_rotation=0,
                ylabel='Effect Size',
                ylabel_fontsize=14,
                title='Drug Efficacy',
                title_fontsize=20
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_511_deltadelta_lims_and_ticks_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts, 
                labels=['Drug1', 'Drug2', 'Drug3'],
                ylim=[-1, 1],
                yticks=[-1, 0, 1],
                yticklabels=['Negative', 'Zero', 'Positive']
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_512_deltadelta_spines_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts, 
                labels=['Drug1', 'Drug2', 'Drug3'],
                remove_spines=False
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_513_deltadelta_violinkwargs_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts, 
                labels=['Drug1', 'Drug2', 'Drug3'],
                violin_kwargs={
                    "widths": 0.8, "showextrema": True, 
                    "showmedians": True, "orientation": 'vertical'
                }
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_514_deltadelta_zerolinekwargs_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts, 
                labels=['Drug1', 'Drug2', 'Drug3'],
                zeroline_kwargs={"linewidth": 2, "color": "red"}
            )   

@pytest.mark.mpl_image_compare(tolerance=8)
def test_515_deltadelta_esmarkerkwargs_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts, 
                labels=['Drug1', 'Drug2', 'Drug3'],
                marker_kwargs={
                    'marker': '^', 'markersize': 15,'color': 'blue',
                    'alpha': 0.5,
                    }
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_516_deltadelta_eserrorbarkwargs_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts, 
                labels=['Drug1', 'Drug2', 'Drug3'],
                errorbar_kwargs={
                    'color': 'red', 'lw': 4, 'linestyle': '--', 'alpha': 0.6,
                }
            )


@pytest.mark.mpl_image_compare(tolerance=8)
def test_517_regular_delta_no_idx():
    plt.rcdefaults()
    return forest_plot(
                contrasts_deltas,
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_518_regular_delta_idx():
    plt.rcdefaults()
    return forest_plot(
                contrasts_deltas,
                idx = [(0,), (0,), (0,)],
                labels=['Drug1 \nTest 1 - Control 1', 'Drug2 \nTest 2 - Control 2', 'Drug3 \nTest 3 - Control 3']
            )



@pytest.mark.mpl_image_compare(tolerance=8)
def test_519_minimeta_with_deltas_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts_mini_meta, 
                idx=[(0, 3),(0, 3),(0, 3)],
                labels=['Contrast A1', 'Mini_Meta A', 'Contrast B1', 'Mini_Meta B', 'Contrast C1', 'Mini_Meta C']
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_520_minimeta_with_deltas_and_delta_text_kwargs_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts_mini_meta, 
                idx=[(0, 3),(0, 3),(0, 3)],
                labels=['Contrast A1', 'Mini_Meta A', 'Contrast B1', 'Mini_Meta B', 'Contrast C1', 'Mini_Meta C'],
                delta_text_kwargs={'color': 'black','fontsize': 8, 'rotation': 45, 'va': 'bottom',
                                   'x_coordinates': [1.4, 2.4, 3.4, 4.4, 5.4, 6.4], 
                                   'y_coordinates': [0.6, 0.1, -2, -1.5, -1.5, -1.5]}
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_521_minimeta_with_deltas_with_contrast_bars_kwargs_forest():
    plt.rcdefaults()
    return forest_plot(
                contrasts_mini_meta, 
                idx=[(0, 3),(0, 3),(0, 3)],
                labels=['Contrast A1', 'Mini_Meta A', 'Contrast B1', 'Mini_Meta B', 'Contrast C1', 'Mini_Meta C'],
                contrast_bars_kwargs={'color': 'red', 'alpha': 0.4}
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_522a_minimeta_with_deltas_with_summary_bars():
    plt.rcdefaults()
    return forest_plot(
                contrasts_mini_meta, 
                idx=[(0, 3),(0, 3),(0, 3)],
                labels=['Contrast A1', 'Mini_Meta A', 'Contrast B1', 'Mini_Meta B', 'Contrast C1', 'Mini_Meta C'],
                summary_bars=[0, 2],
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_522b_minimeta_with_deltas_with_summary_bars_horizontal():
    plt.rcdefaults()
    return forest_plot(
                contrasts_mini_meta, 
                idx=[(0, 3),(0, 3),(0, 3)],
                labels=['Contrast A1', 'Mini_Meta A', 'Contrast B1', 'Mini_Meta B', 'Contrast C1', 'Mini_Meta C'],
                summary_bars=[0, 2],
                horizontal=True
            )


@pytest.mark.mpl_image_compare(tolerance=8)
def test_522c_minimeta_with_deltas_with_summary_bars_kwargs():
    plt.rcdefaults()
    return forest_plot(
                contrasts_mini_meta, 
                idx=[(0, 3),(0, 3),(0, 3)],
                labels=['Contrast A1', 'Mini_Meta A', 'Contrast B1', 'Mini_Meta B', 'Contrast C1', 'Mini_Meta C'],
                summary_bars=[0, 2],
                summary_bars_kwargs={'span_ax': True, 'color': 'grey', 'alpha': 0.1}
            )

@pytest.mark.mpl_image_compare(tolerance=8)
def test_522d_minimeta_with_deltas_with_summary_bars_kwargs_horizontal():
    plt.rcdefaults()
    return forest_plot(
                contrasts_mini_meta, 
                idx=[(0, 3),(0, 3),(0, 3)],
                labels=['Contrast A1', 'Mini_Meta A', 'Contrast B1', 'Mini_Meta B', 'Contrast C1', 'Mini_Meta C'],
                summary_bars=[0, 2],
                horizontal=True,
                summary_bars_kwargs={'span_ax': True, 'color': 'grey', 'alpha': 0.1}
            )