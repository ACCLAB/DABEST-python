"""
Mocked data for testing multi.py module.

This module provides test fixtures and default parameters for testing
the MultiContrast class and its associated functions (combine, whorlmap).
"""

import pandas as pd
import numpy as np
import dabest

# Set seed for reproducibility
np.random.seed(9999)


def create_two_group_contrast():
    """Create a standard dabest contrast object."""
    N = 20
    y = np.random.normal(loc=3, scale=0.4, size=N*2)
    y[N:] += 1  # Treatment effect
    
    df = pd.DataFrame({
        'ID': list(range(N*2)),
        'Group': ['Control'] * N + ['Treatment'] * N,
        'Y': y
    })
    
    return dabest.load(data=df, x='Group', y='Y', idx=('Control', 'Treatment'))


def create_delta2_contrast():
    """Create a delta-delta (delta2) dabest contrast object."""
    N = 20
    y = np.random.normal(loc=3, scale=0.4, size=N*4)
    y[N:2*N] += 1
    y[2*N:3*N] -= 0.5
    
    treatment = np.repeat(['Placebo', 'Drug'], N*2).tolist()
    genotype = np.repeat(['W', 'M', 'W', 'M'], N).tolist()
    id_col = list(range(0, N*2)) * 2
    
    df = pd.DataFrame({
        'ID': id_col,
        'Genotype': genotype,
        'Treatment': treatment,
        'Y': y
    })
    
    return dabest.load(data=df, x=["Genotype", "Genotype"], y="Y", 
                      delta2=True, experiment="Treatment")


def create_minimeta_contrast():
    """Create a mini-meta analysis dabest contrast object."""
    N = 20
    y = np.random.normal(loc=3, scale=0.4, size=N*4)
    y[N:2*N] += 0.8
    y[2*N:3*N] += 1.2
    
    experiment = ['Exp1'] * (N*2) + ['Exp2'] * (N*2)
    group = (['Control'] * N + ['Treatment'] * N) * 2
    
    df = pd.DataFrame({
        'Experiment': experiment,
        'Group': group,
        'Y': y
    })
    
    return dabest.load(data=df, x='Group', y='Y', idx=('Control', 'Treatment'),
                      mini_meta=True, experiment='Experiment')



# Single contrast objects
two_group_contrast_1 = create_two_group_contrast()
two_group_contrast_2 = create_two_group_contrast()
two_group_contrast_3 = create_two_group_contrast()

delta2_contrast_1 = create_delta2_contrast()
delta2_contrast_2 = create_delta2_contrast()

minimeta_contrast_1 = create_minimeta_contrast()
minimeta_contrast_2 = create_minimeta_contrast()


default_combine_kwargs = {
    "dabest_objs": [two_group_contrast_1, two_group_contrast_2],
    "labels": ["Contrast 1", "Contrast 2"],
    "row_labels": None,
    "effect_size": "mean_diff",
    "ci_type": "bca",
    "allow_mixed_types": False
}


# Create a valid MultiContrast object for whorlmap testing
from dabest.multi import MultiContrast

valid_multi_contrast_1d = MultiContrast(
    dabest_objs=[two_group_contrast_1, two_group_contrast_2],
    labels=["Treatment A", "Treatment B"],
    effect_size="mean_diff",
    ci_type="bca"
)

valid_multi_contrast_2d = MultiContrast(
    dabest_objs=[[two_group_contrast_1, two_group_contrast_2],
                 [two_group_contrast_3, two_group_contrast_1]],
    labels=["Col1", "Col2"],
    row_labels=["Row1", "Row2"],
    effect_size="mean_diff",
    ci_type="bca"
)

valid_multi_contrast_mixed = MultiContrast(
    dabest_objs=[[two_group_contrast_1, two_group_contrast_2],
                 [delta2_contrast_1, delta2_contrast_2]],
    labels=["Col1", "Col2"],
    row_labels=["Standard", "Delta2"],
    effect_size="mean_diff",
    ci_type="bca"
)


default_whorlmap_kwargs = {
    "multi_contrast": valid_multi_contrast_1d,
    "n": 21,
    "sort_by": None,
    "cmap": "vlag",
    "vmax": None,
    "vmin": None,
    "reverse_neg": True,
    "abs_rank": False,
    "chop_tail": 0,
    "ax": None,
    "fig_size": None,
    "title": None,
    "heatmap_kwargs": None
}