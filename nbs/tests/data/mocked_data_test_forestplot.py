import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from scipy.stats import norm
import dabest 

np.random.seed(9999)  # Set the seed for reproducibility
N=20
# Create samples
y = norm.rvs(loc=3, scale=0.4, size=N*4)
y[N:2*N] += 1
y[2*N:3*N] -= 0.5

# Treatment, Rep, Genotype, and ID columns
treatment = np.repeat(['Placebo', 'Drug'], N*2).tolist()
rep = ['Rep1', 'Rep2'] * (N*2)
genotype = np.repeat(['W', 'M', 'W', 'M'], N).tolist()
id_col = list(range(0, N*2)) * 2

    # Combine all columns into a DataFrame
dummy_df = pd.DataFrame({
    'ID': id_col,
    'Rep': rep,
    'Genotype': genotype,
    'Treatment': treatment,
    'Y': y
})

unpaired_delta_01 = dabest.load(data = dummy_df, 
                                x = ["Genotype", "Genotype"], 
                                y = "Y", delta2 = True, 
                                experiment = "Treatment")

dummy_contrasts = [unpaired_delta_01]

# Default forestplot params for unit testing
default_forestplot_kwargs = {
    "data": dummy_contrasts,  # Ensure this is a list of contrast objects.
    "idx": None,  # Valid as None or a list of lists of integers.
    "effect_size": "mean_diff",  # Ensure it's a string.
    "labels": ["Drug1"],  # This should be a list of strings.
    "ylabel": "Effect Size",  # Ensure it's a string.
    "title": "ΔΔ Forest Plot",  # Ensure it's a string.
    "custom_palette": None,  # Valid as None, a dictionary, list, or string.
    "violin_kwargs": None,  # No specific checks needed based on your tests.
    "marker_size": 20,  # Ensure it's a positive integer or float.
    "remove_spines": True,  # Ensure it's a boolean.
    "labels_rotation": 45,  # Ensure it's an integer or float between 0 and 360.
    "halfviolin_alpha": 0.8,  # Ensure it's a float between 0 and 1.
    "horizontal": False,  # Ensure it's a boolean.
}

