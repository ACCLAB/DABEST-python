import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

# Dummy Pandas DataFrame used for swarmplots unit testing
random.seed(88888)
N = 10
c1 = sp.stats.norm.rvs(loc=100, scale=5, size=N)
t1 = sp.stats.norm.rvs(loc=115, scale=5, size=N)

females = np.repeat("Female", N / 2).tolist()
males = np.repeat("Male", N / 2).tolist()
gender = females + males

dummy_df = pd.DataFrame({"Control 1": c1, "Test 1": t1, "gender": gender})
dummy_df = pd.melt(
    dummy_df,
    id_vars=["gender"],
    value_vars=["Control 1", "Test 1"],
    var_name="group",
    value_name="value",
)

# Default swarmplot params for unit testing
default_swarmplot_kwargs = {
    "data": dummy_df,
    "x": "group",
    "y": "value",
    "ax": plt.gca(),
}
