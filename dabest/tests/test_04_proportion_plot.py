import pytest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .._api import load

df_proportion = pd.DataFrame({"Control 1":[0,0,0,1,1,0,1,0,0,0,0,1,1,1,1,0,0,1,1,0,0,1,1,1,0,0,0,0,0,1,1,1,
                                           0,1,1,0,1,0,1,0,0,1,1,1,0,1,0,1,1,0,1,1,1,0,1,0],
                              "Test 1":[1,0,0,1,1,0,1,0,0,0,0,1,1,1,1,0,0,1,1,0,0,1,1,1,0,0,1,0,0,1,1,1,
                                        0,1,1,0,1,0,1,0,0,1,1,1,0,1,0,1,1,0,1,1,1,0,1,0]})

proportion_groups = load(df_proportion, idx=("Control 1", "Test 1"))

def test_01_gardner_altman_unpaired_propdiff():
    fig = proportion_groups.mean_diff.plot()
    plt.show()
