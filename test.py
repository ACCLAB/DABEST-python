import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib as mpl
import os
from pathlib import Path

import matplotlib.ticker as Ticker
import matplotlib.pyplot as plt

from dabest._api import load

import dabest

columns = [1, 2.0]
columns_str = ["1", "2.0"]
# create a test database
N = 100
df = pd.DataFrame(np.vstack([np.random.normal(loc=i, size=(N,)) for i in range(len(columns))]).T, columns=columns_str)
females = np.repeat("Female", N / 2).tolist()
males = np.repeat("Male", N / 2).tolist()
df['gender'] = females + males

# Add an `id` column for paired data plotting.
df['ID'] = pd.Series(range(1, N + 1))


db = dabest.load(data=df, idx=columns_str, paired="baseline", id_col="ID")
print(db.mean_diff)
db.mean_diff.plot();

# def create_demo_dataset(seed=9999, N=20):
#     import numpy as np
#     import pandas as pd
#     from scipy.stats import norm  # Used in generation of populations.

#     np.random.seed(9999)  # Fix the seed so the results are replicable.
#     # pop_size = 10000 # Size of each population.

#     # Create samples
#     c1 = norm.rvs(loc=3, scale=0.4, size=N)
#     c2 = norm.rvs(loc=3.5, scale=0.75, size=N)
#     c3 = norm.rvs(loc=3.25, scale=0.4, size=N)

#     t1 = norm.rvs(loc=3.5, scale=0.5, size=N)
#     t2 = norm.rvs(loc=2.5, scale=0.6, size=N)
#     t3 = norm.rvs(loc=3, scale=0.75, size=N)
#     t4 = norm.rvs(loc=3.5, scale=0.75, size=N)
#     t5 = norm.rvs(loc=3.25, scale=0.4, size=N)
#     t6 = norm.rvs(loc=3.25, scale=0.4, size=N)

#     # Add a `gender` column for coloring the data.
#     females = np.repeat("Female", N / 2).tolist()
#     males = np.repeat("Male", N / 2).tolist()
#     gender = females + males

#     # Add an `id` column for paired data plotting.
#     id_col = pd.Series(range(1, N + 1))

#     # Combine samples and gender into a DataFrame.
#     df = pd.DataFrame(
#         {
#             "Control 1": c1,
#             "Test 1": t1,
#             "Control 2": c2,
#             "Test 2": t2,
#             "Control 3": c3,
#             "Test 3": t3,
#             "Test 4": t4,
#             "Test 5": t5,
#             "Test 6": t6,
#             "Gender": gender,
#             "ID": id_col,
#         }
#     )

#     return df


# df = create_demo_dataset()

# two_groups_unpaired = load(df, idx=("Control 1", "Test 1"))

# two_groups_paired = load(
#     df, idx=("Control 1", "Test 1"), paired="baseline", id_col="ID"
# )

# two_groups_unpaired.mean_diff.plot()
