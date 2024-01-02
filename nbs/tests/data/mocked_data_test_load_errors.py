import pandas as pd
import scipy as sp
from numpy import random

random.seed(88888)
N = 10
c1 = sp.stats.norm.rvs(loc=100, scale=5, size=N)
t1 = sp.stats.norm.rvs(loc=115, scale=5, size=N)
dummy_df = pd.DataFrame({"Control 1": c1, "Test 1": t1})
