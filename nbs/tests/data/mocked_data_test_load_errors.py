import pandas as pd
import scipy as sp
from numpy import random

random.seed(88888)
N = 10
c1 = sp.stats.norm.rvs(loc=100, scale=5, size=N)
c2 = sp.stats.norm.rvs(loc=115, scale=5, size=N)
c3 = sp.stats.norm.rvs(loc=3.25, scale=0.4, size=N)

t1 = sp.stats.norm.rvs(loc=3.5, scale=0.5, size=N)
t2 = sp.stats.norm.rvs(loc=2.5, scale=0.6, size=N)
id_col = pd.Series(range(1, N+1))
dummy_df = pd.DataFrame({'Control 1' : c1,     'Test 1' : t1,
                   'Control 2' : c2,     'Test 2' : t2,
                   'Control 3' : c3,     'ID'  : id_col
                  })
