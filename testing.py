import numpy as np
import pandas as pd
import dabest
import pylab

from scipy.stats import norm
np.random.seed(9999) # Fix the seed so the results are replicable.
# pop_size = 10000 # Size of each population.
Ns = 20 # The number of samples taken from each population

# Create samples
c1 = norm.rvs(loc=3, scale=0.4, size=Ns)
c2 = norm.rvs(loc=3.5, scale=0.75, size=Ns)
c3 = norm.rvs(loc=3.25, scale=0.4, size=Ns)

t1 = norm.rvs(loc=3.5, scale=0.5, size=Ns)
t2 = norm.rvs(loc=2.5, scale=0.6, size=Ns)
t3 = norm.rvs(loc=3, scale=0.75, size=Ns)
t4 = norm.rvs(loc=3.5, scale=0.75, size=Ns)
t5 = norm.rvs(loc=3.25, scale=0.4, size=Ns)
t6 = norm.rvs(loc=3.25, scale=0.4, size=Ns)

# Add a `gender` column for coloring the data.
females = np.repeat('Female', Ns/2).tolist()
males = np.repeat('Male', Ns/2).tolist()
gender = females + males

# Add an `id` column for paired data plotting.
id_col = pd.Series(range(1, Ns+1))

# Combine samples and gender into a DataFrame.
df = pd.DataFrame({'Day0' : c1,     'Day1' : t1,
                     'Day2' : c2,     'Day3' : t2,
                     'Day4' : c3,     'Day5' : t3,
                     'Day6'    : t4,     'Day7' : t5, 'Day8' : t6,
                     'Gender'    : gender, 'ID'  : id_col
                    })

shared_control = dabest.load(df, id_col = "ID", idx=(("Day0", "Day1"),
                                       ("Day2", "Day3","Day4"),
                                       ("Day5", "Day6","Day7", "Day8")
                                     ), repeated_measures = "baseline")

shared_control.mean_diff.plot(color_col="Gender");
pylab.show()

