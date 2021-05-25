import numpy as np
import pandas as pd
import dabest
import pylab

'''
#################### test on shared control
# Load the iris dataset. Requires internet access.
iris = pd.read_csv("https://github.com/mwaskom/seaborn-data/raw/master/iris.csv")

# Load the above data into `dabest`.
iris_dabest = dabest.load(data=iris, x="species", y="petal_width",
                          idx=("setosa", "versicolor", "virginica"))

# Produce a Cumming estimation plot.
iris_dabest.mean_diff.plot();
'''

from scipy.stats import norm
np.random.seed(9999) # Fix the seed so the results are replicable.
# pop_size = 10000 # Size of each population.
Ns = 20 # The number of samples taken from each population

# Create samples
d0 = norm.rvs(loc=3, scale=0.4, size=Ns)
d1 = norm.rvs(loc=3.5, scale=0.75, size=Ns)
d2 = norm.rvs(loc=3.25, scale=0.4, size=Ns)
d3 = norm.rvs(loc=3.5, scale=0.5, size=Ns)
d4 = norm.rvs(loc=2.5, scale=0.6, size=Ns)
d5 = norm.rvs(loc=3, scale=0.75, size=Ns)
d6 = norm.rvs(loc=3.5, scale=0.75, size=Ns)
d7 = norm.rvs(loc=3.25, scale=0.4, size=Ns)
d8 = norm.rvs(loc=3.25, scale=0.4, size=Ns)

# Add a `gender` column for coloring the data.
females = np.repeat('Female', Ns/2).tolist()
males = np.repeat('Male', Ns/2).tolist()
gender = females + males

# Add an `id` column for paired data plotting.
id_col = pd.Series(range(1, Ns+1))

# Combine samples and gender into a DataFrame.
df = pd.DataFrame({'Day0' : d0,     'Day1' : d1,
                     'Day2' : d2,     'Day3' : d3,
                     'Day4' : d4,     'Day5' : d5,
                     'Day6' : d6,     'Day7' : d7,
                    'Day8' : d8,
                     'Gender': gender, 'ID'  : id_col
                    })

#################### test on paired example (paired = True)
paired = dabest.load(df, id_col = "ID", idx=(("Day0", "Day1"),
                                       ("Day2", "Day3"),("Day4","Day5")), paired = True)

print(paired.mean_diff)
paired.mean_diff.plot(color_col="Gender")
pylab.show()

print(paired.median_diff)
paired.median_diff.plot(color_col="Gender")
pylab.show()

print(paired.cohens_d)
paired.cohens_d.plot(color_col="Gender")
pylab.show()

print(paired.hedges_g)
paired.hedges_g.plot(color_col="Gender")
pylab.show()

print(paired.cliffs_delta)
#baseline.cliffs_delta.plot(color_col="Gender")
#pylab.show()



#################### test on repeated_measure = baseline example
baseline = dabest.load(df, id_col = "ID", idx=(("Day0", "Day1"),
                                       ("Day2", "Day3","Day4"),
                                       ("Day5", "Day6","Day7", "Day8")), repeated_measures = "baseline"
                                     )

print(baseline.mean_diff)
baseline.mean_diff.plot(color_col="Gender")
pylab.show()

print(baseline.median_diff)
baseline.median_diff.plot(color_col="Gender")
pylab.show()

print(baseline.cohens_d)
baseline.cohens_d.plot(color_col="Gender")
pylab.show()

print(baseline.hedges_g)
baseline.hedges_g.plot(color_col="Gender")
pylab.show()

print(baseline.cliffs_delta)
#baseline.cliffs_delta.plot(color_col="Gender")
#pylab.show()

#####################repeated_measure = sequential example
sequential = dabest.load(df, id_col = "ID", idx=(("Day0", "Day1"),
                                       ("Day2", "Day3","Day4"),
                                       ("Day5", "Day6","Day7", "Day8")), repeated_measures = "sequential"
                                     )
print(sequential.mean_diff)
sequential.mean_diff.plot(color_col="Gender")
sequential.mean_diff.plot(color_col="Gender", show_pairs=False)
pylab.show()

print(sequential.median_diff)
sequential.median_diff.plot(color_col="Gender")
sequential.median_diff.plot(color_col="Gender", show_pairs=False)
pylab.show()

print(sequential.cohens_d)
sequential.cohens_d.plot(color_col="Gender")
sequential.cohens_d.plot(color_col="Gender", show_pairs=False)
pylab.show()

print(sequential.hedges_g)
sequential.hedges_g.plot(color_col="Gender")
sequential.hedges_g.plot(color_col="Gender", show_pairs=False)
pylab.show()

print(sequential.cliffs_delta)
#sequential.cliffs_delta.plot(color_col="Gender")
#sequential.cliffs_delta.plot(color_col="Gender", show_pairs=False)
#pylab.show()

