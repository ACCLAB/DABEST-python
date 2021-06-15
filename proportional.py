from scipy.stats import norm, bernoulli
import pandas as pd
import dabest
import pylab

control = bernoulli.rvs(0.3, loc=0, size=1000, random_state=12345)
test    = bernoulli.rvs(0.4, loc=0, size=1000, random_state=12345)
test2    = bernoulli.rvs(0.5, loc=0, size=1000, random_state=12345)

my_df   = pd.DataFrame({"control": control,
                            "test": test,
                        "t2":test2})
my_dabest_object = dabest.load(my_df, idx=("control", "test", "t2"), proportional=True)
my_dabest_object.mean_diff.plot()
pylab.show()
