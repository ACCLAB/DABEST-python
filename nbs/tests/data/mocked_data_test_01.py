import pandas as pd
import numpy as np

# Data for tests.
# See Cumming, G. Understanding the New Statistics:
# Effect Sizes, Confidence Intervals, and Meta-Analysis. Routledge, 2012,
# from Cumming 2012 Table 11.1 Pg 287.
wb = {
    "control": [34, 54, 33, 44, 45, 53, 37, 26, 38, 58],
    "expt": [66, 38, 35, 55, 48, 39, 65, 32, 57, 41],
}
wellbeing = pd.DataFrame(wb)


# from Cumming 2012 Table 11.2 Page 291
paired_wb = {
    "pre": [43, 28, 54, 36, 31, 48, 50, 69, 29, 40],
    "post": [51, 33, 58, 42, 39, 45, 54, 68, 35, 44],
    "ID": np.arange(10),
}
paired_wellbeing = pd.DataFrame(paired_wb)


# Data for testing Cohen's calculation.
# Only work with binary data.
# See Venables, W. N. and Ripley, B. D. (2002) Modern Applied Statistics with S. Fourth edition. Springer.
# Make two groups of `smoke` by choosing `low` as a standard, and the data is trimed from the back.

# to remove the array wrapping behaviour of black
# fmt: off
sk = {  "low":  [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,
                 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
        "high": [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0,
                 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1]}
# fmt: on
smoke = pd.DataFrame(sk)


# Data from Hogarty and Kromrey (1999)
# Kromrey, Jeffrey D., and Kristine Y. Hogarty. 1998.
# "Analysis Options for Testing Group Differences on Ordered Categorical
# Variables: An Empirical Investigation of Type I Error Control
# Statistical Power."
# Multiple Linear Regression Viewpoints 25 (1): 70 - 82.
likert_control = [1, 1, 2, 2, 2, 3, 3, 3, 4, 5]
likert_treatment = [1, 2, 3, 4, 4, 5]


# Data from Cliff (1993)
# Cliff, Norman. 1993. "Dominance Statistics: Ordinal Analyses to Answer
# Ordinal Questions."
# Psychological Bulletin 114 (3): 494-509.
a_scores = [6, 7, 9, 10]
b_scores = [1, 3, 4, 7, 8]


# kwargs for Dabest class init.
dabest_default_kwargs = dict(
    x=None,
    y=None,
    ci=95,
    resamples=5000,
    random_seed=12345,
    proportional=False,
    delta2=False,
    experiment=None,
    experiment_label=None,
    x1_level=None,
    mini_meta=False,
    ps_adjust=False,
)
