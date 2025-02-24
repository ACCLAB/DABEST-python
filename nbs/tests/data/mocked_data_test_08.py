import pandas as pd

# Data for tests.
# See Oehlert, G. W. (2000). A First Course in Design
# and Analysis of Experiments (1st ed.). W. H. Freeman.
# from Problem 16.3 Pg 444.

rep1_yes = [53.4, 54.3, 55.9, 53.8, 56.3, 58.6]
rep1_no = [58.2, 60.4, 62.4, 59.5, 64.5, 64.5]
rep2_yes = [46.5, 57.2, 57.4, 51.1, 56.9, 60.2]
rep2_no = [49.2, 61.6, 57.2, 51.3, 66.8, 62.7]
df_mini_meta = pd.DataFrame(
    {"Rep1_Yes": rep1_yes, "Rep1_No": rep1_no, "Rep2_Yes": rep2_yes, "Rep2_No": rep2_no}
)
N = 6  # Size of each group

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
    paired=None,
    id_col=None,
    ps_adjust=False,
)
