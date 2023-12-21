import pandas as pd
import numpy as np


# Data for tests.
# See: Asheber Abebe. Introduction to Design and Analysis of Experiments
# with the SAS, from Example: Two-way RM Design Pg 137.
# to remove the array wrapping behaviour of black
# fmt: off
hr = [72, 78, 71, 72, 66, 74, 62, 69, 69, 66, 84, 80, 72, 65, 75, 71,
      86, 83, 82, 83, 79, 83, 73, 75, 73, 62, 90, 81, 72, 62, 69, 70]
# fmt: on

# Add experiment column
e1 = np.repeat("Treatment1", 8).tolist()
e2 = np.repeat("Control", 8).tolist()
experiment = e1 + e2 + e1 + e2

# Add a `Drug` column as the first variable
d1 = np.repeat("AX23", 8).tolist()
d2 = np.repeat("CONTROL", 8).tolist()
drug = d1 + d2 + d1 + d2

# Add a `Time` column as the second variable
t1 = np.repeat("T1", 16).tolist()
t2 = np.repeat("T2", 16).tolist()
time = t1 + t2

# Add an `id` column for paired data plotting.
id_col = []
for i in range(1, 9):
    id_col.append(str(i) + "a")
for i in range(1, 9):
    id_col.append(str(i) + "c")
id_col.extend(id_col)

# Combine samples and gender into a DataFrame.
df_test = pd.DataFrame(
    {
        "ID": id_col,
        "Drug": drug,
        "Time": time,
        "Experiment": experiment,
        "Heart Rate": hr,
    }
)


df_test_control = df_test[df_test["Experiment"] == "Control"]
df_test_control = df_test_control.pivot(index="ID", columns="Time", values="Heart Rate")


df_test_treatment1 = df_test[df_test["Experiment"] == "Treatment1"]
df_test_treatment1 = df_test_treatment1.pivot(
    index="ID", columns="Time", values="Heart Rate"
)

dabest_default_kwargs = dict(
    ci=95,
    resamples=5000,
    random_seed=12345,
    idx=None,
    proportional=False,
    mini_meta=False,
)
