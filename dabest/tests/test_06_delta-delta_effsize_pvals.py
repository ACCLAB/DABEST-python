import sys
import pytest
import lqrt
import numpy as np
import scipy as sp
import pandas as pd
from .._stats_tools import effsize
from .._classes import TwoGroupsEffectSize, PermutationTest, Dabest


# Data for tests.
# See: Asheber Abebe. Introduction to Design and Analysis of Experiments 
# with the SAS, from Example: Two-way RM Design Pg 137.
hr = [72, 78, 71, 72, 66, 74, 62, 69, 69, 66, 84, 80, 72, 65, 75, 71, 
      86, 83, 82, 83, 79, 83, 73, 75, 73, 62, 90, 81, 72, 62, 69, 70]

# Add experiment column
e1 = np.repeat('Treatment1', 8).tolist()
e2 = np.repeat('Control', 8).tolist()
experiment = e1 + e2 + e1 + e2

# Add a `Drug` column as the first variable
d1 = np.repeat('AX23', 8).tolist()
d2 = np.repeat('CONTROL', 8).tolist()
drug = d1 + d2 + d1 + d2

# Add a `Time` column as the second variable
t1 = np.repeat('T1', 16).tolist()
t2 = np.repeat('T2', 16).tolist()
time = t1 + t2

# Add an `id` column for paired data plotting.
id_col = []
for i in range(1, 9):
    id_col.append(str(i)+"a")
for i in range(1, 9):
    id_col.append(str(i)+"c")
id_col.extend(id_col)

# Combine samples and gender into a DataFrame.
df_test = pd.DataFrame({'ID'   : id_col,
                   'Drug'      : drug,
                   'Time'      : time, 
                   'Experiment': experiment,
                   'Heart Rate': hr
                    })


df_test_control = df_test[df_test["Experiment"]=="Control"]
df_test_control = df_test_control.pivot(index="ID", columns="Time", values="Heart Rate")


df_test_treatment1 = df_test[df_test["Experiment"]=="Treatment1"]
df_test_treatment1 = df_test_treatment1.pivot(index="ID", columns="Time", values="Heart Rate")


# kwargs for Dabest class init.
dabest_default_kwargs = dict(ci=95, 
                            resamples=5000, random_seed=12345,
                            idx=None, proportional=False, mini_meta=False
                            )

# example of unpaired delta-delta calculation
unpaired = Dabest(data = df_test, x = ["Time", "Drug"], y = "Heart Rate", 
                  delta2 = True, experiment = "Experiment",
                  experiment_label=None, x1_level=None, paired=None, id_col=None,
                  **dabest_default_kwargs)


# example of paired delta-delta calculation
paired = Dabest(data = df_test, x = ["Time", "Drug"], y = "Heart Rate", 
                  delta2 = True, experiment = "Experiment", paired="sequential", id_col="ID",
                  experiment_label=None, x1_level=None,
                  **dabest_default_kwargs)


# example of paired data with specified experiment/x1 level
paired_specified_level = Dabest(data = df_test, x = ["Time", "Drug"], y = "Heart Rate", 
                  delta2 = True, experiment = "Experiment", paired="sequential", id_col="ID",
                  experiment_label=["Control", "Treatment1"], x1_level=["T2", "T1"],
                  **dabest_default_kwargs)


def test_mean_diff_delta_unpaired():
    mean_diff_results = unpaired.mean_diff.results
    all_mean_diff = mean_diff_results['difference'].to_list()
    diff1 = np.mean(df_test_treatment1["T2"])-np.mean(df_test_treatment1["T1"])
    diff2 = np.mean(df_test_control["T2"])-np.mean(df_test_control["T1"])
    np_result = [diff1, diff2]
    assert all_mean_diff == pytest.approx(np_result)


def test_mean_diff_delta_paired():
    mean_diff_results = paired.mean_diff.results
    all_mean_diff = mean_diff_results['difference'].to_list()
    diff1 = np.mean(df_test_treatment1["T2"]-df_test_treatment1["T1"])
    diff2 = np.mean(df_test_control["T2"]-df_test_control["T1"])
    np_result = [diff1, diff2]
    assert all_mean_diff == pytest.approx(np_result)


def test_mean_diff_delta_paired_specified_level():
    mean_diff_results = paired_specified_level.mean_diff.results
    all_mean_diff = mean_diff_results['difference'].to_list()
    diff1 = np.mean(df_test_control["T1"]-df_test_control["T2"])
    diff2 = np.mean(df_test_treatment1["T1"]-df_test_treatment1["T2"])
    np_result = [diff1, diff2]
    assert all_mean_diff == pytest.approx(np_result)


def test_median_diff_unpaired():
    all_median_diff = unpaired.median_diff.results
    median_diff = all_median_diff['difference'].to_list()
    diff1 = np.median(df_test_treatment1["T2"])-np.median(df_test_treatment1["T1"])
    diff2 = np.median(df_test_control["T2"])-np.median(df_test_control["T1"])
    np_result = [diff1, diff2]
    assert median_diff == pytest.approx(np_result)


def test_median_diff_paired():
    all_median_diff = paired.median_diff.results
    median_diff = all_median_diff['difference'].to_list()
    diff1 = np.median(df_test_treatment1["T2"]-df_test_treatment1["T1"])
    diff2 = np.median(df_test_control["T2"]-df_test_control["T1"])
    np_result = [diff1, diff2]
    assert median_diff == pytest.approx(np_result)


def test_median_diff_paired_specified_level():
    all_median_diff = paired_specified_level.median_diff.results
    median_diff = all_median_diff['difference'].to_list()
    diff1 = np.median(df_test_control["T1"]-df_test_control["T2"])
    diff2 = np.median(df_test_treatment1["T1"]-df_test_treatment1["T2"])
    np_result = [diff1, diff2]
    assert median_diff == pytest.approx(np_result)


def test_cohens_d_unpaired():
    all_cohens_d = unpaired.cohens_d.results
    cohens_d = all_cohens_d['difference'].to_list()
    diff1 = np.mean(df_test_treatment1["T2"])-np.mean(df_test_treatment1["T1"])
    diff1 = diff1/np.sqrt((np.var(df_test_treatment1["T2"], ddof=1)+np.var(df_test_treatment1["T1"], ddof=1))/2) 
    diff2 = np.mean(df_test_control["T2"])-np.mean(df_test_control["T1"])
    diff2 = diff2/np.sqrt((np.var(df_test_control["T2"], ddof=1)+np.var(df_test_control["T1"], ddof=1))/2) 
    np_result = [diff1, diff2]	        
    assert cohens_d == pytest.approx(np_result)


def test_cohens_d_paired():
    all_cohens_d = paired.cohens_d.results
    cohens_d = all_cohens_d['difference'].to_list()
    diff1 = np.mean(df_test_treatment1["T2"]-df_test_treatment1["T1"])
    diff1 = diff1/np.sqrt((np.var(df_test_treatment1["T2"], ddof=1)+np.var(df_test_treatment1["T1"], ddof=1))/2) 
    diff2 = np.mean(df_test_control["T2"]-df_test_control["T1"])
    diff2 = diff2/np.sqrt((np.var(df_test_control["T2"], ddof=1)+np.var(df_test_control["T1"], ddof=1))/2) 
    np_result = [diff1, diff2]	        
    assert cohens_d == pytest.approx(np_result)


def test_cohens_d_paired_specified_level():
    all_cohens_d = paired_specified_level.cohens_d.results
    cohens_d = all_cohens_d['difference'].to_list()
    diff1 = np.mean(df_test_control["T1"])-np.mean(df_test_control["T2"])
    diff1 = diff1/np.sqrt((np.var(df_test_control["T2"], ddof=1)+np.var(df_test_control["T1"], ddof=1))/2)
    diff2 = np.mean(df_test_treatment1["T1"])-np.mean(df_test_treatment1["T2"])
    diff2 = diff2/np.sqrt((np.var(df_test_treatment1["T2"], ddof=1)+np.var(df_test_treatment1["T1"], ddof=1))/2)  
    np_result = [diff1, diff2]	        
    assert cohens_d == pytest.approx(np_result)


def test_hedges_g_unpaired():
    from math import gamma
    hedges_g = unpaired.hedges_g.results['difference'].to_list()
    a = 8*2-2
    fac = gamma(a/2)/(np.sqrt(a/2)*gamma((a-1)/2))
    diff1 = (np.mean(df_test_treatment1["T2"])-np.mean(df_test_treatment1["T1"]))*fac
    diff1 = diff1/np.sqrt((np.var(df_test_treatment1["T2"], ddof=1)+np.var(df_test_treatment1["T1"], ddof=1))/2) 
    diff2 = (np.mean(df_test_control["T2"])-np.mean(df_test_control["T1"]))*fac
    diff2 = diff2/np.sqrt((np.var(df_test_control["T2"], ddof=1)+np.var(df_test_control["T1"], ddof=1))/2) 
    np_result=[diff1, diff2]
    assert hedges_g == pytest.approx(np_result)


def test_hedges_g_paired():
    from math import gamma
    hedges_g = paired.hedges_g.results['difference'].to_list()
    a = 8*2-2
    fac = gamma(a/2)/(np.sqrt(a/2)*gamma((a-1)/2))
    diff1 = (np.mean(df_test_treatment1["T2"]-df_test_treatment1["T1"]))*fac
    diff1 = diff1/np.sqrt((np.var(df_test_treatment1["T2"], ddof=1)+np.var(df_test_treatment1["T1"], ddof=1))/2) 
    diff2 = (np.mean(df_test_control["T2"]-df_test_control["T1"]))*fac
    diff2 = diff2/np.sqrt((np.var(df_test_control["T2"], ddof=1)+np.var(df_test_control["T1"], ddof=1))/2) 
    np_result=[diff1, diff2]
    assert hedges_g == pytest.approx(np_result)


def test_hedges_g_paired_specified_level():
    from math import gamma
    hedges_g = paired_specified_level.hedges_g.results['difference'].to_list()
    a = 8*2-2
    fac = gamma(a/2)/(np.sqrt(a/2)*gamma((a-1)/2))
    diff1 = (np.mean(df_test_control["T1"]-df_test_control["T2"]))*fac
    diff1 = diff1/np.sqrt((np.var(df_test_control["T2"], ddof=1)+np.var(df_test_control["T1"], ddof=1))/2) 
    diff2 = (np.mean(df_test_treatment1["T1"]-df_test_treatment1["T2"]))*fac
    diff2 = diff2/np.sqrt((np.var(df_test_treatment1["T2"], ddof=1)+np.var(df_test_treatment1["T1"], ddof=1))/2) 
    np_result=[diff1, diff2]
    assert hedges_g == pytest.approx(np_result)


def test_unpaired_delta_delta():
    delta_delta = unpaired.mean_diff.delta_delta.difference

    diff1 = np.mean(df_test_treatment1["T2"])-np.mean(df_test_treatment1["T1"])
    diff2 = np.mean(df_test_control["T2"])-np.mean(df_test_control["T1"])
    np_result = diff2-diff1

    assert delta_delta == pytest.approx(np_result)


def test_paired_delta_delta():
    delta_delta = paired.mean_diff.delta_delta.difference

    diff1 = np.mean(df_test_treatment1["T2"] - df_test_treatment1["T1"])
    diff2 = np.mean(df_test_control["T2"] - df_test_control["T1"])
    np_result = diff2-diff1

    assert delta_delta == pytest.approx(np_result)


def test_paired_specified_level_delta_delta():
    delta_delta = paired_specified_level.mean_diff.delta_delta.difference

    diff1 = np.mean(df_test_control["T1"] - df_test_control["T2"])
    diff2 = np.mean(df_test_treatment1["T1"] - df_test_treatment1["T2"])
    np_result = diff2-diff1

    assert delta_delta == pytest.approx(np_result)


def test_unpaired_permutation_test():
    delta_delta              = unpaired.mean_diff.delta_delta
    pvalue                   = delta_delta.pvalue_permutation
    permutations_delta_delta = delta_delta.permutations_delta_delta

    perm_test_1 = PermutationTest(df_test_treatment1["T1"], 
                                  df_test_treatment1["T2"], 
                                  effect_size="mean_diff", 
                                  is_paired=False)
    perm_test_2 = PermutationTest(df_test_control["T1"], 
                                  df_test_control["T2"], 
                                  effect_size="mean_diff", 
                                  is_paired=False)
    permutations_1 = perm_test_1.permutations
    permutations_2 = perm_test_2.permutations
    
    delta_deltas = permutations_2-permutations_1
    assert permutations_delta_delta == pytest.approx(delta_deltas)

    diff1 = np.mean(df_test_treatment1["T2"])-np.mean(df_test_treatment1["T1"])
    diff2 = np.mean(df_test_control["T2"])-np.mean(df_test_control["T1"])
    np_diff = diff2-diff1

    np_pvalues = len(list(filter(lambda x: np.abs(x)>np.abs(np_diff), 
                                delta_deltas)))/len(delta_deltas)

    assert pvalue == pytest.approx(np_pvalues)


def test_paired_permutation_test():
    delta_delta              = paired.mean_diff.delta_delta
    pvalue                   = delta_delta.pvalue_permutation
    permutations_delta_delta = delta_delta.permutations_delta_delta

    perm_test_1 = PermutationTest(df_test_treatment1["T1"], 
                                  df_test_treatment1["T2"], 
                                  effect_size="mean_diff", 
                                  is_paired="sequential")
    perm_test_2 = PermutationTest(df_test_control["T1"], 
                                  df_test_control["T2"], 
                                  effect_size="mean_diff", 
                                  is_paired="sequential")
    permutations_1 = perm_test_1.permutations
    permutations_2 = perm_test_2.permutations
    
    delta_deltas = permutations_2-permutations_1
    assert permutations_delta_delta == pytest.approx(delta_deltas)

    diff1 = np.mean(df_test_treatment1["T2"]-df_test_treatment1["T1"])
    diff2 = np.mean(df_test_control["T2"]-df_test_control["T1"])
    np_diff = diff2-diff1

    np_pvalues = len(list(filter(lambda x: np.abs(x)>np.abs(np_diff), 
                                delta_deltas)))/len(delta_deltas)

    assert pvalue == pytest.approx(np_pvalues)


def test_paired_specified_level_permutation_test():
    delta_delta              = paired_specified_level.mean_diff.delta_delta
    pvalue                   = delta_delta.pvalue_permutation
    permutations_delta_delta = delta_delta.permutations_delta_delta

    perm_test_1 = PermutationTest(df_test_control["T2"], 
                                  df_test_control["T1"], 
                                  effect_size="mean_diff", 
                                  is_paired="sequential")
    perm_test_2 = PermutationTest(df_test_treatment1["T2"], 
                                  df_test_treatment1["T1"], 
                                  effect_size="mean_diff", 
                                  is_paired="sequential")
    permutations_1 = perm_test_1.permutations
    permutations_2 = perm_test_2.permutations
    
    delta_deltas = permutations_2-permutations_1
    assert permutations_delta_delta == pytest.approx(delta_deltas)

    diff1 = np.mean(df_test_control["T1"]-df_test_control["T2"])
    diff2 = np.mean(df_test_treatment1["T1"]-df_test_treatment1["T2"])
    np_diff = diff2-diff1

    np_pvalues = len(list(filter(lambda x: np.abs(x)>np.abs(np_diff), 
                                delta_deltas)))/len(delta_deltas)

    assert pvalue == pytest.approx(np_pvalues)