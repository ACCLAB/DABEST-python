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
                            idx=None, proportional=False
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
    diff3 = np.mean(mean_diff_results['bootstraps'][1]) - np.mean(mean_diff_results['bootstraps'][0])
    np_result = [diff1, diff2, diff3]
    assert all_mean_diff == pytest.approx(np_result)


def test_mean_diff_delta_paired():
    mean_diff_results = paired.mean_diff.results
    all_mean_diff = mean_diff_results['difference'].to_list()
    diff1 = np.mean(df_test_treatment1["T2"]-df_test_treatment1["T1"])
    diff2 = np.mean(df_test_control["T2"]-df_test_control["T1"])
    diff3 = np.mean(mean_diff_results['bootstraps'][1] - mean_diff_results['bootstraps'][0])
    np_result = [diff1, diff2, diff3]
    assert all_mean_diff == pytest.approx(np_result)


def test_mean_diff_delta_paired_specified_level():
    mean_diff_results = paired_specified_level.mean_diff.results
    all_mean_diff = mean_diff_results['difference'].to_list()
    diff1 = np.mean(df_test_control["T1"]-df_test_control["T2"])
    diff2 = np.mean(df_test_treatment1["T1"]-df_test_treatment1["T2"])
    diff3 = np.mean(mean_diff_results['bootstraps'][1] - mean_diff_results['bootstraps'][0])
    np_result = [diff1, diff2, diff3]
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


def test_paired_stats_unpaired():
    np_result = unpaired.mean_diff.results
    
    p1 = sp.stats.ttest_rel(np_result["bootstraps"][0], \
                np_result["bootstraps"][1], nan_policy='omit').pvalue
    assert np_result["pvalue_paired_students_t"].to_list()[2] == pytest.approx(p1)
    
    p2 = sp.stats.wilcoxon(np_result["bootstraps"][0], \
                np_result["bootstraps"][1],).pvalue
    assert np_result["pvalue_wilcoxon"].to_list()[2] == pytest.approx(p2)


def test_paired_stats_paired():
    np_result = paired.mean_diff.results
    p1 = sp.stats.ttest_rel(df_test_treatment1["T1"], \
                df_test_treatment1["T2"], nan_policy='omit').pvalue
    p2 = sp.stats.ttest_rel(df_test_control["T1"], \
                df_test_control["T2"], nan_policy='omit').pvalue
    p3 = sp.stats.ttest_rel(np_result["bootstraps"][0], \
                np_result["bootstraps"][1], nan_policy='omit').pvalue
    assert np_result["pvalue_paired_students_t"].to_list() == pytest.approx([p1, p2, p3])
    
    p1 = sp.stats.wilcoxon(df_test_treatment1["T1"], \
                df_test_treatment1["T2"]).pvalue
    p2 = sp.stats.wilcoxon(df_test_control["T1"], \
                df_test_control["T2"]).pvalue
    p3 = sp.stats.wilcoxon(np_result["bootstraps"][0], \
                np_result["bootstraps"][1]).pvalue
    assert np_result["pvalue_wilcoxon"].to_list() == pytest.approx([p1, p2, p3])


def test_paired_stats_paired_speficied_level():
    np_result = paired_specified_level.mean_diff.results
    p1 = sp.stats.ttest_rel(df_test_control["T2"], \
                df_test_control["T1"], nan_policy='omit').pvalue
    p2 = sp.stats.ttest_rel(df_test_treatment1["T2"], \
                df_test_treatment1["T1"], nan_policy='omit').pvalue
    p3 = sp.stats.ttest_rel(np_result["bootstraps"][0], \
                np_result["bootstraps"][1], nan_policy='omit').pvalue
    assert np_result["pvalue_paired_students_t"].to_list() == pytest.approx([p1, p2, p3])
    
    p1 = sp.stats.wilcoxon(df_test_control["T2"], \
                df_test_control["T1"]).pvalue
    p2 = sp.stats.wilcoxon(df_test_treatment1["T2"], \
                df_test_treatment1["T1"]).pvalue
    p3 = sp.stats.wilcoxon(np_result["bootstraps"][0], \
                np_result["bootstraps"][1]).pvalue
    assert np_result["pvalue_wilcoxon"].to_list() == pytest.approx([p1, p2, p3])


def test_lqrt_delta_unpaired():
    all_mean_diff = unpaired.mean_diff
    lqrt_result = all_mean_diff.lqrt["pvalue_paired_lqrt"].to_list()[2]
                             
    p1 = lqrt.lqrtest_rel(all_mean_diff.results["bootstraps"][0], 
          all_mean_diff.results["bootstraps"][1], 
          random_state=12345).pvalue
    
    assert lqrt_result == pytest.approx(p1)


def test_lqrt_delta_paired():
    all_mean_diff = paired.mean_diff
    lqrt_result = all_mean_diff.lqrt["pvalue_paired_lqrt"].to_list()

    p1 = lqrt.lqrtest_rel(df_test_treatment1["T2"], 
                df_test_treatment1["T1"], 
          random_state=12345).pvalue
    p2 = lqrt.lqrtest_rel(df_test_control["T2"], 
                df_test_control["T1"], 
          random_state=12345).pvalue
    p3 = lqrt.lqrtest_rel(all_mean_diff.results["bootstraps"][0], 
          all_mean_diff.results["bootstraps"][1], 
          random_state=12345).pvalue
    
    assert lqrt_result == pytest.approx([p1, p2, p3])

def test_lqrt_delta_paired_specified_level():
    all_mean_diff = paired_specified_level.mean_diff
    lqrt_result = all_mean_diff.lqrt["pvalue_paired_lqrt"].to_list()

    p1 = lqrt.lqrtest_rel(df_test_control["T2"], 
                df_test_control["T1"], 
          random_state=12345).pvalue
    p2 = lqrt.lqrtest_rel(df_test_treatment1["T2"], 
                df_test_treatment1["T1"], 
          random_state=12345).pvalue
    p3 = lqrt.lqrtest_rel(all_mean_diff.results["bootstraps"][0], 
          all_mean_diff.results["bootstraps"][1], 
          random_state=12345).pvalue
    
    assert lqrt_result == pytest.approx([p1, p2, p3])

