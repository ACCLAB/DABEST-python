import sys
import pytest
import lqrt
import numpy as np
import scipy as sp
import pandas as pd
from .._stats_tools import effsize
from .._classes import TwoGroupsEffectSize, PermutationTest, Dabest, EffectSizeDataFrame



# Data for tests
# See Der, G., &amp; Everitt, B. S. (2009). A handbook
# of statistical analyses using SAS, from Display 11.1
group = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
first = [20, 14, 7, 6, 9, 9, 7, 18, 6, 10, 5, 11, 10, 17, 16, 7, 5, 16, 2, 7, 9, 2, 7, 19,
         7, 9, 6, 13, 9, 6, 11, 7, 8, 3, 4, 11, 1, 6, 0, 18, 15, 10,  6,  9,  4,  4, 10]
second = [15, 12, 5, 10, 7, 9, 3, 17, 9, 15, 9, 11, 2, 12, 15, 10, 0, 7, 1, 11, 16,
        5, 3, 13, 5, 12, 7, 18, 10, 7, 11, 10, 18, 3, 10, 10, 3, 7, 3, 18, 15, 14, 6, 9, 3, 13, 11]
third = [14, 12, 5, 9, 9, 9, 7, 16, 9, 12, 7, 8, 9, 14, 12, 4, 5, 7, 1, 7, 14, 6, 5, 14, 8, 16, 10,
         14, 12, 8, 12, 11, 19, 3, 11, 10, 2, 7, 3, 19, 15, 16, 7, 13, 4, 13, 13]
fourth = [13, 10, 6, 8, 5, 11, 6, 14, 9, 12, 3, 8, 3, 10, 7, 7, 0, 6, 2, 5, 10, 7, 5, 12, 8, 17, 15,
          21, 14, 9, 14, 12, 19, 7, 17, 15, 4, 9, 4, 22, 18, 17, 9, 16, 7, 16, 17]
fifth = [13, 10, 5, 7, 4, 8, 5, 12, 9, 11, 5, 9, 5, 9, 9, 5, 0, 4, 2, 8, 6, 6, 5, 10, 6, 18, 16, 21,
         15, 12, 16, 14, 22, 8, 18, 16, 5, 10, 6, 22, 19, 19, 10, 20, 9, 19, 21] 

df = pd.DataFrame({'Group' : group,
                   'First' : first,
                   'Second': second,
                   'Third' : third,
                   'Fourth': fourth,
                   'Fifth' : fifth,
                   'ID': np.arange(0, 47)
                    })


# kwargs for Dabest class init.
dabest_default_kwargs = dict(x=None, y=None, ci=95, 
                            resamples=5000, random_seed=12345, proportional=False,
                            delta2 = False, experiment=None, 
                            experiment_label=None, x1_level=None, mini_meta=False)

# example of sequential repeated measures
sequential = Dabest(df, id_col = "ID",
                         idx=("First", "Second", "Third", "Fourth", "Fifth"),
                         paired = "sequential",
                         **dabest_default_kwargs)

# example of baseline repeated measures
baseline = Dabest(df, id_col = "ID",
                       idx=("First", "Second", "Third", "Fourth", "Fifth"),
                       paired = "baseline",
                       **dabest_default_kwargs)



def test_mean_diff_sequential():
    mean_diff = sequential.mean_diff.results['difference'].to_list()
    np_result = [np.mean(df.iloc[:,i+1]-df.iloc[:,i]) for i in range(1,5)]
    assert mean_diff == pytest.approx(np_result)



def test_median_diff_sequential():
    median_diff = sequential.median_diff.results['difference'].to_list()
    np_result = [np.median(df.iloc[:,i+1]-df.iloc[:,i]) for i in range(1,5)]
    assert median_diff == pytest.approx(np_result)



def test_mean_diff_baseline():
    mean_diff = baseline.mean_diff.results['difference'].to_list()
    np_result = [np.mean(df.iloc[:,i]-df.iloc[:,1]) for i in range(2,6)]
    assert mean_diff == pytest.approx(np_result)



def test_median_diff_baseline():
    median_diff = baseline.median_diff.results['difference'].to_list()
    np_result = [np.median(df.iloc[:,i]-df.iloc[:,1]) for i in range(2,6)]
    assert median_diff == pytest.approx(np_result)



def test_cohens_d_sequential():
    cohens_d = sequential.cohens_d.results['difference'].to_list()
    np_result = [np.mean(df.iloc[:,i+1]-df.iloc[:,i])
		            /np.sqrt((np.var(df.iloc[:,i+1], ddof=1)+np.var(df.iloc[:,i], ddof=1))/2) 
		        for i in range(1,5)]
    assert cohens_d == pytest.approx(np_result)



def test_hedges_g_sequential():
    from math import gamma
    hedges_g = sequential.hedges_g.results['difference'].to_list()
    a = 47*2-2
    fac = gamma(a/2)/(np.sqrt(a/2)*gamma((a-1)/2))
    np_result = [np.mean(df.iloc[:,i+1]-df.iloc[:,i])*fac
		            /np.sqrt((np.var(df.iloc[:,i+1], ddof=1)+np.var(df.iloc[:,i], ddof=1))/2) 
		        for i in range(1,5)] 
    assert hedges_g == pytest.approx(np_result)



def test_cohens_d_baseline():
    cohens_d = baseline.cohens_d.results['difference'].to_list()
    np_result = [np.mean(df.iloc[:,i]-df.iloc[:,1])
		            /np.sqrt((np.var(df.iloc[:,i], ddof=1)+np.var(df.iloc[:,1], ddof=1))/2) 
		        for i in range(2,6)]
    assert cohens_d == pytest.approx(np_result)



def test_hedges_g_baseline():
    from math import gamma
    hedges_g = baseline.hedges_g.results['difference'].to_list()
    a = 47*2-2
    fac = gamma(a/2)/(np.sqrt(a/2)*gamma((a-1)/2))
    np_result = [np.mean(df.iloc[:,i]-df.iloc[:,1])*fac
		            /np.sqrt((np.var(df.iloc[:,i], ddof=1)+np.var(df.iloc[:,1], ddof=1))/2) 
		        for i in range(2,6)]
    assert hedges_g == pytest.approx(np_result)



def test_paired_stats_sequential():
    np_result = sequential.mean_diff.results
    
    p1 = [sp.stats.ttest_rel(df.iloc[:,i], df.iloc[:,i+1], nan_policy='omit').pvalue
		        for i in range(1,5)] 
    assert np_result["pvalue_paired_students_t"].to_list() == pytest.approx(p1)
    
    p2 = [sp.stats.wilcoxon(df.iloc[:,i], df.iloc[:,i+1]).pvalue
		        for i in range(1,5)] 
    assert np_result["pvalue_wilcoxon"].to_list() == pytest.approx(p2)



def test_paired_stats_baseline():
    np_result = baseline.mean_diff.results
    
    p1 = [sp.stats.ttest_rel(df.iloc[:,1], df.iloc[:,i], nan_policy='omit').pvalue
		        for i in range(2,6)] 
    assert np_result["pvalue_paired_students_t"].to_list() == pytest.approx(p1)
    
    p2 = [sp.stats.wilcoxon(df.iloc[:,1], df.iloc[:,i]).pvalue
		        for i in range(2,6)] 
    assert np_result["pvalue_wilcoxon"].to_list() == pytest.approx(p2)
    

   
def test_lqrt_paired_sequential():
    lqrt_result = sequential.mean_diff.lqrt["pvalue_paired_lqrt"].to_list()
                             
    p1 = [lqrt.lqrtest_rel(df.iloc[:,i], df.iloc[:,i+1], random_state=12345).pvalue
		        for i in range(1,5)] 
    
    assert lqrt_result == pytest.approx(p1)



def test_lqrt_paired_baseline():
    lqrt_result = baseline.mean_diff.lqrt["pvalue_paired_lqrt"].to_list()
                             
    p1 = [lqrt.lqrtest_rel(df.iloc[:,1], df.iloc[:,i], random_state=12345).pvalue
		        for i in range(2,6)] 
    
    assert lqrt_result == pytest.approx(p1)



