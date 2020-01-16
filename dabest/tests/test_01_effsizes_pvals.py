#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com


import sys
import pytest
import lqrt
import numpy as np
import scipy as sp
import pandas as pd
from .._stats_tools import effsize
from .._classes import TwoGroupsEffectSize



# Data for tests.
# See Cumming, G. Understanding the New Statistics:
# Effect Sizes, Confidence Intervals, and Meta-Analysis. Routledge, 2012,
# from Cumming 2012 Table 11.1 Pg 287.
wb = {"control": [34, 54, 33, 44, 45, 53, 37, 26, 38, 58],
      "expt":    [66, 38, 35, 55, 48, 39, 65, 32, 57, 41]}
wellbeing = pd.DataFrame(wb)



# from Cumming 2012 Table 11.2 Page 291
paired_wb = {"pre":     [43, 28, 54, 36, 31, 48, 50, 69, 29, 40],
             "post":    [51, 33, 58, 42, 39, 45, 54, 68, 35, 44]}
paired_wellbeing = pd.DataFrame(paired_wb)



# Data from Hogarty and Kromrey (1999)
# Kromrey, Jeffrey D., and Kristine Y. Hogarty. 1998.
# “Analysis Options for Testing Group Differences on Ordered Categorical
# Variables: An Empirical Investigation of Type I Error Control
# Statistical Power.”
# Multiple Linear Regression Viewpoints 25 (1): 70 - 82.
likert_control   = [1, 1, 2, 2, 2, 3, 3, 3, 4, 5]
likert_treatment = [1, 2, 3, 4, 4, 5]



# Data from Cliff (1993)
# Cliff, Norman. 1993. “Dominance Statistics: Ordinal Analyses to Answer
# Ordinal Questions.”
# Psychological Bulletin 114 (3): 494–509.
a_scores = [6, 7, 9, 10]
b_scores = [1, 3, 4, 7, 8]






def test_mean_diff_unpaired():
    import numpy as np
    mean_diff = effsize.func_difference(wellbeing.control, wellbeing.expt,
                                        np.mean, is_paired=False)
    assert mean_diff == pytest.approx(5.4)



def test_median_diff_unpaired():
    from numpy import median as npmedian
    median_diff = effsize.func_difference(wellbeing.control, wellbeing.expt,
                                        npmedian, is_paired=False)
    assert median_diff == pytest.approx(3.5)



def test_mean_diff_paired():
    from numpy import mean as npmean
    mean_diff = effsize.func_difference(paired_wellbeing.pre,
                                        paired_wellbeing.post,
                                        npmean, is_paired=True)
    assert mean_diff == pytest.approx(4.10)



def test_median_diff_paired():
    from numpy import median as npmedian
    median_diff = effsize.func_difference(paired_wellbeing.pre,
                                          paired_wellbeing.post,
                                          npmedian, is_paired=True)
    assert median_diff == pytest.approx(4.5)



def test_cohens_d_unpaired():
    import numpy as np
    cohens_d = effsize.cohens_d(wellbeing.control, wellbeing.expt,
                                is_paired=False)
    assert np.round(cohens_d, 2) == pytest.approx(0.47)



def test_hedges_g_unpaired():
    import numpy as np
    hedges_g = effsize.hedges_g(wellbeing.control, wellbeing.expt,
                                is_paired=False)
    assert np.round(hedges_g, 2) == pytest.approx(0.45)



def test_cohens_d_paired():
    import numpy as np
    cohens_d = effsize.cohens_d(paired_wellbeing.pre, paired_wellbeing.post,
                                is_paired=True)
    assert np.round(cohens_d, 2) == pytest.approx(0.34)



def test_hedges_g_paired():
    import numpy as np
    hedges_g = effsize.hedges_g(paired_wellbeing.pre, paired_wellbeing.post,
                                is_paired=True)
    assert np.round(hedges_g, 2) == pytest.approx(0.33)



def test_cliffs_delta():
    likert_delta = effsize.cliffs_delta(likert_treatment, likert_control)
    assert likert_delta == pytest.approx(-0.25)

    scores_delta = effsize.cliffs_delta(b_scores, a_scores)
    assert scores_delta == pytest.approx(0.65)
    
    
    
def test_unpaired_stats():
    c = wellbeing.control
    t = wellbeing.expt
    
    unpaired_es = TwoGroupsEffectSize(c, t, "mean_diff", is_paired=False)
    
    p1 = sp.stats.mannwhitneyu(c, t, alternative="two-sided").pvalue
    assert unpaired_es.pvalue_mann_whitney == pytest.approx(p1)
    
    p2 = sp.stats.ttest_ind(c, t, nan_policy='omit').pvalue
    assert unpaired_es.pvalue_students_t == pytest.approx(p2)
    
    p3 = sp.stats.ttest_ind(c, t, equal_var=False, nan_policy='omit').pvalue
    assert unpaired_es.pvalue_welch == pytest.approx(p3)
    
    
    
def test_paired_stats():
    before = paired_wellbeing.pre
    after = paired_wellbeing.post
    
    paired_es = TwoGroupsEffectSize(before, after, "mean_diff", is_paired=True)
    
    p1 = sp.stats.ttest_rel(before, after, nan_policy='omit').pvalue
    assert paired_es.pvalue_paired_students_t == pytest.approx(p1)
    
    p2 = sp.stats.wilcoxon(before, after).pvalue
    assert paired_es.pvalue_wilcoxon == pytest.approx(p2)
    
    

def test_median_diff_stats():
    c = wellbeing.control
    t = wellbeing.expt
    
    es = TwoGroupsEffectSize(c, t, "median_diff", is_paired=False)
    
    p1 = sp.stats.kruskal(c, t, nan_policy='omit').pvalue
    assert es.pvalue_kruskal == pytest.approx(p1)
    
    
    
def test_ordinal_dominance():
    es = TwoGroupsEffectSize(likert_control, likert_treatment, 
                             "cliffs_delta", is_paired=False)
                             
    p1 = sp.stats.brunnermunzel(likert_control, likert_treatment).pvalue
    assert es.pvalue_brunner_munzel == pytest.approx(p1)
    
    

def test_lqrt_unpaired():
    es = TwoGroupsEffectSize(wellbeing.control, wellbeing.expt, 
                             "mean_diff", is_paired=False)
                             
    p1 = lqrt.lqrtest_ind(wellbeing.control, wellbeing.expt,
                          equal_var=False,
                          random_state=12345)
                          
    p2 = lqrt.lqrtest_ind(wellbeing.control, wellbeing.expt,
                          equal_var=True,
                          random_state=12345)
    
    assert es.pvalue_lqrt_unpaired_unequal_variance == pytest.approx(p1.pvalue)
    assert es.pvalue_lqrt_unpaired_equal_variance == pytest.approx(p2.pvalue)
    
    
def test_lqrt_paired():
    es = TwoGroupsEffectSize(paired_wellbeing.pre, paired_wellbeing.post, 
                             "mean_diff", is_paired=True)
                             
    p1 = lqrt.lqrtest_rel(paired_wellbeing.pre, paired_wellbeing.post, 
                 random_state=12345)
    
    assert es.pvalue_lqrt_paired == pytest.approx(p1.pvalue)