#!/usr/bin/python
# -*-coding: utf-8 -*-


import sys
import pytest
import lqrt
import numpy as np
import scipy as sp
import pandas as pd
from .._stats_tools import effsize
from .._stats_tools import confint_2group_diff as ci2g
from .._classes import PermutationTest, Dabest



# Data for tests.
# See Oehlert, G. W. (2000). A First Course in Design 
# and Analysis of Experiments (1st ed.). W. H. Freeman.
# from Problem 16.3 Pg 444.

rep1_yes = [53.4,54.3,55.9,53.8,56.3,58.6]
rep1_no = [58.2,60.4,62.4,59.5,64.5,64.5]
rep2_yes = [46.5,57.2,57.4,51.1,56.9,60.2]
rep2_no = [49.2,61.6,57.2,51.3,66.8,62.7]
df_mini_meta = pd.DataFrame({
    "Rep1_Yes":rep1_yes,
    "Rep1_No" :rep1_no,
    "Rep2_Yes":rep2_yes,
    "Rep2_No" :rep2_no
})
N=6 # Size of each group


# kwargs for Dabest class init.
dabest_default_kwargs = dict(x=None, y=None, ci=95, 
                            resamples=5000, random_seed=12345,
                            proportional=False, delta2=False, experiment=None, 
                            experiment_label=None, x1_level=None, paired=None,
                            id_col=None
                            )


unpaired = Dabest(data = df_mini_meta, idx =(("Rep1_No", "Rep1_Yes"), 
                                             ("Rep2_No", "Rep2_Yes")), 
                                             mini_meta=True,
                                             **dabest_default_kwargs)


def test_mean_diff():
    mean_diff = unpaired.mean_diff.results['difference'].to_list()
    np_result = [np.mean(rep1_yes)-np.mean(rep1_no), 
                 np.mean(rep2_yes)-np.mean(rep2_no)]
    assert mean_diff == pytest.approx(np_result)



def test_variances():
    mini_meta_delta = unpaired.mean_diff.mini_meta_delta

    control_var    = mini_meta_delta.control_var
    np_control_var = [np.var(rep1_no, ddof=1),
                      np.var(rep2_no, ddof=1)]
    assert control_var == pytest.approx(np_control_var)

    test_var    = mini_meta_delta.test_var
    np_test_var = [np.var(rep1_yes, ddof=1),
                   np.var(rep2_yes, ddof=1)]
    assert test_var == pytest.approx(np_test_var)

    group_var    = mini_meta_delta.group_var
    np_group_var = [ci2g.calculate_group_var(control_var[i], N,
                                             test_var[i], N)
                    for i in range(0, 2)]
    assert group_var == pytest.approx(np_group_var)



def test_weighted_mean_delta():
    difference = unpaired.mean_diff.mini_meta_delta.difference

    np_means = [np.mean(rep1_yes)-np.mean(rep1_no), 
                np.mean(rep2_yes)-np.mean(rep2_no)]
    np_var   = [np.var(rep1_yes, ddof=1)/N+np.var(rep1_no, ddof=1)/N,
                np.var(rep2_yes, ddof=1)/N+np.var(rep2_no, ddof=1)/N]

    np_difference = effsize.weighted_delta(np_means, np_var)

    assert difference == pytest.approx(np_difference)


def test_unpaired_permutation_test():
    mini_meta_delta   = unpaired.mean_diff.mini_meta_delta
    pvalue             = mini_meta_delta.pvalue_permutation
    permutations_delta = mini_meta_delta.permutations_weighted_delta

    perm_test_1 = PermutationTest(rep1_no, rep1_yes, 
                                effect_size="mean_diff", 
                                is_paired=False)
    perm_test_2 = PermutationTest(rep2_no, rep2_yes, 
                                effect_size="mean_diff", 
                                is_paired=False)
    permutations_1 = perm_test_1.permutations
    permutations_2 = perm_test_2.permutations
    permutations_1_var = perm_test_1.permutations_var
    permutations_2_var = perm_test_2.permutations_var

    weight_1 = np.true_divide(1,permutations_1_var)
    weight_2 = np.true_divide(1,permutations_2_var)
    
    weighted_deltas = (weight_1*permutations_1 + weight_2*permutations_2)/(weight_1+weight_2)
    assert permutations_delta == pytest.approx(weighted_deltas)


    np_means = [np.mean(rep1_yes)-np.mean(rep1_no), 
                np.mean(rep2_yes)-np.mean(rep2_no)]
    np_var   = [np.var(rep1_yes, ddof=1)/N+np.var(rep1_no, ddof=1)/N,
                np.var(rep2_yes, ddof=1)/N+np.var(rep2_no, ddof=1)/N]
    np_weight= np.true_divide(1, np_var)

    np_difference = np.sum(np_means*np_weight)/np.sum(np_weight)

    np_pvalues = len(list(filter(lambda x: x>np.abs(np_difference), 
                                weighted_deltas)))/len(weighted_deltas)

    assert pvalue == pytest.approx(np_pvalues)