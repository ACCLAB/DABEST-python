# coding: utf-8
#! /usr/bin/env python

import pytest
import sys
import numpy as np
import scipy as sp

# This filters out an innocuous warning when pandas is imported,
# but the version has not been compiled against the newest numpy.
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import pandas as pd
from .._stats_tools import effsize

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


def test_cohens_d_unpaired():
    print("Testing Unpaired Cohen's d.")
    cohens_d = effsize.cohens_d(wellbeing.control, wellbeing.expt, paired=False)
    assert cohens_d == pytest.approx(0.47, rel=0.01)



def test_hedges_g_unpaired():
    print("Testing Unpaired Hedges' g.")
    cohens_d = effsize.hedges_g(wellbeing.control, wellbeing.expt, paired=False)
    assert cohens_d == pytest.approx(0.45, rel=0.01)



def test_cohens_d_paired():
    print("Testing Paired Cohen's d.")
    cohens_d = effsize.cohens_d(paired_wellbeing.pre, paired_wellbeing.post,
                                paired=True)
    assert cohens_d == pytest.approx(0.34, rel=0.05)



def test_hedges_g_paired():
    print("Testing Paired Hedges' g.")
    cohens_d = effsize.hedges_g(paired_wellbeing.pre, paired_wellbeing.post,
                                paired=True)
    assert cohens_d == pytest.approx(0.32, rel=0.05)



def test_cliffs_delta():
    print("Testing Cliff's delta.")

    likert_delta = effsize.cliffs_delta(likert_treatment, likert_control)
    assert likert_delta == pytest.approx(-0.25)

    scores_delta = effsize.cliffs_delta(b_scores, a_scores)
    assert scores_delta == pytest.approx(0.65)
