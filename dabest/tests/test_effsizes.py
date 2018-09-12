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



def test_cohens_d_unpaired():
    cohens_d = effsize.cohens_d(wellbeing.control, wellbeing.expt, paired=False)
    assert cohens_d == pytest.approx(0.47, rel=0.01)



def test_hedges_g_unpaired():
    cohens_d = effsize.hedges_g(wellbeing.control, wellbeing.expt, paired=False)
    assert cohens_d == pytest.approx(0.45, rel=0.01)



def test_cohens_d_paired():
    cohens_d = effsize.cohens_d(paired_wellbeing.pre, paired_wellbeing.post,
                                paired=True)
    assert cohens_d == pytest.approx(0.34, rel=0.05)



def test_hedges_g_paired():
    cohens_d = effsize.hedges_g(paired_wellbeing.pre, paired_wellbeing.post,
                                paired=True)
    assert cohens_d == pytest.approx(0.32, rel=0.05)



def cliffs_delta():
    pass
