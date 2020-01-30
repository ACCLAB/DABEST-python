#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com


import sys
import numpy as np
from numpy.random import PCG64, RandomState
import scipy as sp
import pytest
import pandas as pd
from .._api import load



def test_unrelated_columns(N=60, random_seed=12345):
    """
    Test to see if 'unrelated' columns jam up the analysis.
    See Github Issue 43.
    https://github.com/ACCLAB/DABEST-python/issues/44.
    
    Added in v0.2.5.
    """
    
    # rng = RandomState(MT19937(random_seed))
    rng = RandomState(PCG64(12345))
    # rng = np.random.default_rng(seed=random_seed)

    df = pd.DataFrame(
        {'groups': rng.choice(['Group 1', 'Group 2', 'Group 3'], size=(N,)),
         'color' : rng.choice(['green', 'red', 'purple'], size=(N,)),
         'value':  rng.random(size=(N,))})

    df['unrelated'] = np.nan

    test = load(data=df, x='groups', y='value', 
                idx=['Group 1', 'Group 2'])
    
    md = test.mean_diff.results
    
    assert md.difference[0] == pytest.approx(-0.0322, abs=1e-4)
    assert md.bca_low[0]    == pytest.approx(-0.2279, abs=1e-4)
    assert md.bca_high[0]   == pytest.approx(0.1613, abs=1e-4)