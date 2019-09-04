#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com


import sys
import numpy as np
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

    np.random.seed(random_seed)

    df = pd.DataFrame(
        {'groups': np.random.choice(['Group 1', 'Group 2', 'Group 3'], size=(N,)),
         'color' : np.random.choice(['green', 'red', 'purple'], size=(N,)),
         'value': np.random.random(size=(N,))})

    np.random.seed()

    df['unrelated'] = np.nan

    test = load(data=df, x='groups', y='value', 
                idx=['Group 1', 'Group 2'])
    
    md = test.mean_diff.results
    
    assert md.difference[0] == pytest.approx(0.1115, abs=1e-6)
    assert md.bca_low[0]    == pytest.approx(-0.042835, abs=1e-6)
    assert md.bca_high[0]   == pytest.approx(0.264542, abs=1e-6)