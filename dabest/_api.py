#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com


def load(data, idx, x=None, y=None, paired=False, id_col=None, **kwargs):
    '''
    Loads data in preparation for estimation statistics.
    This is designed to work with pandas DataFrames.

    Keywords
    --------
    data: pandas DataFrame

    idx: tuple
        List of column names (if 'x' is not supplied) or of category names
        (if 'x' is supplied). This can be expressed as a tuple of tuples,
        with each individual tuple producing its own contrast plot.

    x, y: strings, default None
        Column names for data to be plotted on the x-axis and y-axis.

    paired: boolean, default False.

    Example
    --------
    1. Load libraries.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import dabest

    2. Create dummy data for demonstration.

    >>> np.random.seed(88888)
    >>> N = 10
    >>> c1 = sp.stats.norm.rvs(loc=100, scale=5, size=N)
    >>> t1 = sp.stats.norm.rvs(loc=115, scale=5, size=N)
    >>> df = pd.DataFrame({'Control 1' : c1, 'Test 1': t1})

    3. Load the data.

    >>> my_data = dabest.load(df, idx=("Control 1", "Test 1"))
    
    '''
    from .classes import Dabest

    return Dabest(data, idx, x, y, paired, id_col, **kwargs)
