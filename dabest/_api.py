#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com


def load(data, idx, x=None, y=None, paired=False, id_col=None,
        ci=95, resamples=5000, random_seed=12345):
    '''
    Loads data in preparation for estimation statistics.

    This is designed to work with pandas DataFrames.

    Parameters
    ----------
    data : pandas DataFrame
    idx : tuple
        List of column names (if 'x' is not supplied) or of category names
        (if 'x' is supplied). This can be expressed as a tuple of tuples,
        with each individual tuple producing its own contrast plot
    x : string, default None
    y : string, default None
        Column names for data to be plotted on the x-axis and y-axis.
    paired : boolean, default False.
    id_col : default None.
        Required if `paired` is True.
    ci : integer, default 95
        The confidence interval width. The default of 95 produces 95%
        confidence intervals.
    resamples : integer, default 5000.
        The number of resamples taken to generate the bootstraps which are used
        to generate the confidence intervals.
    random_seed : int, default 12345
        This integer is used to seed the random number generator during
        bootstrap resampling, ensuring that the confidence intervals
        reported are replicable.

    Returns
    -------
    A `Dabest` object.

    Example
    --------
    Load libraries.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import dabest

    Create dummy data for demonstration.

    >>> np.random.seed(88888)
    >>> N = 10
    >>> c1 = sp.stats.norm.rvs(loc=100, scale=5, size=N)
    >>> t1 = sp.stats.norm.rvs(loc=115, scale=5, size=N)
    >>> df = pd.DataFrame({'Control 1' : c1, 'Test 1': t1})

    Load the data.

    >>> my_data = dabest.load(df, idx=("Control 1", "Test 1"))

    '''
    from ._classes import Dabest

    return Dabest(data, idx, x, y, paired, id_col, ci, resamples, random_seed)
