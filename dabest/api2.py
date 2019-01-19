#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com


def load(data, idx, x=None, y=None, paired=False):
    '''
    Create a specialised object for estimation statistics.
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

    Examples
    --------
    TO BE ADDED

    '''
    from .classes import Dabest

    return Dabest(data, idx, x, y, paired)
