"""Loading data and relevant groups"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/API/load.ipynb.

# %% auto 0
__all__ = ['load', 'prop_dataset']

# %% ../nbs/API/load.ipynb 4
def load(
    data,
    idx=None,
    x=None,
    y=None,
    paired=None,
    id_col=None,
    ci=95,
    resamples=5000,
    random_seed=12345,
    proportional=False,
    delta2=False,
    experiment=None,
    experiment_label=None,
    x1_level=None,
    mini_meta=False,
    ps_adjust=False,
):
    """
    Loads data in preparation for estimation statistics.

    This is designed to work with pandas DataFrames.

    Parameters
    ----------
    data : pandas DataFrame
    idx : tuple
        List of column names (if 'x' is not supplied) or of category names
        (if 'x' is supplied). This can be expressed as a tuple of tuples,
        with each individual tuple producing its own contrast plot
    x : string or list, default None
        Column name(s) of the independent variable. This can be expressed as
        a list of 2 elements if and only if 'delta2' is True; otherwise it
        can only be a string.
    y : string, default None
        Column names for data to be plotted on the x-axis and y-axis.
    paired : string, default None
        The type of the experiment under which the data are obtained. If 'paired'
        is None then the data will not be treated as paired data in the subsequent
        calculations. If 'paired' is 'baseline', then in each tuple of x, other
        groups will be paired up with the first group (as control). If 'paired' is
        'sequential', then in each tuple of x, each group will be paired up with
        its previous group (as control).
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
    proportional : boolean, default False.
        An indicator of whether the data is binary or not. When set to True, it
        specifies that the data consists of binary data, where the values are
        limited to 0 and 1. The code is not suitable for analyzing proportion
        data that contains non-numeric values, such as strings like 'yes' and 'no'.
        When False or not provided, the algorithm assumes that
        the data is continuous and uses a non-proportional representation.
    delta2 : boolean, default False
        Indicator of delta-delta experiment
    experiment : String, default None
        The name of the column of the dataframe which contains the label of
        experiments
    experiment_lab : list, default None
        A list of String to specify the order of subplots for delta-delta plots.
        This can be expressed as a list of 2 elements if and only if 'delta2'
        is True; otherwise it can only be a string.
    x1_level : list, default None
        A list of String to specify the order of subplots for delta-delta plots.
        This can be expressed as a list of 2 elements if and only if 'delta2'
        is True; otherwise it can only be a string.
    mini_meta : boolean, default False
        Indicator of weighted delta calculation.
    ps_adjust : boolean, default False
        Indicator of whether to adjust calculated p-value according to Phipson & Smyth (2010)
        # https://doi.org/10.2202/1544-6115.1585

    Returns
    -------
    A `Dabest` object.
    """
    from dabest import Dabest

    return Dabest(
        data,
        idx,
        x,
        y,
        paired,
        id_col,
        ci,
        resamples,
        random_seed,
        proportional,
        delta2,
        experiment,
        experiment_label,
        x1_level,
        mini_meta,
        ps_adjust,
    )

# %% ../nbs/API/load.ipynb 5
import numpy as np
from typing import Union, Optional
import pandas as pd


def prop_dataset(
    group: Union[
        list, tuple, np.ndarray, dict
    ],  # Accepts lists, tuples, or numpy ndarrays of numeric types.
    group_names: Optional[list] = None,
):
    """
    Convenient function to generate a dataframe of binary data.
    """

    if isinstance(group, dict):
        # If group_names is not provided, use the keys of the dict as group_names
        if group_names is None:
            group_names = list(group.keys())
        elif not set(group_names) == set(group.keys()):
            # Check if the group_names provided is the same as the keys of the dict
            raise ValueError("group_names must be the same as the keys of the dict.")
        
        # Check if the values in the dict are numeric
        if not all(
            [isinstance(group[name], (list, tuple, np.ndarray)) for name in group_names]
        ):
            raise ValueError(
                "group must be a dict of lists, tuples, or numpy ndarrays of numeric types."
            )
        
        # Check if the values in the dict only have two elements under each parent key
        if not all([len(group[name]) == 2 for name in group_names]):
            raise ValueError("Each parent key should have only two elements.")
        group_val = group

    else:
        if group_names is None:
            raise ValueError("group_names must be provided if group is not a dict.")
        
        # Check if the length of group is two times of the length of group_names
        if not len(group) == 2 * len(group_names):
            raise ValueError(
                "The length of group must be two times of the length of group_names."
            )
        group_val = {
            group_names[i]: [group[i * 2], group[i * 2 + 1]]
            for i in range(len(group_names))
        }

    # Check if the sum of values in group_val under each key are the same
    if not all(
        [
            sum(group_val[name]) == sum(group_val[group_names[0]])
            for name in group_val.keys()
        ]
    ):
        raise ValueError("The sum of values under each key must be the same.")

    id_col = pd.Series(range(1, sum(group_val[group_names[0]]) + 1))

    final_df = pd.DataFrame()

    for name in group_val.keys():
        col = (
            np.repeat(0, group_val[name][0]).tolist()
            + np.repeat(1, group_val[name][1]).tolist()
        )
        df = pd.DataFrame({name: col})
        final_df = pd.concat([final_df, df], axis=1)

    final_df["ID"] = id_col

    return final_df
