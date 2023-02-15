#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com
# A set of convenience functions used for producing plots in `dabest`.


from .misc_tools import merge_two_dicts



def halfviolin(v, half='right', fill_color='k', alpha=1,
                line_color='k', line_width=0):
    import numpy as np

    for b in v['bodies']:
        V = b.get_paths()[0].vertices

        mean_vertical = np.mean(V[:, 0])
        mean_horizontal = np.mean(V[:, 1])

        if half == 'right':
            V[:, 0] = np.clip(V[:, 0], mean_vertical, np.inf)
        elif half == 'left':
            V[:, 0] = np.clip(V[:, 0], -np.inf, mean_vertical)
        elif half == 'bottom':
            V[:, 1] = np.clip(V[:, 1], -np.inf, mean_horizontal)
        elif half == 'top':
            V[:, 1] = np.clip(V[:, 1], mean_horizontal, np.inf)

        b.set_color(fill_color)
        b.set_alpha(alpha)
        b.set_edgecolor(line_color)
        b.set_linewidth(line_width)



# def align_yaxis(ax1, v1, ax2, v2):
#     """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
#     # Taken from
#     # http://stackoverflow.com/questions/7630778/
#     # matplotlib-align-origin-of-right-axis-with-specific-left-axis-value
#     _, y1 = ax1.transData.transform((0, v1))
#     _, y2 = ax2.transData.transform((0, v2))
#     inv = ax2.transData.inverted()
#     _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
#     miny, maxy = ax2.get_ylim()
#     ax2.set_ylim(miny+dy, maxy+dy)
#
#
#
# def rotate_ticks(axes, angle=45, alignment='right'):
#     for tick in axes.get_xticklabels():
#         tick.set_rotation(angle)
#         tick.set_horizontalalignment(alignment)



def get_swarm_spans(coll):
    """
    Given a matplotlib Collection, will obtain the x and y spans
    for the collection. Will return None if this fails.
    """
    import numpy as np
    x, y = np.array(coll.get_offsets()).T
    try:
        return x.min(), x.max(), y.min(), y.max()
    except ValueError:
        return None



def gapped_lines(data, x, y, type='mean_sd', offset=0.2, ax=None,
                line_color="black", gap_width_percent=1,
                **kwargs):
    '''
    Convenience function to plot the standard devations as vertical
    errorbars. The mean is a gap defined by negative space.

    This style is inspired by Edward Tufte's redesign of the boxplot.
    See The Visual Display of Quantitative Information (1983), pp.128-130.

    Keywords
    --------
    data: pandas DataFrame.
        This DataFrame should be in 'long' format.

    x, y: string.
        x and y columns to be plotted.

    type: ['mean_sd', 'median_quartiles'], default 'mean_sd'
        Plots the summary statistics for each group. If 'mean_sd', then the
        mean and standard deviation of each group is plotted as a gapped line.
        If 'median_quantiles', then the median and 25th and 75th percentiles of
        each group is plotted instead.

    offset: float (default 0.3) or iterable.
        Give a single float (that will be used as the x-offset of all
        gapped lines), or an iterable containing the list of x-offsets.

    line_color: string (matplotlib color, default "black") or iterable of
        matplotlib colors.

        The color of the vertical line indicating the stadard deviations.

    gap_width_percent: float, default 5
        The width of the gap in the line (indicating the central measure),
        expressed as a percentage of the y-span of the axes.

    ax: matplotlib Axes object, default None
        If a matplotlib Axes object is specified, the gapped lines will be
        plotted in order on this axes. If None, the current axes (plt.gca())
        is used.

    kwargs: dict, default None
        Dictionary with kwargs passed to matplotlib.lines.Line2D
    '''
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    if gap_width_percent < 0 or gap_width_percent > 100:
        raise ValueError("`gap_width_percent` must be between 0 and 100.")

    if ax is None:
        ax = plt.gca()
    ax_ylims = ax.get_ylim()
    ax_yspan = np.abs(ax_ylims[1] - ax_ylims[0])
    gap_width = ax_yspan * gap_width_percent/100

    keys = kwargs.keys()
    if 'clip_on' not in keys:
        kwargs['clip_on'] = False

    if 'zorder' not in keys:
        kwargs['zorder'] = 5

    if 'lw' not in keys:
        kwargs['lw'] = 2.

    # # Grab the order in which the groups appear.
    # group_order = pd.unique(data[x])
    
    # Grab the order in which the groups appear,
    # depending on whether the x-column is categorical.
    if isinstance(data[x].dtype, pd.CategoricalDtype):
        group_order = pd.unique(data[x]).categories
    else:
        group_order = pd.unique(data[x])

    means    = data.groupby(x)[y].mean().reindex(index=group_order)
    sd       = data.groupby(x)[y].std().reindex(index=group_order)
    lower_sd = means - sd
    upper_sd = means + sd


    if (lower_sd < ax_ylims[0]).any() or (upper_sd > ax_ylims[1]).any():
        kwargs['clip_on'] = True

    medians   = data.groupby(x)[y].median().reindex(index=group_order)
    quantiles = data.groupby(x)[y].quantile([0.25, 0.75])\
                                  .unstack()\
                                  .reindex(index=group_order)
    lower_quartiles = quantiles[0.25]
    upper_quartiles = quantiles[0.75]


    if type == 'mean_sd':
        central_measures = means
        lows = lower_sd
        highs = upper_sd
    elif type == 'median_quartiles':
        central_measures = medians
        lows = lower_quartiles
        highs = upper_quartiles


    n_groups = len(central_measures)

    if isinstance(line_color, str):
        custom_palette = np.repeat(line_color, n_groups)
    else:
        if len(line_color) != n_groups:
            err1 = "{} groups are being plotted, but ".format(n_groups)
            err2 = "{} colors(s) were supplied in `line_color`.".format(len(line_color))
            raise ValueError(err1 + err2)
        custom_palette = line_color

    try:
        len_offset = len(offset)
    except TypeError:
        offset = np.repeat(offset, n_groups)
        len_offset = len(offset)

    if len_offset != n_groups:
        err1 = "{} groups are being plotted, but ".format(n_groups)
        err2 = "{} offset(s) were supplied in `offset`.".format(len_offset)
        raise ValueError(err1 + err2)

    kwargs['zorder'] = kwargs['zorder']

    for xpos, central_measure in enumerate(central_measures):
        # add lower vertical span line.

        kwargs['color'] = custom_palette[xpos]

        _xpos = xpos + offset[xpos]
        # add lower vertical span line.
        low = lows[xpos]
        low_to_mean = mlines.Line2D([_xpos, _xpos],
                                    [low, central_measure-gap_width],
                                      **kwargs)
        ax.add_line(low_to_mean)

        # add upper vertical span line.
        high = highs[xpos]
        mean_to_high = mlines.Line2D([_xpos, _xpos],
                                     [central_measure+gap_width, high],
                                      **kwargs)
        ax.add_line(mean_to_high)

        # # add horzontal central measure line.
        # kwargs['zorder'] = 6
        # kwargs['color'] = gap_color
        # kwargs['lw'] = kwargs['lw'] * 1.5
        # line_xpos = xpos + offset[xpos]
        # mean_line = mlines.Line2D([line_xpos-0.015, line_xpos+0.015],
        #                           [central_measure, central_measure], **kwargs)
        # ax.add_line(mean_line)


def proportion_error_bar(data, x, y, type='mean_sd', offset=0.2, ax=None,
                 line_color="black", gap_width_percent=1,
                 **kwargs):
    '''
    Function to plot the standard devations for proportions as vertical
    errorbars. The mean is a gap defined by negative space.

    This style is inspired by Edward Tufte's redesign of the boxplot.
    See The Visual Display of Quantitative Information (1983), pp.128-130.

    Keywords
    --------
    data: pandas DataFrame.
        This DataFrame should be in 'long' format.

    x, y: string.
        x and y columns to be plotted.

    type: ['mean_sd', 'median_quartiles'], default 'mean_sd'
        Plots the summary statistics for each group. If 'mean_sd', then the
        mean and standard deviation of each group is plotted as a gapped line.
        If 'median_quantiles', then the median and 25th and 75th percentiles of
        each group is plotted instead.

    offset: float (default 0.3) or iterable.
        Give a single float (that will be used as the x-offset of all
        gapped lines), or an iterable containing the list of x-offsets.

    line_color: string (matplotlib color, default "black") or iterable of
        matplotlib colors.

        The color of the vertical line indicating the stadard deviations.

    gap_width_percent: float, default 5
        The width of the gap in the line (indicating the central measure),
        expressed as a percentage of the y-span of the axes.

    ax: matplotlib Axes object, default None
        If a matplotlib Axes object is specified, the gapped lines will be
        plotted in order on this axes. If None, the current axes (plt.gca())
        is used.

    kwargs: dict, default None
        Dictionary with kwargs passed to matplotlib.lines.Line2D
    '''
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    if gap_width_percent < 0 or gap_width_percent > 100:
        raise ValueError("`gap_width_percent` must be between 0 and 100.")

    if ax is None:
        ax = plt.gca()
    ax_ylims = ax.get_ylim()
    ax_yspan = np.abs(ax_ylims[1] - ax_ylims[0])
    gap_width = ax_yspan * gap_width_percent / 100

    keys = kwargs.keys()
    if 'clip_on' not in keys:
        kwargs['clip_on'] = False

    if 'zorder' not in keys:
        kwargs['zorder'] = 5

    if 'lw' not in keys:
        kwargs['lw'] = 2.

    # # Grab the order in which the groups appear.
    # group_order = pd.unique(data[x])

    # Grab the order in which the groups appear,
    # depending on whether the x-column is categorical.
    if isinstance(data[x].dtype, pd.CategoricalDtype):
        group_order = pd.unique(data[x]).categories
    else:
        group_order = pd.unique(data[x])

    means = data.groupby(x)[y].mean().reindex(index=group_order)
    g = lambda x: np.sqrt((np.sum(x) * (len(x) - np.sum(x))) / (len(x) * len(x) * len(x)))
    sd = data.groupby(x)[y].apply(g)
    # sd = data.groupby(x)[y].std().reindex(index=group_order)
    lower_sd = means - sd
    upper_sd = means + sd

    if (lower_sd < ax_ylims[0]).any() or (upper_sd > ax_ylims[1]).any():
        kwargs['clip_on'] = True

    medians = data.groupby(x)[y].median().reindex(index=group_order)
    quantiles = data.groupby(x)[y].quantile([0.25, 0.75]) \
        .unstack() \
        .reindex(index=group_order)
    lower_quartiles = quantiles[0.25]
    upper_quartiles = quantiles[0.75]

    if type == 'mean_sd':
        central_measures = means
        lows = lower_sd
        highs = upper_sd
    elif type == 'median_quartiles':
        central_measures = medians
        lows = lower_quartiles
        highs = upper_quartiles

    n_groups = len(central_measures)

    if isinstance(line_color, str):
        custom_palette = np.repeat(line_color, n_groups)
    else:
        if len(line_color) != n_groups:
            err1 = "{} groups are being plotted, but ".format(n_groups)
            err2 = "{} colors(s) were supplied in `line_color`.".format(len(line_color))
            raise ValueError(err1 + err2)
        custom_palette = line_color

    try:
        len_offset = len(offset)
    except TypeError:
        offset = np.repeat(offset, n_groups)
        len_offset = len(offset)

    if len_offset != n_groups:
        err1 = "{} groups are being plotted, but ".format(n_groups)
        err2 = "{} offset(s) were supplied in `offset`.".format(len_offset)
        raise ValueError(err1 + err2)

    kwargs['zorder'] = kwargs['zorder']

    for xpos, central_measure in enumerate(central_measures):
        # add lower vertical span line.

        kwargs['color'] = custom_palette[xpos]

        _xpos = xpos + offset[xpos]
        # add lower vertical span line.
        low = lows[xpos]
        low_to_mean = mlines.Line2D([_xpos, _xpos],
                                    [low, central_measure - gap_width],
                                    **kwargs)
        ax.add_line(low_to_mean)

        # add upper vertical span line.
        high = highs[xpos]
        mean_to_high = mlines.Line2D([_xpos, _xpos],
                                     [central_measure + gap_width, high],
                                     **kwargs)
        ax.add_line(mean_to_high)

        # # add horzontal central measure line.
        # kwargs['zorder'] = 6
        # kwargs['color'] = gap_color
        # kwargs['lw'] = kwargs['lw'] * 1.5
        # line_xpos = xpos + offset[xpos]
        # mean_line = mlines.Line2D([line_xpos-0.015, line_xpos+0.015],
        #                           [central_measure, central_measure], **kwargs)
        # ax.add_line(mean_line)

def check_data_matches_labels(labels, data, side):
    '''
    Function to check that the labels and data match in the sankey diagram. 
    And enforce labels and data to be lists.
    Raises an exception if the labels and data do not match.

    Keywords
    --------
    labels: list of input labels
    data: Pandas Series of input data
    side: string, 'left' or 'right' on the sankey diagram
    '''
    import pandas as pd
    if len(labels > 0):
        if isinstance(data, list):
            data = set(data)
        if isinstance(data, pd.Series):
            data = set(data.unique().tolist())
        if isinstance(labels, list):
            labels = set(labels)
        if labels != data:
            msg = "\n"
            if len(labels) <= 20:
                msg = "Labels: " + ",".join(labels) + "\n"
            if len(data) < 20:
                msg += "Data: " + ",".join(data)
            raise Exception('{0} labels and data do not match.{1}'.format(side, msg))

def single_sankey(left, right, xpos=0, leftWeight=None, rightWeight=None, 
            colorDict=None, leftLabels=None, rightLabels=None, ax=None, 
            width=0.5, alpha=0.65, bar_width=0.1, rightColor=False, align='center'):

    '''
    Make a single Sankey diagram showing proportion flow from left to right
    Original code from: https://github.com/anazalea/pySankey
    Changes are added to normalize each diagram's height to be 1

    Keywords
    --------
    left: NumPy array 
        data on the left of the diagram
    right: NumPy array 
        data on the right of the diagram
        len(left) == len(right)
    xpos: float
        the starting point on the x-axis
    leftWeight: NumPy array
        weights for the left labels, if None, all weights are 1
    rightWeight: NumPy array
         weights for the right labels, if None, all weights are corresponding leftWeight
    colorDict: dictionary of colors for each label
        input format: {'label': 'color'}
    leftLabels: list
        labels for the left side of the diagram. The diagram will be sorted by these labels.
    rightLabels: list
        labels for the right side of the diagram. The diagram will be sorted by these labels.
    ax: matplotlib axes to be drawn on
    aspect: float
        vertical extent of the diagram in units of horizontal extent
    rightColor: bool
        if True, each strip of the diagram will be colored according to the corresponding left labels
    align: bool
        if 'center', the diagram will be centered on each xtick, 
        if 'edge', the diagram will be aligned with the left edge of each xtick
    '''
    
    from collections import defaultdict

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd


    # Initiating values
    if ax is None:
        ax = plt.gca()

    if leftWeight is None:
        leftWeight = []
    if rightWeight is None:
        rightWeight = []
    if leftLabels is None:
        leftLabels = []
    if rightLabels is None:
        rightLabels = []
    # Check weights
    if len(leftWeight) == 0:
        leftWeight = np.ones(len(left))
    if len(rightWeight) == 0:
        rightWeight = leftWeight

    # Create Dataframe
    if isinstance(left, pd.Series):
        left = left.reset_index(drop=True)
    if isinstance(right, pd.Series):
        right = right.reset_index(drop=True)
    dataFrame = pd.DataFrame({'left': left, 'right': right, 'leftWeight': leftWeight,
                              'rightWeight': rightWeight}, index=range(len(left)))
    
    if len(dataFrame[(dataFrame.left.isnull()) | (dataFrame.right.isnull())]):
        raise Exception('Sankey graph does not support null values.')

    # Identify all labels that appear 'left' or 'right'
    allLabels = pd.Series(np.sort(np.r_[dataFrame.left.unique(), dataFrame.right.unique()])[::-1]).unique()

    # Identify left labels
    if len(leftLabels) == 0:
        leftLabels = pd.Series(np.sort(dataFrame.left.unique())[::-1]).unique()
    else:
        check_data_matches_labels(leftLabels, dataFrame['left'], 'left')

    # Identify right labels
    if len(rightLabels) == 0:
        rightLabels = pd.Series(np.sort(dataFrame.right.unique())[::-1]).unique()
    else:
        check_data_matches_labels(leftLabels, dataFrame['right'], 'right')

    # If no colorDict given, make one
    if colorDict is None:
        colorDict = {}
        palette = "hls"
        colorPalette = sns.color_palette(palette, len(allLabels))
        for i, label in enumerate(allLabels):
            colorDict[label] = colorPalette[i]
        fail_color = {0:"grey"}
        colorDict.update(fail_color)
    else:
        missing = [label for label in allLabels if label not in colorDict.keys()]
        if missing:
            msg = "The palette parameter is missing values for the following labels : "
            msg += '{}'.format(', '.join(missing))
            raise ValueError(msg)

    if align not in ("center", "edge"):
        err = '{} assigned for `align` is not valid.'.format(align)
        raise ValueError(err)
    if align == "center":
        try:
            leftpos = xpos - width / 2
        except TypeError as e:
            raise TypeError(f'the dtypes of parameters x ({xpos.dtype}) '
                            f'and width ({width.dtype}) '
                            f'are incompatible') from e
    else: 
        leftpos = xpos


    # Determine positions of left label patches and total widths
    # We also want the height of the graph to be 1
    leftWidths_norm = defaultdict()
    for i, leftLabel in enumerate(leftLabels):
        myD = {}
        myD['left'] = (dataFrame[dataFrame.left == leftLabel].leftWeight.sum()/ \
            dataFrame.leftWeight.sum())*(1-(len(leftLabels)-1)*0.02)
        if i == 0:
            myD['bottom'] = 0
            myD['top'] = myD['left']
        else:
            myD['bottom'] = leftWidths_norm[leftLabels[i - 1]]['top'] + 0.02
            myD['top'] = myD['bottom'] + myD['left']
            topEdge = myD['top']
        leftWidths_norm[leftLabel] = myD

    # Determine positions of right label patches and total widths
    rightWidths_norm = defaultdict()
    for i, rightLabel in enumerate(rightLabels):
        myD = {}
        myD['right'] = (dataFrame[dataFrame.right == rightLabel].rightWeight.sum()/ \
            dataFrame.rightWeight.sum())*(1-(len(leftLabels)-1)*0.02)
        if i == 0:
            myD['bottom'] = 0
            myD['top'] = myD['right']
        else:
            myD['bottom'] = rightWidths_norm[rightLabels[i - 1]]['top'] + 0.02
            myD['top'] = myD['bottom'] + myD['right']
            topEdge = myD['top']
        rightWidths_norm[rightLabel] = myD    

    # Total width of the graph
    xMax = width

    # Determine widths of individual strips, all widths are normalized to 1
    ns_l = defaultdict()
    ns_r = defaultdict()
    ns_l_norm = defaultdict()
    ns_r_norm = defaultdict()
    for leftLabel in leftLabels:
        leftDict = {}
        rightDict = {}
        for rightLabel in rightLabels:
            leftDict[rightLabel] = dataFrame[
                (dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)
                ].leftWeight.sum()
                
            rightDict[rightLabel] = dataFrame[
                (dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)
                ].rightWeight.sum()
        factorleft = leftWidths_norm[leftLabel]['left']/sum(leftDict.values())
        leftDict_norm = {k: v*factorleft for k, v in leftDict.items()}
        ns_l_norm[leftLabel] = leftDict_norm
        ns_r[leftLabel] = rightDict
    
    # ns_r should be using a different way of normalization to fit the right side
    # It is normalized using the value with the same key in each sub-dictionary
    def normalize_dict(nested_dict, target):
        val = {}
        for key in nested_dict.keys():
            val[key] = np.sum([nested_dict[sub_key][key] for sub_key in nested_dict.keys()])
        
        for key, value in nested_dict.items():
            if isinstance(value, dict):
                for subkey in value.keys():
                    value[subkey] = value[subkey] * target[subkey]['right']/val[subkey]
        return nested_dict

    ns_r_norm = normalize_dict(ns_r, rightWidths_norm)

    # Plot vertical bars for each label
    for leftLabel in leftLabels:
        ax.fill_between(
            [leftpos + (-(bar_width) * xMax), leftpos],
            2 * [leftWidths_norm[leftLabel]["bottom"]],
            2 * [leftWidths_norm[leftLabel]["bottom"] + leftWidths_norm[leftLabel]["left"]],
            color=colorDict[leftLabel],
            alpha=0.99,
        )
    for rightLabel in rightLabels:
        ax.fill_between(
            [xMax + leftpos, leftpos + ((1 + bar_width) * xMax)], 
            2 * [rightWidths_norm[rightLabel]['bottom']],
            2 * [rightWidths_norm[rightLabel]['bottom'] + rightWidths_norm[rightLabel]['right']],
            color=colorDict[rightLabel],
            alpha=0.99
        )
    
    # Plot strips
    for leftLabel in leftLabels:
        for rightLabel in rightLabels:
            labelColor = leftLabel
            if rightColor:
                labelColor = rightLabel
            if len(dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)]) > 0:
                # Create array of y values for each strip, half at left value,
                # half at right, convolve
                ys_d = np.array(50 * [leftWidths_norm[leftLabel]['bottom']] + \
                    50 * [rightWidths_norm[rightLabel]['bottom']])
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_u = np.array(50 * [leftWidths_norm[leftLabel]['bottom'] + ns_l_norm[leftLabel][rightLabel]] + \
                    50 * [rightWidths_norm[rightLabel]['bottom'] + ns_r_norm[leftLabel][rightLabel]])
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')

                # Update bottom edges at each label so next strip starts at the right place
                leftWidths_norm[leftLabel]['bottom'] += ns_l_norm[leftLabel][rightLabel]
                rightWidths_norm[rightLabel]['bottom'] += ns_r_norm[leftLabel][rightLabel]
                ax.fill_between(
                    np.linspace(leftpos, leftpos + xMax, len(ys_d)), ys_d, ys_u, alpha=alpha,
                    color=colorDict[labelColor], edgecolor='none'
                )
                
def sankeydiag(data, xvar, yvar, left_idx, right_idx, 
                leftLabels=None, rightLabels=None,  
                palette=None, ax=None, 
                one_sankey=False,
                width=0.5, rightColor=False,
                align='center', alpha=0.65, **kwargs):
    '''
    Read in melted pd.DataFrame, and draw multiple sankey diagram on a single axes
    using the value in column yvar according to the value in column xvar
    left_idx in the column xvar is on the left side of each sankey diagram
    right_idx in the column xvar is on the right side of each sankey diagram

    Keywords
    --------
    data: pd.DataFrame
        input data, melted dataframe created by dabest.load()
    left_idx: str
        the value in column xvar that is on the left side of each sankey diagram
    right_idx: str
        the value in column xvar that is on the right side of each sankey diagram
        if len(left_idx) == 1, it will be broadcasted to the same length as right_idx
        otherwise it should have the same length as right_idx
    leftLabels: list
        labels for the left side of the diagram. The diagram will be sorted by these labels.
    rightLabels: list
        labels for the right side of the diagram. The diagram will be sorted by these labels.
    palette: str or dict
    ax: matplotlib axes to be drawn on
    one_sankey: bool 
        determined by the driver function on plotter.py. 
        if True, draw the sankey diagram across the whole raw data axes
    width: float
        the width of each sankey diagram
    align: str
        the alignment of each sankey diagram, can be 'center' or 'left'
    alpha: float
        the transparency of each strip
    rightColor: bool
        if True, each strip of the diagram will be colored according to the corresponding left labels
    colorDict: dictionary of colors for each label
        input format: {'label': 'color'}
    '''

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    if "width" in kwargs:
        width = kwargs["width"]

    if "align" in kwargs:
        align = kwargs["align"]
    
    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
    
    if "rightColor" in kwargs:
        rightColor = kwargs["rightColor"]
    
    if "bar_width" in kwargs:
        bar_width = kwargs["bar_width"]

    if ax is None:
        ax = plt.gca()

    allLabels = pd.Series(np.sort(data[yvar].unique())[::-1]).unique()
        
    # Check if all the elements in left_idx and right_idx are in xvar column
    if not all(elem in data[xvar].unique() for elem in left_idx):
        raise ValueError(f"{left_idx} not found in {xvar} column")
    if not all(elem in data[xvar].unique() for elem in right_idx):
        raise ValueError(f"{right_idx} not found in {xvar} column")

    xpos = 0

    # For baseline comparison, broadcast left_idx to the same length as right_idx
    # so that the left of sankey diagram will be the same
    # For sequential comparison, left_idx and right_idx can have anything different 
    # but should have the same length
    if len(left_idx) == 1:
        broadcasted_left = np.broadcast_to(left_idx, len(right_idx))
    elif len(left_idx) != len(right_idx):
        raise ValueError(f"left_idx and right_idx should have the same length")
    else:
        broadcasted_left = left_idx

    if isinstance(palette, dict):
        if not all(key in allLabels for key in palette.keys()):
            raise ValueError(f"keys in palette should be in {yvar} column")
        else: 
            plot_palette = palette
    elif isinstance(palette, str):
        plot_palette = {}
        colorPalette = sns.color_palette(palette, len(allLabels))
        for i, label in enumerate(allLabels):
            plot_palette[label] = colorPalette[i]
    else:
        plot_palette = None

    for left, right in zip(broadcasted_left, right_idx):
        if one_sankey == False:
            single_sankey(data[data[xvar]==left][yvar], data[data[xvar]==right][yvar], 
                            xpos=xpos, ax=ax, colorDict=plot_palette, width=width, 
                            leftLabels=leftLabels, rightLabels=rightLabels, 
                            rightColor=rightColor, bar_width=bar_width,
                            align=align, alpha=alpha)
            xpos += 1
        else:
            xpos = 0 + bar_width/2
            width = 1 - bar_width
            single_sankey(data[data[xvar]==left][yvar], data[data[xvar]==right][yvar], 
                            xpos=xpos, ax=ax, colorDict=plot_palette, width=width, 
                            leftLabels=leftLabels, rightLabels=rightLabels, 
                            rightColor=rightColor, bar_width=bar_width,
                            align='edge', alpha=alpha)

    if one_sankey == False:
        sankey_ticks = [f"{left}\n v.s.\n{right}" for left, right in zip(broadcasted_left, right_idx)]
        ax.get_xaxis().set_ticks(np.arange(len(right_idx)))
        ax.get_xaxis().set_ticklabels(sankey_ticks)
    else:
        sankey_ticks = [broadcasted_left[0], right_idx[0]]
        ax.set_xticks([0, 1])
        ax.set_xticklabels(sankey_ticks)
