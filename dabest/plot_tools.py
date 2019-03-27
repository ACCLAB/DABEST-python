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

        if half is 'left':
            V[:, 0] = np.clip(V[:, 0], -np.inf, mean_vertical)
        if half is 'right':
            V[:, 0] = np.clip(V[:, 0], mean_vertical, np.inf)
        if half is 'bottom':
            V[:, 1] = np.clip(V[:, 1], -np.inf, mean_horizontal)
        if half is 'top':
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
