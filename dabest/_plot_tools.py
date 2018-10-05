#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com
# A set of convenience functions used for producing plots in `dabest`.



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from ._misc_tools import merge_two_dicts



def halfviolin(v, half='right', color='k'):
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

        b.set_color(color)
        b.set_linewidth(0)



def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    # Taken from
    # http://stackoverflow.com/questions/7630778/
    # matplotlib-align-origin-of-right-axis-with-specific-left-axis-value
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

def rotate_ticks(axes, angle=45, alignment='right'):
    for tick in axes.get_xticklabels():
        tick.set_rotation(angle)
        tick.set_horizontalalignment(alignment)



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



def gapped_lines(data, x, y,
                 type='mean_sd',
                 offset=0.3,
                 ax=None,
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

    type: ['mean_sd', 'median_quartiles',], default 'mean_sd'
        Plots the summary statistics for each group. If 'mean_sd', then the
        mean and standard deviation of each group is plotted as a gapped line.
        If 'median_quantiles', then the median and 25th and 75th percentiles of
        each group is plotted instead.

    offset: float, default 0.3
        The x-offset of the mean-sd line.

    ax: matplotlib Axes object, default None
        If a matplotlib Axes object is specified, the gapped lines will be
        plotted in order on this axes. If None, the current axes (plt.gca())
        is used.

    kwargs: dict, default None
        Dictionary with kwargs passed to matplotlib.lines.Line2D
    '''
    import matplotlib.lines as mlines

    if ax is None:
        ax = plt.gca()
    ax_ylims = ax.get_ylim()

    keys = kwargs.keys()
    if 'zorder' not in keys:
        kwargs['zorder'] = 5

    if 'lw' not in keys:
        kwargs['lw'] = 4.

    if 'color' not in keys:
        kwargs['color'] = 'black'

    means = data.groupby(x)[y].mean()
    sd = data.groupby(x)[y].std()
    pooled_sd = sd.mean()
    lower_sd = means - sd
    upper_sd = means + sd

    medians = data.groupby(x)[y].median()
    quantiles = data.groupby(x)[y].quantile([0.25, 0.75]).unstack()
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

    if (lows < ax_ylims[0]).any() or (highs > ax_ylims[1]).any():
        kwargs['clip_on'] = True
    else:
        kwargs['clip_on'] = False

    original_zorder = kwargs['zorder']
    span_color = kwargs['color']
    span_lw = kwargs['lw']
    for xpos, cm in enumerate(central_measures):
        # add vertical span line.
        kwargs['zorder'] = original_zorder
        kwargs['color'] = span_color
        kwargs['lw'] = span_lw
        low_to_high = mlines.Line2D([xpos+offset, xpos+offset],
                                    [lows[xpos], highs[xpos]],
                                      **kwargs)
        ax.add_line(low_to_high)

        # add horzontal central measure line.
        kwargs['zorder'] = 6
        kwargs['color'] = 'white'
        kwargs['lw'] = span_lw * 1.5
        mean_line = mlines.Line2D([xpos+offset-0.01,
                                    xpos+offset+0.01],
                                    [cm, cm],
                                    **kwargs)
        ax.add_line(mean_line)
