"""
A set of convenience functions used for producing plots in `dabest`.

Author: Joses W. Ho
Email: joseshowh@gmail.com
License: MIT
"""


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .misc_tools import merge_two_dicts


def halfviolin(v, half = 'right', color = 'k'):
    for b in v['bodies']:
            mVertical = np.mean(b.get_paths()[0].vertices[:, 0])
            mHorizontal = np.mean(b.get_paths()[0].vertices[:, 1])
            vertices = b.get_paths()[0].vertices
            if half is 'left':
                b.get_paths()[0].vertices[:, 0] = np.clip(vertices[:, 0],
                                                    -np.inf, mVertical)
            if half is 'right':
                b.get_paths()[0].vertices[:, 0] = np.clip(vertices[:, 0],
                                                    mVertical, np.inf)
            if half is 'bottom':
                b.get_paths()[0].vertices[:, 1] = np.clip(vertices[:, 1],
                                                    -np.inf, mHorizontal)
            if half is 'top':
                b.get_paths()[0].vertices[:, 1] = np.clip(vertices[:, 1],
                                                    mHorizontal, np.inf)
            b.set_color(color)

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

def tufte_summary_line(df, x, y, type='mean_sd',
                       offset=0.3, ax=None, **kwargs):
    '''Convenience function to plot sumamry statistics (mean and standard
    deviation, or median and 25th & 75th percentiles) for ach group in the `x`
    column of `df`. This style is inspired by Edward Tufte.

    Keywords
    --------
    data: pandas DataFrame.
        This DataFrame should be in 'long' format.

    x, y: string.
        x and y columns to be plotted.

    type: {'mean_sd', 'median_quartiles'}, default 'mean_sd'
        Plots the summary statistics for each group. If 'mean_sd', then the
        mean and standard deviation of each group is plotted as a notched
        line beside each group. If 'median_quantile', then the
        median and 25th and 75th percentiles of each group is plotted
        instead.

    offset: float, default 0.4
        The x-offset of the summary line.

    offset: matplotlib Axes, default None
        If specified, the axes to plot on.

    kwargs: dict, default None
        Dictionary with kwargs passed to `matplotlib.patches.FancyArrow`.
        See docs at
        https://matplotlib.org/api/_as_gen/
        matplotlib.patches.FancyArrow.html#matplotlib.patches.FancyArrow

    '''
    import matplotlib.patches as mpatches

    if ax is None:
        ax = plt.gca()

    means = df.groupby(x)[y].mean()
    sd = df.groupby(x)[y].std()
    lower_sd = means - sd
    upper_sd = means + sd

    medians = df.groupby(x)[y].median()
    quantiles = df.groupby(x)[y].quantile([0.25, 0.75]).unstack()
    lower_quartiles = quantiles[0.25]
    upper_quartiles = quantiles[0.75]

    if type == 'mean_sd':
        central_measures = means
        low = lower_sd
        high = upper_sd
    elif type == 'median_quartiles':
        central_measures = medians
        low = lower_quartiles
        high = upper_quartiles

    total_width = 0.05 # the horizontal span of the line, aka `linewidth`.

    for k, m in enumerate(central_measures):

        kwargs['dx'] = 0
        kwargs['width'] = total_width
        kwargs['head_width'] = total_width
        kwargs['length_includes_head'] = True

        if type == 'mean_sd':
            dy_low = dy_high = sd[k]
        elif type == 'median_quartiles':
            dy_low = m - low[k]
            dy_high = high[k] - m

        arrow = mpatches.FancyArrow(x=offset+k, y=low[k],
                                    dy=dy_low,
                                    head_length=0.3*dy_low,
                                    **kwargs)
        ax.add_patch(arrow)

        arrow = mpatches.FancyArrow(x=offset+k, y=high[k],
                                    dy=-dy_high,
                                    head_length=0.3*dy_high,
                                    **kwargs)
        ax.add_patch(arrow)

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

    keys = kwargs.keys()
    if 'zorder' not in keys:
        kwargs['zorder'] = 5

    if 'lw' not in keys:
        kwargs['lw'] = 2.

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
        kwargs['lw'] = 2
        mean_line = mlines.Line2D([xpos+offset-0.01,
                                    xpos+offset+0.01],
                                    [cm, cm],
                                    **kwargs)
        ax.add_line(mean_line)
