#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com


def EffectSizeDataFrame_plotter(EffectSizeDataFrame, **plot_kwargs):

    """
    Keywords
    --------
        EffectSizeDataFrame: A `dabest` EffectSizeDataFrame object.

        **kwargs:
            color_col=None,

            raw_marker_size=6, es_marker_size=9,

            swarm_label="metric", contrast_label="delta metric",
            swarm_ylim=None, contrast_ylim=None,

            plot_context='talk',
            font_scale=1.,

            custom_palette=None,
            float_contrast=True,
            show_pairs=True,
            show_group_count=True,
            group_summaries="mean_sd",

            ci_linewidth=3,
            summary_linewidth=3,
            multiplot_horizontal_spacing=0.75,
            cumming_vertical_spacing=0.05,

            fig_size=None,
            dpi=100,
            tick_length=10,
            tick_pad=7,

            swarmplot_kwargs=None,
            violinplot_kwargs=None,
            reflines_kwargs=None,
            group_summary_kwargs=None,
            legend_kwargs=None,
            aesthetic_kwargs=None
    """

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    from .misc_tools import merge_two_dicts

    dabest_obj     = EffectSizeDataFrame.dabest_obj

    idx            = dabest_obj.idx
    is_paired      = EffectSizeDataFrame.is_paired

    float_contrast = plot_kwargs["float_contrast"]
    show_pairs     = plot_kwargs["show_pairs"]

    # Disable Gardner-Altman plotting if any of the idxs comprise of more than
    # two groups.
    if len(idx) > 1 or len(idx[0]) > 2:
        float_contrast = False

    # Disable slopegraph plotting if any of the idxs comprise of more than
    # two groups.
    if np.all([len(i)==2 for i in idx]) is False:
        is_paired = False


    # Set default kwargs first, then merge with user-dictated ones.
    default_swarmplot_kwargs = {'size': plot_kwargs["raw_marker_size"]}
    if plot_kwargs["swarmplot_kwargs"] is None:
        swarmplot_kwargs = default_swarmplot_kwargs
    else:
        swarmplot_kwargs = merge_two_dicts(default_swarmplot_kwargs,
                                           plot_kwargs["swarmplot_kwargs"])


    # Violinplot kwargs.
    default_violinplot_kwargs={'widths':0.5, 'vert':True,
                               'showextrema':False, 'showmedians':False}
    if plot_kwargs["violinplot_kwargs"] is None:
        violinplot_kwargs = default_violinplot_kwargs
    else:
        violinplot_kwargs = merge_two_dicts(default_violinplot_kwargs,
                                            plot_kwargs["violinplot_kwargs"])


    # Zero reference-line kwargs.
    default_reflines_kwargs = {'linestyle':'solid', 'linewidth':0.75,
                               'color':'k'}
    if plot_kwargs["reflines_kwargs"] is None:
        reflines_kwargs = default_reflines_kwargs
    else:
        reflines_kwargs = merge_two_dicts(default_reflines_kwargs,
                                          plot_kwargs["reflines_kwargs"])


    # Legend kwargs.
    default_legend_kwargs = {'loc': 'upper left', 'frameon': False,
                             'bbox_to_anchor': (0.95, 1.), 'markerscale': 2}
    if plot_kwargs["legend_kwargs"] is None:
        legend_kwargs = default_legend_kwargs
    else:
        legend_kwargs = merge_two_dicts(default_legend_kwargs,
                                        plot_kwargs["legend_kwargs"])


    # Aesthetic kwargs for sns.set().
    default_aesthetic_kwargs={'context': plot_kwargs["plot_context"],
                               'style': 'ticks',
                              'font_scale': plot_kwargs["font_scale"],
                              'rc': {'axes.linewidth': 1}}
    if plot_kwargs["aesthetic_kwargs"] is None:
        aesthetic_kwargs = default_aesthetic_kwargs
    else:
        aesthetic_kwargs = merge_two_dicts(default_aesthetic_kwargs,
                                           plot_kwargs["aesthetic_kwargs"])


    # if paired is False, set show_pairs as False.
    if is_paired is False:
        show_pairs = False

    gs_default = {'mean_sd', 'median_quartiles', 'None'}
    if plot_kwargs["group_summaries"] not in gs_default:
        raise ValueError('group_summaries must be one of'
        ' these: {}.'.format(gs_default) )

    default_group_summary_kwargs = {'zorder': 5, 'lw': 2,
                                    'color': 'k','alpha': 1}
    if plot_kwargs["group_summary_kwargs"] is None:
        group_summary_kwargs = default_group_summary_kwargs
    else:
        group_summary_kwargs = merge_two_dicts(default_group_summary_kwargs,
                                               plot_kwargs["group_summary_kwargs"])



    # Infer the figsize.
    if plot_kwargs["fig_size"] is None:
        if plot_kwargs["color_col"] is None:
            legend_xspan = 0
        else:
            legend_xspan = 1.5

        all_groups_count = np.sum([len(i) for i in dabest_obj.idx])
        if float_contrast is True:
            height_inches = 4
            each_group_width_inches = 3.5
        else:
            height_inches = 6
            each_group_width_inches = 1.5

        width_inches = (each_group_width_inches * all_groups_count) + legend_xspan
        fsize = (width_inches, height_inches)
    else:
        fsize = plot_kwargs["fig_size"]


    init_fig_kwargs = dict(figsize=fsize, dpi=plot_kwargs["dpi"])
    if float_contrast is True:
        fig, axx = plt.subplots(ncols=2, **init_fig_kwargs)

    else:
        fig, axx = plt.subplots(nrows=2, **init_fig_kwargs)

        # If the contrast axes are NOT floating, create lists to store raw ylims
        # and raw tick intervals, so that I can normalize their ylims later.
        contrast_ax_ylim_low = list()
        contrast_ax_ylim_high = list()
        contrast_ax_ylim_tickintervals = list()

    rawdata_axes  = axx[0]
    contrast_axes = axx[1]

    # Plot the raw data as a swarmplot.
    sns.swarmplot(data=EffectSizeDataFrame._plot_data,
                  x=EffectSizeDataFrame.xvar,
                  y=EffectSizeDataFrame.yvar,
                  ax=rawdata_axes)
    rawdata_axes.set_xlabel("")
    rawdata_axes.set_ylabel(plot_kwargs["swarm_label"])


    return fig
