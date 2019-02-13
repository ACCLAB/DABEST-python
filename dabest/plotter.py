#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com


def EffectSizeDataFramePlotter(EffectSizeDataFrame, **plot_kwargs):

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
    import pandas as pd

    from .misc_tools import merge_two_dicts
    from .plot_tools import halfviolin, align_yaxis





    dabest_obj  = EffectSizeDataFrame.dabest_obj
    plot_data   = EffectSizeDataFrame._plot_data
    xvar        = EffectSizeDataFrame.xvar
    yvar        = EffectSizeDataFrame.yvar
    is_paired   = EffectSizeDataFrame.is_paired

    all_plot_groups = dabest_obj._all_plot_groups
    idx             = dabest_obj.idx





    # Disable Gardner-Altman plotting if any of the idxs comprise of more than
    # two groups.
    float_contrast   = plot_kwargs["float_contrast"]
    effect_size_type = EffectSizeDataFrame.effect_size
    if len(idx) > 1 or len(idx[0]) > 2:
        float_contrast = False
    if effect_size_type not in ['mean_diff', 'median_diff']:
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
    show_pairs  = plot_kwargs["show_pairs"]
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


    # Create color palette that will be shared across subplots.
    color_col = plot_kwargs["color_col"]
    if color_col is None:
        color_groups = pd.unique(plot_data[xvar])
    else:
        color_groups = pd.unique(plot_data[color_col])

    n_groups = len(color_groups)

    custom_pal = plot_kwargs["custom_palette"]
    if custom_pal is None:
        plotPal = dict(zip(color_groups,
                           sns.color_palette(n_colors=len(color_groups))
                           )
                       )
    else:
        if isinstance(custom_pal, dict):
            # check that all the keys in custom_pal are found in the
            # color column.
            col_grps = {k for k in color_groups}
            pal_grps = {k for k in custom_pal.keys()}
            not_in_pal = pal_grps.difference(col_grps)
            if len(not_in_pal) > 0:
                err1 = 'The custom palette keys {} '.format(not_in_pal)
                err2 = 'are not found in `{}`. Please check.'.format(color_col)
                errstring = (err1 + err2)
                raise IndexError(errstring)
            plotPal = custom_pal

        elif isinstance(custom_pal, list):
            plotPal = dict(zip(color_groups, custom_pal[0: n_groups]))

        elif isinstance(custom_pal, str):
            # check it is in the list of matplotlib palettes.
            if custom_pal in plt.colormaps():
                plotPal = dict(zip(color_groups,
                                   sns.color_palette(custom_pal, n_groups))
                               )
            else:
                err1 = 'The specified `custom_palette` {}'.format(custom_pal)
                err2 = ' is not a matplotlib palette. Please check.'
                raise ValueError(err1 + err2)


    # Infer the figsize.
    fig_size = plot_kwargs["fig_size"]
    if fig_size is None:
        if color_col is None:
            legend_xspan = 0
        else:
            legend_xspan = 1.5

        all_groups_count = np.sum([len(i) for i in dabest_obj.idx])
        if float_contrast is True:
            height_inches = 4
            each_group_width_inches = 2.5
        else:
            height_inches = 6
            each_group_width_inches = 1.5

        width_inches = (each_group_width_inches * all_groups_count) + legend_xspan
        fsize = (width_inches, height_inches)
    else:
        fsize = fig_size



    # Initialise the figure.
    sns.set(context="talk", style='ticks')
    init_fig_kwargs = dict(figsize=fsize, dpi=plot_kwargs["dpi"])

    if float_contrast is True:
        fig, axx = plt.subplots(ncols=2,
                                gridspec_kw={"width_ratios": [2.5, 1],
                                             "wspace": 0},
                                **init_fig_kwargs)

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
    swarm_ylim = plot_kwargs["swarm_ylim"]
    if swarm_ylim is not None:
        rawdata_axes.set_ylim(swarm_ylim)

    rawdata_axes.set_xlabel("")

    rawdata_axes.set_ylabel(plot_kwargs["swarm_label"])

    sns.swarmplot(data=plot_data, x=xvar, y=yvar,
                  ax=rawdata_axes, order=all_plot_groups, hue=color_col,
                  palette=plotPal, zorder=3, **swarmplot_kwargs)

    # TODO:
    # Plot the gapped line summaries, if this is not a Cumming plot.







    # Plot effect sizes and bootstraps.
    # Take note of where the `control` groups are.
    ticks_to_skip   = [len(i) for i in idx][:-1]
    ticks_to_skip.insert(0, 0)

    # Then obtain the ticks where we have to plot the effect sizes.
    ticks_to_plot = [t for t in range(0, len(all_plot_groups))
                    if t not in ticks_to_skip]

    # Plot the bootstraps, then the effect sizes and CIs.
    es_marker_size   = plot_kwargs["es_marker_size"]
    halfviolin_alpha = plot_kwargs["halfviolin_alpha"]


    results      = EffectSizeDataFrame.results
    xtick_labels = []

    for j, tick in enumerate(ticks_to_plot):
        current_group     = results.test[j]
        current_control   = results.control[j]
        current_color     = plotPal[current_group]
        current_bootstrap = results.bootstraps[j]
        current_effsize   = results.difference[j]
        current_ci_low    = results.bca_low[j]
        current_ci_high   = results.bca_high[j]

        # Create the violinplot.
        v = contrast_axes.violinplot(current_bootstrap,
                                     positions=[tick],
                                     **violinplot_kwargs)
        # Turn the violinplot into half, and color it the same as the swarmplot.
        # Ideally, the alpha (transparency) fo the violin plot should be
        # less than one so the effect size and CIs are visible.
        halfviolin(v, fill_color=current_color, alpha=halfviolin_alpha)

        # Plot the effect size.
        contrast_axes.plot([tick], current_effsize, marker='o', color='k',
                        markersize=es_marker_size)
        # Plot the confidence interval.
        contrast_axes.plot([tick, tick],
                           [current_ci_low, current_ci_high],
                           'k-', linewidth=group_summary_kwargs['lw'])

        xtick_labels.append("{}\nminus\n{}".format(current_group,
                                                   current_control))


    # Make sure the contrast_axes x-lims match the rawdata_axes xlims.
    contrast_axes.set_xticks(rawdata_axes.get_xticks())
    contrast_axes.set_xlim(rawdata_axes.get_xlim())

    # Properly label the contrast ticks.
    for t in ticks_to_skip:
        xtick_labels.insert(t, "")
    contrast_axes.set_xticklabels(xtick_labels)






    if float_contrast is True:
        # For Gardner-Altman plots only.

        # Normalize ylims and despine the floating contrast axes.
        # Check that the effect size is within the swarm ylims.
        if effect_size_type == "mean_diff":
            control_group_summary = plot_data.groupby(xvar)\
                                             .mean().loc[current_control]
            test_group_summary = plot_data.groupby(xvar)\
                                          .mean().loc[current_group]
        elif effect_size_type == "median_diff":
            control_group_summary = plot_data.groupby(xvar)\
                                             .median().loc[current_control]
            test_group_summary = plot_data.groupby(xvar)\
                                          .median().loc[current_group]

        if swarm_ylim is None:
            swarm_ylim = rawdata_axes.get_ylim()

        _, contrast_xlim_max = contrast_axes.get_xlim()

        min_check = float(swarm_ylim[0] - test_group_summary)
        max_check = float(swarm_ylim[1] - test_group_summary)
        difference = float(results.difference[0])
        if min_check <= difference <= max_check:
            pass
        else:
            err1 = 'The mean of the reference group {} '.format(test_group_summary)
            err2 = 'does not fall in the specified `swarm_ylim` {}. '.format(swarm_ylim)
            err3 = 'Please select a `swarm_ylim` that includes the '
            err4 = 'reference mean, or set `float_contrast=False`.'
            err = err1 + err2 + err3 + err4
            raise ValueError(err)

        # Align 0 of contrast_axes to reference group mean of rawdata_axes.
        # If the effect size is positive, shift the contrast axis up.
        rawdata_ylims = np.array(rawdata_axes.get_ylim())
        if current_effsize > 0:
            rightmin, rightmax = rawdata_ylims - current_effsize
        # If the effect size is negative, shift the contrast axis down.
        elif current_effsize < 0:
            rightmin, rightmax = rawdata_ylims + current_effsize

        contrast_axes.set_ylim(rightmin, rightmax)

        # align statfunc(exp) on rawdata_axes with the effect size on contrast_axes.
        align_yaxis(rawdata_axes, test_group_summary,
                    contrast_axes, current_effsize)
        contrast_axes.set_xlim(contrast_xlim_max-1, contrast_xlim_max)

        # Draw summary lines for control and test groups..
        for jj, axx in enumerate([rawdata_axes, contrast_axes]):

            # Draw effect size line.
            if jj == 0:
                ref = control_group_summary
                diff = test_group_summary
                effsize_line_start = 1

            elif jj == 1:
                ref = 0
                diff = difference
                effsize_line_start = contrast_xlim_max-1

            xlimlow, xlimhigh = axx.get_xlim()

            # Draw reference line.
            axx.hlines(ref,            # y-coordinates
                       0, xlimhigh,  # x-coordinates, start and end.
                       **reflines_kwargs)

            # Draw effect size line.
            axx.hlines(diff, effsize_line_start, xlimhigh,
                       **reflines_kwargs)

        # Despine appropriately.
        sns.despine(ax=rawdata_axes,
                    trim=True)
        sns.despine(ax=contrast_axes,
                    left=True, right=False)

    else:
        # For Cumming Plots only.
        sns.despine(ax=rawdata_axes, trim=True)
        sns.despine(ax=contrast_axes, trim=True)






    # Reset plot settings.
    sns.set()

    # Return the figure.
    return fig
