#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com


def EffectSizeDataFramePlotter(EffectSizeDataFrame, **plot_kwargs):

    """
    Custom function that creates an estimation plot from an EffectSizeDataFrame.

    Keywords
    --------
    EffectSizeDataFrame: A `dabest` EffectSizeDataFrame object.

    **plot_kwargs:
        color_col=None

        raw_marker_size=6, es_marker_size=9,

        swarm_label=None, contrast_label=None,
        swarm_ylim=None, contrast_ylim=None,

        custom_palette=None, swarm_desat=0.5, halfviolin_desat=1,
        halfviolin_alpha=0.8, 

        float_contrast=True,
        show_pairs=True,
        group_summaries=None,
        group_summaries_offset=0.1,

        fig_size=None,
        dpi=100,

        swarmplot_kwargs=None,
        violinplot_kwargs=None,
        slopegraph_kwargs=None,
        reflines_kwargs=None,
        group_summary_kwargs=None,
        legend_kwargs=None,
    """

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    from .misc_tools import merge_two_dicts
    from .plot_tools import halfviolin, get_swarm_spans, gapped_lines
    from ._stats_tools.effsize import _compute_standardizers, _compute_hedges_correction_factor

    import logging
    # Have to disable logging of warning when get_legend_handles_labels()
    # tries to get from slopegraph.
    logging.disable(logging.WARNING)

    # Save rcParams that I will alter, so I can reset back.
    original_rcParams = {}
    _changed_rcParams = ['axes.grid']
    for parameter in _changed_rcParams:
        original_rcParams[parameter] = plt.rcParams[parameter]

    plt.rcParams['axes.grid'] = False


    ytick_color = plt.rcParams["ytick.color"]
    axes_facecolor = plt.rcParams['axes.facecolor']

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

    if effect_size_type in ['cliffs_delta']:
        float_contrast = False



    # Disable slopegraph plotting if any of the idxs comprise of more than
    # two groups.
    if np.all([len(i)==2 for i in idx]) is False:
        is_paired = False
    # if paired is False, set show_pairs as False.
    if is_paired is False:
        show_pairs = False
    else:
        show_pairs = plot_kwargs["show_pairs"]


    # Set default kwargs first, then merge with user-dictated ones.
    default_swarmplot_kwargs = {'size': plot_kwargs["raw_marker_size"]}
    if plot_kwargs["swarmplot_kwargs"] is None:
        swarmplot_kwargs = default_swarmplot_kwargs
    else:
        swarmplot_kwargs = merge_two_dicts(default_swarmplot_kwargs,
                                           plot_kwargs["swarmplot_kwargs"])



    # Violinplot kwargs.
    default_violinplot_kwargs = {'widths':0.5, 'vert':True,
                               'showextrema':False, 'showmedians':False}
    if plot_kwargs["violinplot_kwargs"] is None:
        violinplot_kwargs = default_violinplot_kwargs
    else:
        violinplot_kwargs = merge_two_dicts(default_violinplot_kwargs,
                                            plot_kwargs["violinplot_kwargs"])


    # slopegraph kwargs.
    default_slopegraph_kwargs = {'lw':1, 'alpha':0.5}
    if plot_kwargs["slopegraph_kwargs"] is None:
        slopegraph_kwargs = default_slopegraph_kwargs
    else:
        slopegraph_kwargs = merge_two_dicts(slopegraph_kwargs,
                                            plot_kwargs["slopegraph_kwargs"])




    # Zero reference-line kwargs.
    default_reflines_kwargs = {'linestyle':'solid', 'linewidth':0.75,
                                'zorder': 2,
                                'color': ytick_color}
    if plot_kwargs["reflines_kwargs"] is None:
        reflines_kwargs = default_reflines_kwargs
    else:
        reflines_kwargs = merge_two_dicts(default_reflines_kwargs,
                                          plot_kwargs["reflines_kwargs"])



    # Legend kwargs.
    default_legend_kwargs = {'loc': 'upper left', 'frameon': False}
    if plot_kwargs["legend_kwargs"] is None:
        legend_kwargs = default_legend_kwargs
    else:
        legend_kwargs = merge_two_dicts(default_legend_kwargs,
                                        plot_kwargs["legend_kwargs"])


    gs_default = {'mean_sd', 'median_quartiles', None}
    if plot_kwargs["group_summaries"] not in gs_default:
        raise ValueError('group_summaries must be one of'
        ' these: {}.'.format(gs_default) )

    default_group_summary_kwargs = {'zorder': 3, 'lw': 2,
                                    'alpha': 1}
    if plot_kwargs["group_summary_kwargs"] is None:
        group_summary_kwargs = default_group_summary_kwargs
    else:
        group_summary_kwargs = merge_two_dicts(default_group_summary_kwargs,
                                               plot_kwargs["group_summary_kwargs"])



    # Create color palette that will be shared across subplots.
    color_col = plot_kwargs["color_col"]
    if color_col is None:
        color_groups = pd.unique(plot_data[xvar])
        bootstraps_color_by_group = True
    else:
        if color_col not in plot_data.columns:
            raise KeyError("``{}`` is not a column in the data.".format(color_col))
        color_groups = pd.unique(plot_data[color_col])
        bootstraps_color_by_group = False
    if show_pairs:
        bootstraps_color_by_group = False

    # Handle the color palette.
    names = color_groups
    n_groups = len(color_groups)
    custom_pal = plot_kwargs["custom_palette"]
    swarm_desat = plot_kwargs["swarm_desat"]
    contrast_desat = plot_kwargs["halfviolin_desat"]

    if custom_pal is None:
        unsat_colors = sns.color_palette(n_colors=n_groups)
    else:

        if isinstance(custom_pal, dict):
            groups_in_palette = {k: v for k,v in custom_pal.items()
                                 if k in color_groups}

            # # check that all the keys in custom_pal are found in the
            # # color column.
            # col_grps = {k for k in color_groups}
            # pal_grps = {k for k in custom_pal.keys()}
            # not_in_pal = pal_grps.difference(col_grps)
            # if len(not_in_pal) > 0:
            #     err1 = 'The custom palette keys {} '.format(not_in_pal)
            #     err2 = 'are not found in `{}`. Please check.'.format(color_col)
            #     errstring = (err1 + err2)
            #     raise IndexError(errstring)

            names = groups_in_palette.keys()
            unsat_colors = groups_in_palette.values()

        elif isinstance(custom_pal, list):
            unsat_colors = custom_pal[0: n_groups]

        elif isinstance(custom_pal, str):
            # check it is in the list of matplotlib palettes.
            if custom_pal in plt.colormaps():
                unsat_colors = sns.color_palette(custom_pal, n_groups)
            else:
                err1 = 'The specified `custom_palette` {}'.format(custom_pal)
                err2 = ' is not a matplotlib palette. Please check.'
                raise ValueError(err1 + err2)

    swarm_colors = [sns.desaturate(c, swarm_desat) for c in unsat_colors]
    plot_palette_raw = dict(zip(names, swarm_colors))

    contrast_colors = [sns.desaturate(c, contrast_desat) for c in unsat_colors]
    plot_palette_contrast = dict(zip(names, contrast_colors))


    # Infer the figsize.
    fig_size   = plot_kwargs["fig_size"]
    if fig_size is None:
        all_groups_count = np.sum([len(i) for i in dabest_obj.idx])
        if is_paired is True and show_pairs is True:
            frac = 0.75
        else:
            frac = 1
        if float_contrast is True:
            height_inches = 4
            each_group_width_inches = 2.5 * frac
        else:
            height_inches = 6
            each_group_width_inches = 1.5 * frac

        width_inches = (each_group_width_inches * all_groups_count)
        fig_size = (width_inches, height_inches)



    # Initialise the figure.
    # sns.set(context="talk", style='ticks')
    init_fig_kwargs = dict(figsize=fig_size, dpi=plot_kwargs["dpi"])

    # Here, we hardcode some figure parameters.
    if float_contrast is True:
        fig, axx = plt.subplots(ncols=2,
                                gridspec_kw={"width_ratios": [2.5, 1],
                                             "wspace": 0},
                                **init_fig_kwargs)

    else:
        fig, axx = plt.subplots(nrows=2,
                                gridspec_kw={"hspace": 0.3},
                                **init_fig_kwargs)

        # If the contrast axes are NOT floating, create lists to store raw ylims
        # and raw tick intervals, so that I can normalize their ylims later.
        contrast_ax_ylim_low = list()
        contrast_ax_ylim_high = list()
        contrast_ax_ylim_tickintervals = list()

    rawdata_axes  = axx[0]
    contrast_axes = axx[1]

    rawdata_axes.set_frame_on(False)
    contrast_axes.set_frame_on(False)

    redraw_axes_kwargs = {'colors'     : ytick_color,
                          'facecolors' : ytick_color,
                          'lw'      : 1,
                          'zorder'  : 10,
                          'clip_on' : False}

    swarm_ylim = plot_kwargs["swarm_ylim"]

    if swarm_ylim is not None:
        rawdata_axes.set_ylim(swarm_ylim)

    if show_pairs is True:
        # Plot the raw data as a slopegraph.
        # Pivot the long (melted) data.
        if color_col is None:
            pivot_values = yvar
        else:
            pivot_values = [yvar, color_col]
        pivoted_plot_data = pd.pivot(data=plot_data, index=dabest_obj.id_col,
                                     columns=xvar, values=pivot_values)

        for ii, current_tuple in enumerate(idx):
            if len(idx) > 1:
                # Select only the data for the current tuple.
                if color_col is None:
                    current_pair = pivoted_plot_data.reindex(columns=current_tuple)
                else:
                    current_pair = pivoted_plot_data[yvar].reindex(columns=current_tuple)
            else:
                if color_col is None:
                    current_pair = pivoted_plot_data
                else:
                    current_pair = pivoted_plot_data[yvar]

            # Iterate through the data for the current tuple.
            for ID, observation in current_pair.iterrows():
                x_start  = (ii * 2)
                x_points = [x_start, x_start+1]
                y_points = observation.tolist()

                if color_col is None:
                    slopegraph_kwargs['color'] = ytick_color
                else:
                    color_key = pivoted_plot_data[color_col,
                                                  current_tuple[0]].loc[ID]
                    slopegraph_kwargs['color']  = plot_palette_raw[color_key]
                    slopegraph_kwargs['label']  = color_key

                rawdata_axes.plot(x_points, y_points, **slopegraph_kwargs)

        # Set the tick labels, because the slopegraph plotting doesn't.
        rawdata_axes.set_xticks(np.arange(0, len(all_plot_groups)))
        rawdata_axes.set_xticklabels(all_plot_groups)


    else:
        # Plot the raw data as a swarmplot.
        rawdata_plot = sns.swarmplot(data=plot_data, x=xvar, y=yvar,
                                     ax=rawdata_axes,
                                     order=all_plot_groups, hue=color_col,
                                     palette=plot_palette_raw, zorder=1,
                                     **swarmplot_kwargs)

        # Plot the gapped line summaries, if this is not a Cumming plot.
        # Also, we will not plot gapped lines for paired plots. For now.
        group_summaries = plot_kwargs["group_summaries"]
        if float_contrast is False and group_summaries is None:
            group_summaries = "mean_sd"

        if group_summaries is not None:
            # Create list to gather xspans.
            xspans = []
            line_colors = []
            for jj, c in enumerate(rawdata_axes.collections):
                try:
                    _, x_max, _, _ = get_swarm_spans(c)
                    x_max_span = x_max - jj
                    xspans.append(x_max_span)
                except TypeError:
                    # we have got a None, so skip and move on.
                    pass

                if bootstraps_color_by_group is True:
                    line_colors.append(plot_palette_raw[all_plot_groups[jj]])

            if len(line_colors) != len(all_plot_groups):
                line_colors = ytick_color

            gapped_lines(plot_data, x=xvar, y=yvar,
                         # Hardcoded offset...
                         offset=xspans + np.array(plot_kwargs["group_summaries_offset"]),
                         line_color=line_colors,
                         gap_width_percent=1.5,
                         type=group_summaries, ax=rawdata_axes,
                         **group_summary_kwargs)


    # Add the counts to the rawdata axes xticks.
    counts = plot_data.groupby(xvar).count()[yvar]
    ticks_with_counts = []
    for xticklab in rawdata_axes.xaxis.get_ticklabels():
        t = xticklab.get_text()
        N = str(counts.loc[t])

        ticks_with_counts.append("{}\nN = {}".format(t, N))

    rawdata_axes.set_xticklabels(ticks_with_counts)



    # Save the handles and labels for the legend.
    handles, labels = rawdata_axes.get_legend_handles_labels()
    legend_labels  = [l for l in labels]
    legend_handles = [h for h in handles]
    if bootstraps_color_by_group is False:
        rawdata_axes.legend().set_visible(False)

    # Plot effect sizes and bootstraps.
    # Take note of where the `control` groups are.
    ticks_to_skip   = np.cumsum([len(t) for t in idx])[:-1].tolist()
    ticks_to_skip.insert(0, 0)

    # Then obtain the ticks where we have to plot the effect sizes.
    ticks_to_plot = [t for t in range(0, len(all_plot_groups))
                    if t not in ticks_to_skip]

    # Plot the bootstraps, then the effect sizes and CIs.
    es_marker_size   = plot_kwargs["es_marker_size"]
    halfviolin_alpha = plot_kwargs["halfviolin_alpha"]


    results      = EffectSizeDataFrame.results
    contrast_xtick_labels = []

    for j, tick in enumerate(ticks_to_plot):
        current_group     = results.test[j]
        current_control   = results.control[j]
        current_bootstrap = results.bootstraps[j]
        current_effsize   = results.difference[j]
        current_ci_low    = results.bca_low[j]
        current_ci_high   = results.bca_high[j]

        # Create the violinplot.
        v = contrast_axes.violinplot(current_bootstrap,
                                     positions=[tick],
                                     **violinplot_kwargs)
        # Turn the violinplot into half, and color it the same as the swarmplot.
        # Do this only if the color column is not specified.
        # Ideally, the alpha (transparency) fo the violin plot should be
        # less than one so the effect size and CIs are visible.
        if bootstraps_color_by_group is True:
            fc = plot_palette_contrast[current_group]
        else:
            fc = "grey"

        halfviolin(v, fill_color=fc, alpha=halfviolin_alpha)

        # Plot the effect size.
        contrast_axes.plot([tick], current_effsize, marker='o',
                           color=ytick_color,
                           markersize=es_marker_size)
        # Plot the confidence interval.
        contrast_axes.plot([tick, tick],
                           [current_ci_low, current_ci_high],
                           linestyle="-",
                           color=ytick_color,
                           linewidth=group_summary_kwargs['lw'])

        contrast_xtick_labels.append("{}\nminus\n{}".format(current_group,
                                                   current_control))


    # Make sure the contrast_axes x-lims match the rawdata_axes xlims.
    contrast_axes.set_xticks(rawdata_axes.get_xticks())
    if show_pairs is True:
        max_x = contrast_axes.get_xlim()[1]
        rawdata_axes.set_xlim(-0.375, max_x)

    if float_contrast is True:
        contrast_axes.set_xlim(0.5, 1.5)
    else:
        contrast_axes.set_xlim(rawdata_axes.get_xlim())

    # Properly label the contrast ticks.
    for t in ticks_to_skip:
        contrast_xtick_labels.insert(t, "")
    contrast_axes.set_xticklabels(contrast_xtick_labels)


    if bootstraps_color_by_group is False:
        legend_labels_unique = np.unique(legend_labels)
        unique_idx = np.unique(legend_labels, return_index=True)[1]
        legend_handles_unique = (pd.Series(legend_handles).loc[unique_idx]).tolist()

        if len(legend_handles_unique) > 0:
            if float_contrast is True:
                axes_with_legend = contrast_axes
                if show_pairs is True:
                    bta = (1.75, 1.02)
                else:
                    bta = (1.5, 1.02)
            else:
                axes_with_legend = rawdata_axes
                if show_pairs is True:
                    bta = (1.02, 1.)
                else:
                    bta = (1.,1.)
            leg = axes_with_legend.legend(legend_handles_unique,
                                          legend_labels_unique,
                                          bbox_to_anchor=bta,
                                          **legend_kwargs)
            if show_pairs is True:
                for line in leg.get_lines():
                    line.set_linewidth(3.0)



    og_ylim_raw = rawdata_axes.get_ylim()

    if float_contrast is True:
        # For Gardner-Altman plots only.

        # Normalize ylims and despine the floating contrast axes.
        # Check that the effect size is within the swarm ylims.
        if effect_size_type in ["mean_diff", "cohens_d", "hedges_g"]:
            control_group_summary = plot_data.groupby(xvar)\
                                             .mean().loc[current_control, yvar]
            test_group_summary = plot_data.groupby(xvar)\
                                          .mean().loc[current_group, yvar]
        elif effect_size_type == "median_diff":
            control_group_summary = plot_data.groupby(xvar)\
                                             .median().loc[current_control, yvar]
            test_group_summary = plot_data.groupby(xvar)\
                                          .median().loc[current_group, yvar]

        if swarm_ylim is None:
            swarm_ylim = rawdata_axes.get_ylim()

        _, contrast_xlim_max = contrast_axes.get_xlim()

        difference = float(results.difference[0])
        
        if effect_size_type in ["mean_diff", "median_diff"]:
            # Align 0 of contrast_axes to reference group mean of rawdata_axes.
            # If the effect size is positive, shift the contrast axis up.
            rawdata_ylims = np.array(rawdata_axes.get_ylim())
            if current_effsize > 0:
                rightmin, rightmax = rawdata_ylims - current_effsize
            # If the effect size is negative, shift the contrast axis down.
            elif current_effsize < 0:
                rightmin, rightmax = rawdata_ylims + current_effsize
            else:
                rightmin, rightmax = rawdata_ylims

            contrast_axes.set_ylim(rightmin, rightmax)

            og_ylim_contrast = rawdata_axes.get_ylim() - np.array(control_group_summary)

            contrast_axes.set_ylim(og_ylim_contrast)
            contrast_axes.set_xlim(contrast_xlim_max-1, contrast_xlim_max)

        elif effect_size_type in ["cohens_d", "hedges_g"]:
            if is_paired:
                which_std = 1
            else:
                which_std = 0
            temp_control = plot_data[plot_data[xvar] == current_control][yvar]
            temp_test    = plot_data[plot_data[xvar] == current_group][yvar]
            
            stds = _compute_standardizers(temp_control, temp_test)
            if is_paired:
                pooled_sd = stds[1]
            else:
                pooled_sd = stds[0]
            
            if effect_size_type == 'hedges_g':
                gby_count   = plot_data.groupby(xvar).count()
                len_control = gby_count.loc[current_control, yvar]
                len_test    = gby_count.loc[current_group, yvar]
                            
                hg_correction_factor = _compute_hedges_correction_factor(len_control, len_test)
                            
                ylim_scale_factor = pooled_sd / hg_correction_factor
                
            else:
                ylim_scale_factor = pooled_sd
                
            scaled_ylim = ((rawdata_axes.get_ylim() - control_group_summary) / ylim_scale_factor).tolist()

            contrast_axes.set_ylim(scaled_ylim)
            og_ylim_contrast = scaled_ylim

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
                diff = ref + difference
                effsize_line_start = contrast_xlim_max-1.1

            xlimlow, xlimhigh = axx.get_xlim()

            # Draw reference line.
            axx.hlines(ref,            # y-coordinates
                       0, xlimhigh,  # x-coordinates, start and end.
                       **reflines_kwargs)
                        
            # Draw effect size line.
            axx.hlines(diff, effsize_line_start, xlimhigh,
                       **reflines_kwargs)

        # Despine appropriately.
        sns.despine(ax=rawdata_axes,  bottom=True)
        sns.despine(ax=contrast_axes, left=True, right=False)

        # Insert break between the rawdata axes and the contrast axes
        # by re-drawing the x-spine.
        rawdata_axes.hlines(og_ylim_raw[0],                  # yindex
                            rawdata_axes.get_xlim()[0], 1.3, # xmin, xmax
                            **redraw_axes_kwargs)
        rawdata_axes.set_ylim(og_ylim_raw)

        contrast_axes.hlines(contrast_axes.get_ylim()[0],
                             contrast_xlim_max-0.8, contrast_xlim_max,
                             **redraw_axes_kwargs)


    else:
        # For Cumming Plots only.

        # Set custom contrast_ylim, if it was specified.
        if plot_kwargs['contrast_ylim'] is not None:
            custom_contrast_ylim = plot_kwargs['contrast_ylim']

            if len(custom_contrast_ylim) != 2:
                err1 = "Please check `contrast_ylim` consists of "
                err2 = "exactly two numbers."
                raise ValueError(err1 + err2)

            if effect_size_type == "cliffs_delta":
                # Ensure the ylims for a cliffs_delta plot never exceed [-1, 1].
                l = plot_kwargs['contrast_ylim'][0]
                h = plot_kwargs['contrast_ylim'][1]
                low = -1 if l < -1 else l
                high = 1 if h > 1 else h
                contrast_axes.set_ylim(low, high)
            else:
                contrast_axes.set_ylim(custom_contrast_ylim)

        # If 0 lies within the ylim of the contrast axes,
        # draw a zero reference line.
        contrast_axes_ylim = contrast_axes.get_ylim()
        if contrast_axes_ylim[0] < contrast_axes_ylim[1]:
            contrast_ylim_low, contrast_ylim_high = contrast_axes_ylim
        else:
            contrast_ylim_high, contrast_ylim_low = contrast_axes_ylim
        if contrast_ylim_low < 0 < contrast_ylim_high:
            contrast_axes.axhline(y=0, lw=0.75, color=ytick_color)

        # Compute the end of each x-axes line.
        rightend_ticks = np.array([len(i)-1 for i in idx]) + np.array(ticks_to_skip)

        for ax in fig.axes:
            sns.despine(ax=ax, bottom=True)

            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            redraw_axes_kwargs['y'] = ylim[0]

            for k, start_tick in enumerate(ticks_to_skip):
                end_tick = rightend_ticks[k]
                ax.hlines(xmin=start_tick, xmax=end_tick,
                          **redraw_axes_kwargs)

            ax.set_ylim(ylim)
            del redraw_axes_kwargs['y']





    # Place raw axes y-label.
    if plot_kwargs['swarm_label'] is not None:
        swarm_label = plot_kwargs['swarm_label']
    else:
        swarm_label = yvar
    rawdata_axes.set_ylabel(swarm_label)



    # Place contrast axes y-label.
    contrast_label_dict = {'mean_diff'    : "mean difference",
                           'median_diff'  : "median difference",
                           'cohens_d'     : "Cohen's d",
                           'hedges_g'     : "Hedges' g",
                           'cliffs_delta' : "Cliff's delta"}
    default_contrast_label = contrast_label_dict[EffectSizeDataFrame.effect_size]

    if plot_kwargs['contrast_label'] is None:
        if is_paired is True:
            contrast_label = "paired\n{}".format(default_contrast_label)
        else:
            contrast_label = default_contrast_label
        contrast_label = contrast_label.capitalize()
    else:
        contrast_label = plot_kwargs['contrast_label']

    contrast_axes.set_ylabel(contrast_label)
    if float_contrast is True:
        contrast_axes.yaxis.set_label_position("right")


    # Set the rawdata axes labels appropriately
    rawdata_axes.set_ylabel(plot_kwargs["swarm_label"])
    rawdata_axes.set_xlabel("")



    # Because we turned the axes frame off, we also need to draw back
    # the y-spine for both axes.
    og_xlim_raw = rawdata_axes.get_xlim()
    rawdata_axes.vlines(og_xlim_raw[0],
                         og_ylim_raw[0], og_ylim_raw[1],
                         **redraw_axes_kwargs)

    og_xlim_contrast = contrast_axes.get_xlim()

    if float_contrast is True:
        xpos = og_xlim_contrast[1]
    else:
        xpos = og_xlim_contrast[0]

    og_ylim_contrast = contrast_axes.get_ylim()
    contrast_axes.vlines(xpos,
                         og_ylim_contrast[0], og_ylim_contrast[1],
                         **redraw_axes_kwargs)



    # Make sure no stray ticks appear!
    rawdata_axes.xaxis.set_ticks_position('bottom')
    rawdata_axes.yaxis.set_ticks_position('left')
    contrast_axes.xaxis.set_ticks_position('bottom')
    if float_contrast is False:
        contrast_axes.yaxis.set_ticks_position('left')



    # Reset rcParams.
    for parameter in _changed_rcParams:
        plt.rcParams[parameter] = original_rcParams[parameter]



    # Return the figure.
    return fig
