"""Creating estimation plots."""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/API/plotter.ipynb.

# %% auto 0
__all__ = ['effectsize_df_plotter']

# %% ../nbs/API/plotter.ipynb 4
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import warnings
import logging

# %% ../nbs/API/plotter.ipynb 5
# TODO refactor function name
def effectsize_df_plotter(effectsize_df: object, **plot_kwargs) -> matplotlib.figure.Figure:
    """
    Custom function that creates an estimation plot from an EffectSizeDataFrame.
    Keywords
    --------
    Parameters
    ----------
    effectsize_df
        A `dabest` EffectSizeDataFrame object.
    plot_kwargs
        color_col=None
        raw_marker_size=6, es_marker_size=9,
        swarm_label=None, contrast_label=None, delta2_label=None,
        swarm_ylim=None, contrast_ylim=None, delta2_ylim=None,
        custom_palette=None, swarm_desat=0.5, halfviolin_desat=1,
        halfviolin_alpha=0.8,
        face_color = None,
        bar_label=None, bar_desat=0.8, bar_width = 0.5,bar_ylim = None,
        ci=None, ci_type='bca', err_color=None,
        float_contrast=True,
        show_pairs=True,
        show_delta2=True,
        group_summaries=None,
        fig_size=None,
        dpi=100,
        ax=None,
        gridkey_rows=None, gridkey_kwargs=None,
        swarmplot_kwargs=None,
        violinplot_kwargs=None,
        slopegraph_kwargs=None,
        sankey_kwargs=None,
        reflines_kwargs=None,
        group_summaries_kwargs=None,
        legend_kwargs=None,
        title=None, fontsize_title=16,
        fontsize_rawxlabel=12, fontsize_rawylabel=12,
        fontsize_contrastxlabel=12, fontsize_contrastylabel=12,
        fontsize_delta2label=12,
        swarm_bars=True, swarm_bars_kwargs=None,
        contrast_bars=True, contrast_bars_kwargs=None,
        delta_text=True, delta_text_kwargs=None,
        delta_dot=True, delta_dot_kwargs=None,
		show_baseline_ec=False,
        horizontal=False, horizontal_table_kwargs=None,
        es_marker_kwargs=None, es_errorbar_kwargs=None,
        prop_sample_counts=False, prop_sample_counts_kwargs=None, 
        es_paired_lines=True, es_paired_lines_kwargs=None,
    """
    from .misc_tools import (
        get_params,
        get_kwargs,
        get_color_palette,
        initialize_fig,
        get_plot_groups,
        add_counts_to_ticks,
        extract_contrast_plotting_ticks,
        set_xaxis_ticks_and_lims,
        show_legend,
        gardner_altman_adjustments,
        extract_group_summaries,
        draw_zeroline,
        redraw_dependent_spines,
        redraw_independent_spines
    )
    from .plot_tools import (
        error_bar,
        sankeydiag,
        swarmplot,
        summary_bars_plotter,
        delta_text_plotter,
        delta_dots_plotter,
        slopegraph_plotter,
        plot_minimeta_or_deltadelta_violins,
        effect_size_curve_plotter,
        gridkey_plotter,
        barplotter,
        table_for_horizontal_plots,
        add_counts_to_prop_plots,
        swarm_contrast_bar_plotter
    )

    warnings.filterwarnings(
        "ignore", "This figure includes Axes that are not compatible with tight_layout"
    )

    # Have to disable logging of warning when get_legend_handles_labels()
    # tries to get from slopegraph.
    logging.disable(logging.WARNING)

    # Save rcParams that I will alter, so I can reset back.
    original_rcParams = {}
    _changed_rcParams = ["axes.grid"]
    for parameter in _changed_rcParams:
        original_rcParams[parameter] = plt.rcParams[parameter]

    plt.rcParams["axes.grid"] = False
    ytick_color = plt.rcParams["ytick.color"]

    # Extract parameters and set kwargs
    (swarmplot_kwargs, barplot_kwargs, sankey_kwargs, 
     violinplot_kwargs, slopegraph_kwargs, reflines_kwargs, legend_kwargs,
     group_summaries_kwargs, redraw_axes_kwargs, delta_dot_kwargs, delta_text_kwargs,
     summary_bars_kwargs, swarm_bars_kwargs, contrast_bars_kwargs, table_kwargs,
     gridkey_kwargs, es_marker_kwargs, es_errorbar_kwargs, prop_sample_counts_kwargs, es_paired_lines_kwargs) = get_kwargs(
                                                                                                                    plot_kwargs = plot_kwargs, 
                                                                                                                    ytick_color = ytick_color
    )

    (dabest_obj, plot_data, xvar, yvar, is_paired, effect_size, proportional, 
     all_plot_groups, idx, show_delta2, show_mini_meta, float_contrast, 
     show_pairs, group_summaries, err_color, horizontal, results, halfviolin_alpha, ci_type,
     x1_level, experiment_label, show_baseline_ec, one_sankey, two_col_sankey, asymmetric_side) = get_params(
     																							effectsize_df = effectsize_df, 
                                                                                                plot_kwargs = plot_kwargs,
                                                                                                sankey_kwargs = sankey_kwargs
    )

    # Extract Color palette
    (color_col, bootstraps_color_by_group, n_groups, filled, plot_palette_raw, 
     bar_color, plot_palette_bar, plot_palette_contrast, plot_palette_sankey) = get_color_palette(
                                                                                        plot_kwargs = plot_kwargs, 
                                                                                        plot_data = plot_data, 
                                                                                        xvar = xvar, 
                                                                                        show_pairs = show_pairs,
                                                                                        idx = idx,
                                                                                        all_plot_groups = all_plot_groups,
                                                                                        delta2 = effectsize_df.delta2,
                                                                                        sankey = True if proportional and show_pairs else False,
    )

    # Initialise the figure.
    fig, rawdata_axes, contrast_axes, table_axes = initialize_fig(
                                                            plot_kwargs = plot_kwargs, 
                                                            dabest_obj = dabest_obj, 
                                                            show_delta2 = show_delta2, 
                                                            show_mini_meta = show_mini_meta, 
                                                            is_paired = is_paired, 
                                                            show_pairs = show_pairs, 
                                                            proportional = proportional, 
                                                            float_contrast = float_contrast,
                                                            effect_size_type = effect_size,
                                                            yvar = yvar,
                                                            horizontal = horizontal,
                                                            show_table = table_kwargs['show']
    )
    
    # Plotting the rawdata.
    if show_pairs:  ## Paired plots!
        temp_idx, temp_all_plot_groups = get_plot_groups(
                                                    is_paired = is_paired, 
                                                    idx = idx, 
                                                    proportional = proportional, 
                                                    all_plot_groups = all_plot_groups
        )
        if proportional:  ## Plot the raw data as a set of Sankey Diagrams aligned like barplot.
            sankey_control_test_groups = sankeydiag(
                                                    plot_data,
                                                    xvar = xvar,
                                                    yvar = yvar,
                                                    temp_all_plot_groups = temp_all_plot_groups,
                                                    idx = idx,
                                                    temp_idx = temp_idx,
                                                    palette = plot_palette_sankey,
                                                    ax = rawdata_axes,
                                                    horizontal = horizontal,
                                                    **sankey_kwargs
            )
        else: ## Plot the raw data as a slopegraph.
            slopegraph_plotter(
                dabest_obj = dabest_obj, 
                plot_data = plot_data, 
                xvar = xvar, 
                yvar = yvar, 
                color_col = color_col, 
                plot_palette_raw = plot_palette_raw, 
                slopegraph_kwargs = slopegraph_kwargs, 
                rawdata_axes = rawdata_axes, 
                ytick_color = ytick_color, 
                temp_idx = temp_idx,
                horizontal = horizontal,
                temp_all_plot_groups = temp_all_plot_groups
            )
            
            ## Add delta dots to the contrast axes for paired plots.
            show_delta_dots = plot_kwargs["delta_dot"]
            unavailable_effect_sizes = ["hedges_g", "delta_g", "cohens_d"]
            if show_delta_dots and is_paired and not any([es in effect_size for es in unavailable_effect_sizes]):
                delta_dots_plotter(
                    plot_data = plot_data, 
                    contrast_axes = contrast_axes, 
                    delta_id_col = dabest_obj.id_col, 
                    idx = idx, 
                    xvar = xvar, 
                    yvar = yvar, 
                    is_paired = is_paired, 
                    color_col = color_col, 
                    float_contrast = float_contrast, 
                    plot_palette_raw = plot_palette_raw, 
                    delta_dot_kwargs = delta_dot_kwargs,
                    horizontal = horizontal,
                    )
                
    else:  ## Unpaired plots!
        if proportional:  # Plot the raw data as a barplot.
            barplotter(
                xvar = xvar, 
                yvar = yvar, 
                all_plot_groups = all_plot_groups, 
                rawdata_axes = rawdata_axes, 
                plot_data = plot_data, 
                bar_color = bar_color, 
                plot_palette_bar = plot_palette_bar, 
                color_col = color_col,
                plot_kwargs = plot_kwargs, 
                barplot_kwargs = barplot_kwargs,
                horizontal = horizontal,
            )
        else:   ## Plot the raw data as a swarmplot.
            ## swarmplot() plots swarms based on current size of ax
            ## Therefore, since the ax size for show_mini_meta and show_delta changes later on, there has to be increased jitter
            rawdata_plot = swarmplot(
                            data = plot_data,
                            x = xvar,
                            y = yvar,
                            ax = rawdata_axes,
                            order = all_plot_groups,
                            hue = color_col,
                            palette = plot_palette_raw,
                            zorder = 1,
                            side = asymmetric_side,
                            jitter = 1.25 if show_mini_meta else 1.4 if show_delta2 else 1, # TODO: to make jitter value more accurate and not just a hardcoded eyeball value
                            filled = filled,
                            is_drop_gutter = True,
                            gutter_limit = 0.45,
                            horizontal = horizontal,
                            **swarmplot_kwargs
            )
            if color_col is None:
                rawdata_plot.legend().set_visible(False)
            
        ## Plot the error bars on unpaired plots.
        if group_summaries is not None:
            (group_summaries_method, 
             group_summaries_offset, group_summaries_line_color) = extract_group_summaries(
                                                                            proportional = proportional, 
                                                                            err_color = err_color, 
                                                                            rawdata_axes = rawdata_axes, 
                                                                            asymmetric_side = asymmetric_side if not proportional else None, 
                                                                            horizontal = horizontal, 
                                                                            bootstraps_color_by_group = bootstraps_color_by_group, 
                                                                            plot_palette_raw = plot_palette_raw, 
                                                                            all_plot_groups = all_plot_groups,
                                                                            n_groups = n_groups, 
                                                                            color_col = color_col, 
                                                                            ytick_color = ytick_color, 
                                                                            group_summaries_kwargs = group_summaries_kwargs
            )
            ## Plot the error bar
            error_bar(
                plot_data,
                x = xvar,
                y = yvar,
                offset = group_summaries_offset,
                line_color = group_summaries_line_color,
                type = group_summaries,
                ax = rawdata_axes,
                method = group_summaries_method,
                horizontal = horizontal,
                **group_summaries_kwargs
            )

    # Add the counts to the rawdata axes xticks.
    add_counts_to_ticks(
            plot_data = plot_data, 
            xvar = xvar, 
            yvar = yvar, 
            rawdata_axes = rawdata_axes, 
            plot_kwargs = plot_kwargs,
            flow = sankey_kwargs["flow"],
            horizontal = horizontal,
    )

    # Add counts to prop plots (embedded in the plot bars)
    if proportional and plot_kwargs['prop_sample_counts'] and sankey_kwargs["flow"]:
        add_counts_to_prop_plots(
                        plot_data = plot_data, 
                        xvar = xvar, 
                        yvar = yvar, 
                        rawdata_axes = rawdata_axes, 
                        horizontal = horizontal,
                        is_paired = is_paired,
                        prop_sample_counts_kwargs = prop_sample_counts_kwargs,
        )

    ## Swarm bars
    swarm_bars = plot_kwargs["swarm_bars"]
    if swarm_bars and not proportional and not horizontal: #Currently not supporting swarm bars for horizontal plots (looks weird)
        swarm_contrast_bar_plotter(
            bar_type = 'Swarm',
            axes = [rawdata_axes, contrast_axes],
            bar_kwargs = swarm_bars_kwargs,
            color_col = color_col,
            show_pairs = show_pairs,
            plot_palette_raw = plot_palette_raw,
            idx = idx,
            plot_data = plot_data,
            xvar = xvar,
            yvar = yvar
        )


    # Plot the contrast axes - effect sizes and bootstraps!
    plot_groups = (temp_all_plot_groups if (is_paired == "baseline" and show_pairs and two_col_sankey) 
                   else temp_idx if two_col_sankey 
                   else all_plot_groups
    )

    ## Extract ticks for contrast plot
    (ticks_to_skip, ticks_to_plot, ticks_for_baseline_ec,
     ticks_to_skip_contrast, ticks_to_start_twocol_sankey) = extract_contrast_plotting_ticks(
                                                                                    is_paired = is_paired, 
                                                                                    show_pairs = show_pairs, 
                                                                                    two_col_sankey = two_col_sankey, 
                                                                                    plot_groups = plot_groups,
                                                                                    idx = idx,
                                                                                    sankey_control_group = sankey_control_test_groups[0] if two_col_sankey else None,
    )                                                                  

    ## Adjust contrast tick locations to account for different plotting styles in horizontal plots
    table_axes_ticks_to_plot = ticks_to_plot
    if (horizontal and proportional and not show_pairs) or (horizontal and plot_kwargs["swarm_side"] == "right"):
        ticks_to_plot = [x+0.25 for x in ticks_to_plot]

    ## Plot the bootstraps, then the effect sizes and CIs.
    es_paired_lines = False if float_contrast or not sankey_kwargs["flow"] else plot_kwargs["es_paired_lines"]
    (current_group, current_control,
     current_effsize, contrast_xtick_labels) = effect_size_curve_plotter(
                                                                ticks_to_plot = ticks_to_plot, 
                                                                ticks_for_baseline_ec = ticks_for_baseline_ec,
                                                                results = results, 
                                                                ci_type = ci_type, 
                                                                contrast_axes = contrast_axes, 
                                                                violinplot_kwargs = violinplot_kwargs, 
                                                                halfviolin_alpha = halfviolin_alpha, 
                                                                bootstraps_color_by_group = bootstraps_color_by_group,
                                                                plot_palette_contrast = plot_palette_contrast,
                                                                horizontal = horizontal,
                                                                es_marker_kwargs = es_marker_kwargs,
                                                                es_errorbar_kwargs = es_errorbar_kwargs,
                                                                idx = idx,
                                                                is_paired = is_paired,
                                                                es_paired_lines = es_paired_lines,
																es_paired_lines_kwargs = es_paired_lines_kwargs,
																show_baseline_ec = show_baseline_ec,
    )

    ## Plot mini-meta or delta-delta violin
    delta2_axes = None
    if show_mini_meta or show_delta2:
        delta2_axes, contrast_xtick_labels = plot_minimeta_or_deltadelta_violins(
                                                                show_mini_meta = show_mini_meta, 
                                                                effectsize_df = effectsize_df, 
                                                                ci_type = ci_type, 
                                                                rawdata_axes = rawdata_axes,
                                                                contrast_axes = contrast_axes, 
                                                                violinplot_kwargs = violinplot_kwargs, 
                                                                halfviolin_alpha = halfviolin_alpha, 
                                                                contrast_xtick_labels = contrast_xtick_labels, 
                                                                effect_size = effect_size,
                                                                show_delta2 = show_delta2, 
                                                                plot_kwargs = plot_kwargs, 
                                                                horizontal = horizontal,
                                                                show_pairs = show_pairs,
                                                                es_marker_kwargs = es_marker_kwargs,
                                                                es_errorbar_kwargs = es_errorbar_kwargs
        )
    ## Contrast bars
    contrast_bars = plot_kwargs["contrast_bars"]
    if contrast_bars:
        swarm_contrast_bar_plotter(
                bar_type = 'Contrast',
                axes = [rawdata_axes, contrast_axes],
                bar_kwargs = contrast_bars_kwargs,
                color_col = color_col,
                show_pairs = show_pairs,
                plot_palette_raw = plot_palette_raw,
                idx = idx,
                order = ticks_to_plot,
                results = results,
                horizontal = horizontal,
                diff = (effectsize_df.mini_meta.difference if show_mini_meta 
                        else effectsize_df.delta_delta.difference if show_delta2
                        else None)
        )
    

    ## Delta text
    delta_text = plot_kwargs["delta_text"]
    if delta_text and not horizontal: 
        delta_text_plotter(
                    results = results, 
                    ax_to_plot = contrast_axes, 
                    swarm_plot_ax = rawdata_axes, 
                    ticks_to_plot = ticks_to_plot, 
                    delta_text_kwargs = delta_text_kwargs, 
                    color_col = color_col, 
                    plot_palette_raw = plot_palette_raw, 
                    show_pairs = show_pairs,
                    proportional = proportional, 
                    float_contrast = float_contrast, 
                    show_mini_meta = show_mini_meta, 
                    mini_meta = effectsize_df.mini_meta if show_mini_meta else None, 
                    show_delta2 = show_delta2, 
                    delta_delta = effectsize_df.delta_delta if show_delta2 else None,
                    idx = idx
        )
    
    ## Make sure the contrast_axes x-lims match the rawdata_axes xlims,
    ## and add an extra violinplot tick for delta-delta plot.
    ## Name is xaxis but it is actually y-axis for horizontal plots
    set_xaxis_ticks_and_lims(
                    show_delta2 = show_delta2, 
                    show_mini_meta = show_mini_meta, 
                    rawdata_axes = rawdata_axes, 
                    contrast_axes = contrast_axes, 
                    show_pairs = show_pairs, 
                    float_contrast = float_contrast,
                    ticks_to_skip = ticks_to_skip, 
                    contrast_xtick_labels = contrast_xtick_labels, 
                    plot_kwargs = plot_kwargs,
                    proportional = proportional,
                    horizontal = horizontal,
    )
    # Plot aesthetic adjustments.
    if float_contrast: # For Gardner-Altman (float contrast) plots only.
        gardner_altman_adjustments(
                                effect_size_type = effect_size, 
                                plot_data = plot_data, 
                                xvar = xvar, 
                                yvar = yvar, 
                                current_control = current_control, 
                                current_group = current_group,
                                rawdata_axes = rawdata_axes, 
                                contrast_axes = contrast_axes, 
                                results = results, 
                                current_effsize = current_effsize, 
                                is_paired = is_paired, 
                                one_sankey = one_sankey,
                                reflines_kwargs = reflines_kwargs, 
                                redraw_axes_kwargs = redraw_axes_kwargs, 
        )
    else: # For Cumming plots only.
        ## Add Zero line if lies within the ylim of contrast axes
        draw_zeroline(
                ax = contrast_axes,
                horizontal = horizontal,
                reflines_kwargs = reflines_kwargs,
                extra_delta = True if show_delta2 else False,
        )
        ## Axes independent spine lines
        is_gridkey = True if plot_kwargs["gridkey_rows"] is not None else False
        if not is_gridkey:
            redraw_independent_spines(
                        rawdata_axes = rawdata_axes,
                        contrast_axes = contrast_axes,
                        horizontal = horizontal,
                        two_col_sankey = two_col_sankey,
                        ticks_to_start_twocol_sankey = ticks_to_start_twocol_sankey,
                        idx = idx,
                        is_paired = is_paired,
                        show_pairs = show_pairs,
                        proportional = proportional,
                        ticks_to_skip = ticks_to_skip,
                        temp_idx = temp_idx if is_paired == "baseline" and show_pairs else None,
                        ticks_to_skip_contrast = ticks_to_skip_contrast,
                        extra_delta = True if (show_delta2 or show_mini_meta) else False,
                        redraw_axes_kwargs = redraw_axes_kwargs
            )

    # Modify ylims of axes to flip the plot for horizontal format
    if horizontal:
        if not proportional or (proportional and show_pairs):
            swarm_ylim, contrast_ylim = rawdata_axes.get_ylim(), contrast_axes.get_ylim()
            rawdata_axes.set_ylim(swarm_ylim[1], swarm_ylim[0])
            contrast_axes.set_ylim(contrast_ylim[1], contrast_ylim[0])

        ## Modify the ylim to reduce whitespace in specific plots.
        if show_delta2 or show_mini_meta or (proportional and show_pairs):
            swarm_ylim, contrast_ylim = rawdata_axes.get_ylim(), contrast_axes.get_ylim()
            rawdata_axes.set_ylim(swarm_ylim[0]-0.5, swarm_ylim[1])
            contrast_axes.set_ylim(contrast_ylim[0]-0.5, contrast_ylim[1])

    # Add the dependent axes spines back in.
    redraw_dependent_spines(
        rawdata_axes = rawdata_axes, 
        contrast_axes = contrast_axes, 
        redraw_axes_kwargs = redraw_axes_kwargs, 
        float_contrast = float_contrast, 
        horizontal = horizontal,
        show_delta2 = show_delta2, 
        delta2_axes = delta2_axes
    )

    # Table Axes for horizontal plots
    if horizontal and table_kwargs['show']:
        table_for_horizontal_plots(
                            effectsize_df = effectsize_df,
                            ax = table_axes,
                            contrast_axes = contrast_axes,
                            ticks_to_plot = table_axes_ticks_to_plot, 
                            show_mini_meta = show_mini_meta,
                            show_delta2 = show_delta2,
                            table_kwargs = table_kwargs,
                            ticks_to_skip = ticks_to_skip
        )

    # Gridkey
    gridkey_rows = plot_kwargs["gridkey_rows"]
    if gridkey_rows is not None:
        gridkey_plotter(
                is_paired = is_paired, 
                idx = idx, 
                all_plot_groups = all_plot_groups, 
                gridkey_rows = gridkey_rows, 
                rawdata_axes = rawdata_axes,
                contrast_axes = contrast_axes,
                plot_data = plot_data, 
                xvar = xvar, 
                yvar = yvar, 
                results = results, 
                show_delta2 = show_delta2, 
                show_mini_meta = show_mini_meta, 
                x1_level = x1_level,
                experiment_label = experiment_label,
                float_contrast = float_contrast,
                horizontal = horizontal,
                delta_delta = effectsize_df.delta_delta if show_delta2 else None,
                mini_meta = effectsize_df.mini_meta if show_mini_meta else None,
                effect_size = effect_size,
                gridkey_kwargs = gridkey_kwargs,
        )
    
    # Summary bars
    summary_bars = plot_kwargs["summary_bars"]
    if summary_bars is not None:
        summary_bars_plotter(
                        summary_bars = summary_bars, 
                        results = results, 
                        ax_to_plot = contrast_axes, 
                        float_contrast = float_contrast,
                        summary_bars_kwargs = summary_bars_kwargs, 
                        ci_type = ci_type, 
                        ticks_to_plot = ticks_to_plot, 
                        color_col = color_col,
                        plot_palette_raw = plot_palette_raw, 
                        proportional = proportional, 
                        show_pairs = show_pairs,
                        horizontal = horizontal,
        )
        
    # Legend
    handles, labels = rawdata_axes.get_legend_handles_labels()
    legend_labels = [l for l in labels]
    legend_handles = [h for h in handles]

    if bootstraps_color_by_group is False:
        rawdata_axes.legend().set_visible(False)
        show_legend(
            legend_labels = legend_labels, 
            legend_handles = legend_handles, 
            rawdata_axes = rawdata_axes, 
            contrast_axes = contrast_axes, 
            table_axes = table_axes,
            float_contrast = float_contrast, 
            show_pairs = show_pairs, 
            horizontal = horizontal,
            legend_kwargs = legend_kwargs,
            table_kwargs = table_kwargs
        )

    # Make sure no stray ticks appear!
    rawdata_axes.xaxis.set_ticks_position("bottom")
    rawdata_axes.yaxis.set_ticks_position("left")
    contrast_axes.xaxis.set_ticks_position("bottom")
    if float_contrast is False:
        contrast_axes.yaxis.set_ticks_position("left")

    # Reset rcParams.
    for parameter in _changed_rcParams:
        plt.rcParams[parameter] = original_rcParams[parameter]

    # Return the figure.
    return fig
