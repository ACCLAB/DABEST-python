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
def effectsize_df_plotter(effectsize_df, **plot_kwargs):
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
        horizontal=False, horizontal_table_kwargs=None,
        es_marker_kwargs=None, es_errorbar_kwargs=None,
        prop_sample_counts=False, prop_sample_counts_kwargs=None
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
        Gardner_Altman_Plot_Aesthetic_Adjustments,
        Cumming_Plot_Aesthetic_Adjustments,
        Redraw_Spines,
        extract_group_summaries
    )
    from .plot_tools import (
        error_bar,
        sankeydiag,
        swarmplot,
        swarm_bars_plotter,
        contrast_bars_plotter,
        summary_bars_plotter,
        delta_text_plotter,
        DeltaDotsPlotter,
        slopegraph_plotter,
        plot_minimeta_or_deltadelta_violins,
        effect_size_curve_plotter,
        gridkey_plotter,
        barplotter,
        table_for_horizontal_plots,
        add_counts_to_prop_plots,
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
    (dabest_obj, plot_data, xvar, yvar, is_paired, 
     effect_size, proportional, all_plot_groups, idx, show_delta2, 
     show_mini_meta, float_contrast, show_pairs, effect_size_type, group_summaries, 
     err_color, horizontal, results, halfviolin_alpha, ci_type, x1_level, experiment_label) = get_params(
                                                                                                effectsize_df=effectsize_df, 
                                                                                                plot_kwargs=plot_kwargs,
                                                                                                )

    (swarmplot_kwargs, barplot_kwargs, sankey_kwargs, 
     violinplot_kwargs, slopegraph_kwargs, reflines_kwargs, 
     legend_kwargs, group_summaries_kwargs, redraw_axes_kwargs, delta_dot_kwargs, 
     delta_text_kwargs, summary_bars_kwargs, swarm_bars_kwargs, contrast_bars_kwargs, 
     table_kwargs, gridkey_kwargs, es_marker_kwargs, es_errorbar_kwargs, prop_sample_counts_kwargs) = get_kwargs(
                                                                                                                plot_kwargs=plot_kwargs, 
                                                                                                                ytick_color=ytick_color
                                                                                                                )

    # We also need to extract the `sankey` and `flow` from the kwargs for plotter.py
    # to use for varying different kinds of paired proportional plots
    # We also don't want to pop the parameter from the kwargs
    one_sankey = (
        False if is_paired is not None else None
    )  # Flag to indicate if only one sankey is plotted.
    two_col_sankey = (
        True if proportional and not one_sankey and sankey_kwargs["sankey"] and not sankey_kwargs["flow"] else False
    )

    # Extract Color palette
    (color_col, bootstraps_color_by_group, n_groups, filled, plot_palette_raw, 
     bar_color, plot_palette_bar, plot_palette_contrast, plot_palette_sankey) = get_color_palette(
                                                                                        plot_kwargs=plot_kwargs, 
                                                                                        plot_data=plot_data, 
                                                                                        xvar=xvar, 
                                                                                        show_pairs=show_pairs,
                                                                                        idx=idx,
                                                                                        all_plot_groups=all_plot_groups
                                                                                        )

    # Initialise the figure.
    fig, rawdata_axes, contrast_axes, table_axes = initialize_fig(
                                                            plot_kwargs=plot_kwargs, 
                                                            dabest_obj=dabest_obj, 
                                                            show_delta2=show_delta2, 
                                                            show_mini_meta=show_mini_meta, 
                                                            is_paired=is_paired, 
                                                            show_pairs=show_pairs, 
                                                            proportional=proportional, 
                                                            float_contrast=float_contrast,
                                                            effect_size_type=effect_size_type,
                                                            yvar=yvar,
                                                            horizontal=horizontal,
                                                            show_table = table_kwargs['show']
                                                            )
    
    # Plotting the rawdata.
    if show_pairs:
        temp_idx, temp_all_plot_groups = get_plot_groups(
                                                    is_paired=is_paired, 
                                                    idx=idx, 
                                                    proportional=proportional, 
                                                    all_plot_groups=all_plot_groups
                                                    )
        if not proportional:
            # Plot the raw data as a slopegraph.
            slopegraph_plotter(
                dabest_obj=dabest_obj, 
                plot_data=plot_data, 
                xvar=xvar, 
                yvar=yvar, 
                color_col=color_col, 
                plot_palette_raw=plot_palette_raw, 
                slopegraph_kwargs=slopegraph_kwargs, 
                rawdata_axes=rawdata_axes, 
                ytick_color=ytick_color, 
                temp_idx=temp_idx,
                horizontal=horizontal
                )
            
            # DELTA PTS ON CONTRAST PLOT WIP
            show_delta_dots = plot_kwargs["delta_dot"]
            if show_delta_dots and is_paired is not None:
                DeltaDotsPlotter(
                    plot_data=plot_data, 
                    contrast_axes=contrast_axes, 
                    delta_id_col=dabest_obj.id_col, 
                    idx=idx, 
                    xvar=xvar, 
                    yvar=yvar, 
                    is_paired=is_paired, 
                    color_col=color_col, 
                    float_contrast=float_contrast, 
                    plot_palette_raw=plot_palette_raw, 
                    delta_dot_kwargs=delta_dot_kwargs,
                    horizontal=horizontal,
                    )

            # Set the tick labels, because the slopegraph plotting doesn't.
            if horizontal:
                rawdata_axes.set_yticks(np.arange(0, len(temp_all_plot_groups)))
                rawdata_axes.set_yticklabels(temp_all_plot_groups)
            else:
                rawdata_axes.set_xticks(np.arange(0, len(temp_all_plot_groups)))
                rawdata_axes.set_xticklabels(temp_all_plot_groups)

        else:
            # Plot the raw data as a set of Sankey Diagrams aligned like barplot.
            sankey_control_group, sankey_test_group = sankeydiag(
                                                            plot_data,
                                                            xvar=xvar,
                                                            yvar=yvar,
                                                            temp_all_plot_groups=temp_all_plot_groups,
                                                            idx=idx,
                                                            temp_idx=temp_idx,
                                                            palette=plot_palette_sankey,
                                                            ax=rawdata_axes,
                                                            horizontal=horizontal,
                                                            **sankey_kwargs
                                                            )
    else:
        if not proportional:
            # Plot the raw data as a swarmplot.
            asymmetric_side = (
                plot_kwargs["swarm_side"]
                if plot_kwargs["swarm_side"] is not None
                else "right" if not horizontal
                else "left"
            )  # Default asymmetric side is right

            # swarmplot() plots swarms based on current size of ax
            # Therefore, since the ax size for mini_meta and show_delta changes later on, there has to be increased jitter
            rawdata_plot = swarmplot(
                            data=plot_data,
                            x=xvar,
                            y=yvar,
                            ax=rawdata_axes,
                            order=all_plot_groups,
                            hue=color_col,
                            palette=plot_palette_raw,
                            zorder=1,
                            side=asymmetric_side,
                            jitter=1.25 if show_mini_meta else 1.4 if show_delta2 else 1, # TODO: to make jitter value more accurate and not just a hardcoded eyeball value
                            filled=filled,
                            is_drop_gutter=True,
                            gutter_limit=0.45,
                            horizontal=horizontal,
                            **swarmplot_kwargs
            )
            if color_col is None:
                rawdata_plot.legend().set_visible(False)

        else:
            # Plot the raw data as a barplot.
            barplotter(
                xvar=xvar, 
                yvar=yvar, 
                all_plot_groups=all_plot_groups, 
                rawdata_axes=rawdata_axes, 
                plot_data=plot_data, 
                bar_color=bar_color, 
                plot_palette_bar=plot_palette_bar, 
                plot_kwargs=plot_kwargs, 
                barplot_kwargs=barplot_kwargs,
                horizontal=horizontal,
                )
            
        # Plot the error bars.
        if group_summaries is not None:
            (group_summaries_method, group_summaries_offset, group_summaries_line_color) = extract_group_summaries(
                                                                                                    proportional=proportional, 
                                                                                                    err_color=err_color, 
                                                                                                    rawdata_axes=rawdata_axes, 
                                                                                                    asymmetric_side=asymmetric_side if not proportional else None, 
                                                                                                    horizontal=horizontal, 
                                                                                                    bootstraps_color_by_group=bootstraps_color_by_group, 
                                                                                                    plot_palette_raw=plot_palette_raw, 
                                                                                                    all_plot_groups=all_plot_groups,
                                                                                                    n_groups=n_groups, 
                                                                                                    color_col=color_col, 
                                                                                                    ytick_color=ytick_color, 
                                                                                                    group_summaries_kwargs=group_summaries_kwargs
                                                                                                    )
            # Plot
            error_bar(
                plot_data,
                x=xvar,
                y=yvar,
                offset=group_summaries_offset,
                line_color=group_summaries_line_color,
                type=group_summaries,
                ax=rawdata_axes,
                method=group_summaries_method,
                horizontal=horizontal,
                **group_summaries_kwargs
                )

    # Add the counts to the rawdata axes xticks.
    add_counts_to_ticks(
            plot_data=plot_data, 
            xvar=xvar, 
            yvar=yvar, 
            rawdata_axes=rawdata_axes, 
            plot_kwargs=plot_kwargs,
            flow = sankey_kwargs["flow"],
            horizontal=horizontal,
            )

    # Add counts to prop plots
    if proportional and plot_kwargs['prop_sample_counts'] and sankey_kwargs["flow"]:
        add_counts_to_prop_plots(
                        plot_data=plot_data, 
                        xvar=xvar, 
                        yvar=yvar, 
                        rawdata_axes=rawdata_axes, 
                        horizontal=horizontal,
                        is_paired = is_paired,
                        prop_sample_counts_kwargs=prop_sample_counts_kwargs,
                        )

    # Plot effect sizes and bootstraps.
    plot_groups = (temp_all_plot_groups if (is_paired == "baseline" and show_pairs and two_col_sankey) 
                   else temp_idx if two_col_sankey 
                   else all_plot_groups
                   )

    (ticks_to_skip, ticks_to_plot, 
     ticks_to_skip_contrast, ticks_to_start_twocol_sankey) = extract_contrast_plotting_ticks(
                                                                                    is_paired=is_paired, 
                                                                                    show_pairs=show_pairs, 
                                                                                    two_col_sankey=two_col_sankey, 
                                                                                    plot_groups=plot_groups,
                                                                                    idx=idx,
                                                                                    sankey_control_group=sankey_control_group if two_col_sankey else None,
                                                                                    )

    # Adjust contrast tick locations to account for different plotting styles in horizontal plots
    table_axes_ticks_to_plot = ticks_to_plot
    if (horizontal and proportional and not show_pairs) or (horizontal and plot_kwargs["swarm_side"] == "right"):
        ticks_to_plot = [x+0.25 for x in ticks_to_plot]

    # Plot the bootstraps, then the effect sizes and CIs.
    (current_group, current_control, 
     current_effsize, contrast_xtick_labels) = effect_size_curve_plotter(
                                                                    ticks_to_plot=ticks_to_plot, 
                                                                    results=results, 
                                                                    ci_type=ci_type, 
                                                                    contrast_axes=contrast_axes, 
                                                                    violinplot_kwargs=violinplot_kwargs, 
                                                                    halfviolin_alpha=halfviolin_alpha, 
                                                                    ytick_color=ytick_color, 
                                                                    bootstraps_color_by_group=bootstraps_color_by_group,
                                                                    plot_palette_contrast=plot_palette_contrast,
                                                                    horizontal=horizontal,
                                                                    es_marker_kwargs=es_marker_kwargs,
                                                                    es_errorbar_kwargs=es_errorbar_kwargs
                                                                    )

    # Plot mini-meta or delta-delta violin
    delta2_axes = None
    if show_mini_meta or show_delta2:
        delta2_axes, contrast_xtick_labels = plot_minimeta_or_deltadelta_violins(
                                                                show_mini_meta=show_mini_meta, 
                                                                effectsize_df=effectsize_df, 
                                                                ci_type=ci_type, 
                                                                rawdata_axes=rawdata_axes,
                                                                contrast_axes=contrast_axes, 
                                                                violinplot_kwargs=violinplot_kwargs, 
                                                                halfviolin_alpha=halfviolin_alpha, 
                                                                ytick_color=ytick_color, 
                                                                contrast_xtick_labels=contrast_xtick_labels, 
                                                                effect_size=effect_size,
                                                                show_delta2=show_delta2, 
                                                                plot_kwargs=plot_kwargs, 
                                                                horizontal=horizontal,
                                                                es_marker_kwargs=es_marker_kwargs,
                                                                es_errorbar_kwargs=es_errorbar_kwargs
                                                                )
    
    # Make sure the contrast_axes x-lims match the rawdata_axes xlims,
    # and add an extra violinplot tick for delta-delta plot.
    # Name is xaxis but it is actually y-axis for horizontal plots
    set_xaxis_ticks_and_lims(
                        show_delta2=show_delta2, 
                        show_mini_meta=show_mini_meta, 
                        rawdata_axes=rawdata_axes, 
                        contrast_axes=contrast_axes, 
                        show_pairs=show_pairs, 
                        float_contrast=float_contrast,
                        ticks_to_skip=ticks_to_skip, 
                        contrast_xtick_labels=contrast_xtick_labels, 
                        plot_kwargs=plot_kwargs,
                        proportional=proportional,
                        horizontal=horizontal,
                        )

    # Plot aesthetic adjustments.
    if float_contrast and not horizontal:
        # For Gardner-Altman plots only.
        Gardner_Altman_Plot_Aesthetic_Adjustments(
                                            effect_size_type=effect_size_type, 
                                            plot_data=plot_data, 
                                            xvar=xvar, 
                                            yvar=yvar, 
                                            current_control=current_control, 
                                            current_group=current_group,
                                            rawdata_axes=rawdata_axes, 
                                            contrast_axes=contrast_axes, 
                                            results=results, 
                                            current_effsize=current_effsize, 
                                            is_paired=is_paired, 
                                            one_sankey=one_sankey,
                                            reflines_kwargs=reflines_kwargs, 
                                            redraw_axes_kwargs=redraw_axes_kwargs, 
                                            )
    else:
        # For Cumming Plots only.
        skip_redraw_lines = True if plot_kwargs["gridkey_rows"] is not None else False
        Cumming_Plot_Aesthetic_Adjustments(
                                    contrast_axes=contrast_axes, 
                                    reflines_kwargs=reflines_kwargs, 
                                    is_paired=is_paired, 
                                    show_pairs=show_pairs, 
                                    two_col_sankey=two_col_sankey, 
                                    idx=idx, 
                                    ticks_to_start_twocol_sankey=ticks_to_start_twocol_sankey,
                                    proportional=proportional, 
                                    ticks_to_skip=ticks_to_skip, 
                                    temp_idx=temp_idx if is_paired == "baseline" and show_pairs else None, 
                                    rawdata_axes=rawdata_axes, 
                                    redraw_axes_kwargs=redraw_axes_kwargs,
                                    ticks_to_skip_contrast=ticks_to_skip_contrast,
                                    show_delta2 = show_delta2,
                                    show_mini_meta = show_mini_meta,
                                    horizontal=horizontal,
                                    skip_redraw_lines=skip_redraw_lines,
                                    )
    
    # Add the dependent axes spines back in.
    Redraw_Spines(
        rawdata_axes=rawdata_axes, 
        contrast_axes=contrast_axes, 
        redraw_axes_kwargs=redraw_axes_kwargs, 
        float_contrast=float_contrast, 
        horizontal=horizontal,
        show_delta2=show_delta2, 
        delta2_axes=delta2_axes
        )

    # Swarm bars WIP
    swarm_bars = plot_kwargs["swarm_bars"]
    if swarm_bars and not proportional and not horizontal:
        swarm_bars_plotter(
                    plot_data=plot_data, 
                    xvar=xvar, 
                    yvar=yvar, 
                    ax=rawdata_axes, 
                    swarm_bars_kwargs=swarm_bars_kwargs, 
                    color_col=color_col, 
                    plot_palette_raw=plot_palette_raw,
                    is_paired=is_paired,
                    idx = idx,
                    )

    # Contrast bars WIP
    contrast_bars = plot_kwargs["contrast_bars"]
    if contrast_bars:
        contrast_bars_plotter(
                        results=results, 
                        ax_to_plot=contrast_axes, 
                        swarm_plot_ax=rawdata_axes,
                        ticks_to_plot=ticks_to_plot, 
                        contrast_bars_kwargs=contrast_bars_kwargs, 
                        color_col=color_col, 
                        plot_palette_raw=plot_palette_raw, 
                        show_mini_meta=show_mini_meta, 
                        mini_meta_delta=effectsize_df.mini_meta_delta if show_mini_meta else None, 
                        show_delta2=show_delta2, 
                        delta_delta=effectsize_df.delta_delta if show_delta2 else None, 
                        is_paired=is_paired,
                        horizontal=horizontal,
                        idx=idx
                        )

    # Summary bars WIP
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
                        is_paired = is_paired,
                        horizontal = horizontal,
                        )
    # Delta text WIP
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
                    is_paired = is_paired,
                    proportional = proportional, 
                    float_contrast = float_contrast, 
                    show_mini_meta = show_mini_meta, 
                    mini_meta_delta = effectsize_df.mini_meta_delta if show_mini_meta else None, 
                    show_delta2 = show_delta2, 
                    delta_delta = effectsize_df.delta_delta if show_delta2 else None,
                    idx = idx
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
                            )

    # GRIDKEY
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
                mini_meta_delta = effectsize_df.mini_meta_delta if show_mini_meta else None,
                effect_size = effect_size,
                gridkey_kwargs = gridkey_kwargs,
                )
        

    # Legend
    handles, labels = rawdata_axes.get_legend_handles_labels()
    legend_labels = [l for l in labels]
    legend_handles = [h for h in handles]

    if bootstraps_color_by_group is False and not effectsize_df.delta2:
        rawdata_axes.legend().set_visible(False)
        show_legend(
            legend_labels=legend_labels, 
            legend_handles=legend_handles, 
            rawdata_axes=rawdata_axes, 
            contrast_axes=contrast_axes, 
            table_axes=table_axes,
            float_contrast=float_contrast, 
            show_pairs=show_pairs, 
            horizontal=horizontal,
            legend_kwargs=legend_kwargs,
            table_kwargs=table_kwargs
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
