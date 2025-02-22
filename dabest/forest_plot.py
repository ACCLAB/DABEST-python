"""Creating forest plots from contrast objects."""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/API/forest_plot.ipynb.

# %% auto 0
__all__ = ['load_plot_data', 'check_for_errors', 'get_kwargs', 'color_palette', 'forest_plot']

# %% ../nbs/API/forest_plot.ipynb 5
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from typing import List, Optional, Union


# %% ../nbs/API/forest_plot.ipynb 6
def load_plot_data(
            data: List, 
            effect_size: str = "mean_diff", 
            contrast_type: str = 'delta2',
            idx: Optional[List[int]] = None
) -> List:
    """
    Loads plot data based on specified effect size and contrast type.

    Parameters
    ----------
    contrasts : List
        List of contrast objects.
    effect_size: str
        Type of effect size ('mean_diff', 'median_diff', etc.).
    contrast_type: str
        Type of dabest object to plot ('delta2' or 'mini-meta')
    idx: Optional[List[int]], default=None
        List of indices to select from the contrast objects if delta-delta experiment. 
        If None, only the delta-delta objects are plotted.

    Returns
    -------
    List: Contrast plot data based on specified parameters.
    """
    # Effect size and contrast types
    effect_attr = "hedges_g" if effect_size == 'delta_g' else effect_size
    contrast_attr = {"delta2": "delta_delta", "mini_meta": "mini_meta"}.get(contrast_type)

    if idx is not None:
        bootstraps, differences, bcalows, bcahighs = [], [], [], []
        for current_idx, index_group in enumerate(idx):
            current_contrast = data[current_idx]
            if len(index_group)>0:
                for index in index_group:
                    if index == 2:
                        current_plot_data = getattr(getattr(current_contrast, effect_attr), contrast_attr)
                        bootstraps.append(current_plot_data.bootstraps_delta_delta)
                        differences.append(current_plot_data.difference)
                        bcalows.append(current_plot_data.bca_low)
                        bcahighs.append(current_plot_data.bca_high)
                    elif index == 0 or index == 1:
                        current_plot_data = getattr(current_contrast, effect_attr)
                        bootstraps.append(current_plot_data.results.bootstraps[index])
                        differences.append(current_plot_data.results.difference[index])
                        bcalows.append(current_plot_data.results.bca_low[index])
                        bcahighs.append(current_plot_data.results.bca_high[index])
                    else:
                        raise ValueError("The selected indices must be 0, 1, or 2.")
    else:
        contrast_plot_data = [getattr(getattr(contrast, effect_attr), contrast_attr) for contrast in data]

        attribute_suffix = "weighted_delta" if contrast_type == "mini_meta" else "delta_delta"

        bootstraps = [getattr(result, f"bootstraps_{attribute_suffix}") for result in contrast_plot_data]
        differences = [result.difference for result in contrast_plot_data]
        bcalows = [result.bca_low for result in contrast_plot_data]
        bcahighs = [result.bca_high for result in contrast_plot_data]

    return bootstraps, differences, bcalows, bcahighs

def check_for_errors(
                data,
                idx,
                ax,
                fig_size,
                effect_size,
                horizontal,
                marker_size,
                custom_palette,
                halfviolin_alpha,
                halfviolin_desat,
                labels,
                labels_rotation,
                labels_fontsize,
                title,
                title_fontsize,
                ylabel,
                ylabel_fontsize,
                ylim,
                yticks,
                yticklabels,
                remove_spines,
    ) -> str:

    # Contrasts
    if not isinstance(data, list) or not data:
        raise ValueError("The `data` argument must be a non-empty list of dabest objects.")
    
    ## Check if all contrasts are delta-delta or all are mini-meta
    contrast_type = "delta2" if data[0].delta2 else "mini_meta"
    for contrast in data:
        check_contrast_type = "delta2" if contrast.delta2 else "mini_meta"
        if check_contrast_type != contrast_type:
            raise ValueError("Each dabest object supplied must be the same experimental type (mini-meta or delta-delta)")

    # Idx
    if idx is not None:
        if not isinstance(idx, (tuple, list)):
            raise TypeError("`idx` must be a tuple or list of integers.")
        if contrast_type == "mini_meta":
            raise ValueError("The `idx` argument is not applicable to mini-meta analyses.")

    # Axes
    if ax is not None and not isinstance(ax, plt.Axes):
        raise TypeError("The `ax` must be a `matplotlib.axes.Axes` instance or `None`.")
    
    # Figure size
    if fig_size is not None and not isinstance(fig_size, (tuple, list)):
        raise TypeError("`fig_size` must be a tuple or list of two integers.")

    # Effect size
    effect_size_options = ['mean_diff', 'hedges_g', 'delta_g']
    if not isinstance(effect_size, str) or effect_size not in effect_size_options:
        raise TypeError("The `effect_size` argument must be a string and please choose from the following effect sizes: `mean_diff`, `hedges_g`, or `delta_g`.")
    if data[0].is_mini_meta and effect_size != 'mean_diff':
        raise ValueError("The `effect_size` argument must be `mean_diff` for mini-meta analyses.")
    if data[0].delta2 and effect_size not in ['mean_diff', 'hedges_g', 'delta_g']:
        raise ValueError("The `effect_size` argument must be `mean_diff`, `hedges_g`, or `delta_g` for delta-delta analyses.")

    # Horizontal
    if not isinstance(horizontal, bool):
        raise TypeError("`horizontal` must be a boolean value.")

    # Marker size
    if not isinstance(marker_size, (int, float)) or marker_size <= 0:
        raise TypeError("`marker_size` must be a positive integer or float.")

    # Custom palette
    if custom_palette is not None and not isinstance(custom_palette, (dict, list, str, type(None))):
        raise TypeError("The `custom_palette` must be either a dictionary, list, string, or `None`.")
    if isinstance(custom_palette, dict) and labels is None:
        raise ValueError("The `labels` argument must be provided if `custom_palette` is a dictionary.")


    # Halfviolin alpha and desat
    if not isinstance(halfviolin_alpha, float) or not 0 <= halfviolin_alpha <= 1:
        raise TypeError("`halfviolin_alpha` must be a float between 0 and 1.")
    
    if not isinstance(halfviolin_desat, (float, int)) or not 0 <= halfviolin_desat <= 1:
        raise TypeError("`halfviolin_desat` must be a float between 0 and 1 or an int (1).")
    

    # Contrast labels
    if labels is not None and not all(isinstance(label, str) for label in labels):
        raise TypeError("The `labels` must be a list of strings or `None`.")
    
    number_of_curves_to_plot = sum([len(i) for i in idx]) if idx is not None else len(data)
    if labels is not None and len(labels) != number_of_curves_to_plot:
        raise ValueError("`labels` must match the number of `data` provided.")
    
    if not isinstance(labels_fontsize, (int, float)):
        raise TypeError("`labels_fontsize` must be an integer or float.")
    
    if labels_rotation is not None and (not isinstance(labels_rotation, (int, float)) or not 0 <= labels_rotation <= 360):
        raise TypeError("`labels_rotation` must be an integer or float between 0 and 360.")   

    # Title
    if title is not None and not isinstance(title, str):
        raise TypeError("The `title` argument must be a string.")
    
    if not isinstance(title_fontsize, (int, float)):
        raise TypeError("`title_fontsize` must be an integer or float.")
    
    # Y-label
    if ylabel is not None and not isinstance(ylabel, str):
        raise TypeError("The `ylabel` argument must be a string.")

    if not isinstance(ylabel_fontsize, (int, float)):
        raise TypeError("`ylabel_fontsize` must be an integer or float.")
    
    # Y-lim
    if ylim is not None and not isinstance(ylim, (tuple, list)):
        raise TypeError("`ylim` must be a tuple or list of two floats.")
    if ylim is not None and len(ylim) != 2:
        raise ValueError("`ylim` must be a tuple or list of two floats.")

    # Y-ticks
    if yticks is not None and not isinstance(yticks, (tuple, list)):
        raise TypeError("`yticks` must be a tuple or list of floats.")
    
    # Y-ticklabels
    if yticklabels is not None and not isinstance(yticklabels, (tuple, list)):
        raise TypeError("`yticklabels` must be a tuple or list of strings.")
    
    if yticklabels is not None and not all(isinstance(label, str) for label in yticklabels):
        raise TypeError("`yticklabels` must be a list of strings.")
    
    # Remove spines
    if not isinstance(remove_spines, bool):
        raise TypeError("`remove_spines` must be a boolean value.")
    
    return contrast_type
    

def get_kwargs(
        violin_kwargs,
        zeroline_kwargs,
        horizontal,
        es_marker_kwargs,
        es_errorbar_kwargs,
        marker_size
    ):
    from .misc_tools import merge_two_dicts

    # Violin kwargs
    default_violin_kwargs = {
        "widths": 0.5,
        "showextrema": False,
        "showmedians": False,
        "orientation": 'horizontal' if horizontal else 'vertical',
    }
    if violin_kwargs is None:
        violin_kwargs = default_violin_kwargs
    else:
        violin_kwargs = merge_two_dicts(default_violin_kwargs, violin_kwargs)

    # zeroline kwargs
    default_zeroline_kwargs = {
        "linewidth": 1,
        "color": "black"
    }
    if zeroline_kwargs is None:
        zeroline_kwargs = default_zeroline_kwargs
    else:
        zeroline_kwargs = merge_two_dicts(default_zeroline_kwargs, zeroline_kwargs)

    # Effect size marker kwargs
    default_es_marker_kwargs = {
                'marker': 'o',
                'markersize': marker_size,
                'color': 'black',
                'alpha': 1,
                'zorder': 2,
    }
    if es_marker_kwargs is None:
        es_marker_kwargs = default_es_marker_kwargs
    else:
        es_marker_kwargs = merge_two_dicts(default_es_marker_kwargs, es_marker_kwargs)

    # Effect size error bar kwargs
    default_es_errorbar_kwargs = {
                'color': 'black',
                'lw': 2.5,
                'linestyle': '-',
                'alpha': 1,
                'zorder': 1,
    }
    if es_errorbar_kwargs is None:
        es_errorbar_kwargs = default_es_errorbar_kwargs
    else:
        es_errorbar_kwargs = merge_two_dicts(default_es_errorbar_kwargs, es_errorbar_kwargs)

    return violin_kwargs, zeroline_kwargs, es_marker_kwargs, es_errorbar_kwargs


def color_palette(
        custom_palette, 
        labels, 
        number_of_curves_to_plot,
        halfviolin_desat
    ):
    if custom_palette is not None:
        if isinstance(custom_palette, dict):
            violin_colors = [
                custom_palette.get(c, sns.color_palette()[0]) for c in labels
            ]
        elif isinstance(custom_palette, list):
            violin_colors = custom_palette[: number_of_curves_to_plot]
        elif isinstance(custom_palette, str):
            if custom_palette in plt.colormaps():
                violin_colors = sns.color_palette(custom_palette, number_of_curves_to_plot)
            else:
                raise ValueError(
                    f"The specified `custom_palette` {custom_palette} is not a recognized Matplotlib palette."
                )
    else:
        violin_colors = sns.color_palette(n_colors=number_of_curves_to_plot)
    violin_colors = [sns.desaturate(color, halfviolin_desat) for color in violin_colors]
    return violin_colors


def forest_plot(
    data: list,
    idx: Optional[list[int]] = None,
    ax: Optional[plt.Axes] = None,
    fig_size: tuple[int, int] = None,
    effect_size: str = "mean_diff",
    horizontal: bool = False, 

    marker_size: int = 10,
    custom_palette: Optional[Union[dict, list, str]] = None,
    halfviolin_alpha: float = 0.8,
    halfviolin_desat: float = 1,

    labels: list[str] = None,
    labels_rotation: int = None,
    labels_fontsize: int = 10,
    title: str = None,
    title_fontsize: int = 16,
    ylabel: str = None,
    ylabel_fontsize: int = 12,
    ylim: Optional[list[float, float]] = None,
    yticks: Optional[list[float]] = None,
    yticklabels: Optional[list[str]] = None,
    remove_spines: bool = True,

    violin_kwargs: Optional[dict] = None,
    zeroline_kwargs: Optional[dict] = None,
    es_marker_kwargs: Optional[dict] = None,
    es_errorbar_kwargs: Optional[dict] = None,
)-> plt.Figure:
    """  
    Custom function that generates a forest plot from given contrast objects, suitable for a range of data analysis types, including those from packages like DABEST-python.

    Parameters
    ----------
    data : List
        List of contrast objects.
    idx : Optional[List[int]], default=None
        List of indices to select from the contrast objects if delta-delta experiment. 
        If None, only the delta-delta objects are plotted.
    ax : Optional[plt.Axes], default=None
        Matplotlib Axes object for the plot; creates new if None.
        additional_plotting_kwargs : Optional[dict], default=None
        Further customization arguments for the plot.
    fig_size : Tuple[int, int], default=None
        Figure size for the plot.
    effect_size : str
        Type of effect size to plot (e.g., 'mean_diff', `hedges_g` or 'delta_g').
    horizontal : bool, default=False
        If True, the plot will be horizontal.
    marker_size : int, default=12
        Marker size for plotting effect size dots.
    custom_palette : Optional[Union[dict, list, str]], default=None
        Custom color palette for the plot.
    halfviolin_alpha : float, default=0.8
        Transparency level for violin plots.
    halfviolin_desat : float, default=1
        Saturation level for violin plots.
    labels : List[str]
        Labels for each contrast. If None, defaults to 'Contrast 1', 'Contrast 2', etc.
    labels_rotation : int, default=45 for vertical, 0 for horizontal
        Rotation angle for contrast labels.
    labels_fontsize : int, default=10
        Font size for contrast labels.
    title : str
        Plot title, summarizing the visualized data.
    title_fontsize : int, default=16
        Font size for the plot title.
    ylabel : str
        Label for the y-axis, describing the plotted data or effect size.
    ylabel_fontsize : int, default=12
        Font size for the y-axis label.
    ylim : Optional[Tuple[float, float]]
        Limits for the y-axis.
    yticks : Optional[List[float]]
        Custom y-ticks for the plot.
    yticklabels : Optional[List[str]]
        Custom y-tick labels for the plot.
    remove_spines : bool, default=True
        If True, removes plot spines (except the relevant dependent variable spine).
    violin_kwargs : Optional[dict], default=None
        Additional arguments for violin plot customization.
    zeroline_kwargs : Optional[dict], default=None
        Additional arguments for the zero line customization.
    es_marker_kwargs : Optional[dict], default=None
        Additional arguments for the effect size marker customization.
    es_errorbar_kwargs : Optional[dict], default=None
        Additional arguments for the effect size error bar customization.

    Returns
    -------
    plt.Figure
        The matplotlib figure object with the generated forest plot.
    """
    from .plot_tools import halfviolin

    
    # Check for errors in the input arguments
    contrast_type = check_for_errors(
                            data = data,
                            idx = idx,
                            ax = ax,
                            fig_size = fig_size,
                            effect_size = effect_size,
                            horizontal = horizontal,
                            marker_size = marker_size,
                            custom_palette = custom_palette,
                            halfviolin_alpha = halfviolin_alpha,
                            halfviolin_desat = halfviolin_desat,
                            labels = labels,
                            labels_rotation = labels_rotation,
                            labels_fontsize = labels_fontsize,
                            title = title,
                            title_fontsize = title_fontsize,
                            ylabel = ylabel,
                            ylabel_fontsize = ylabel_fontsize,
                            ylim = ylim,
                            yticks = yticks,
                            yticklabels = yticklabels,
                            remove_spines = remove_spines,
    )

    # Load plot data and extract info
    bootstraps, differences, bcalows, bcahighs = load_plot_data(
                                                        data = data, 
                                                        effect_size = effect_size, 
                                                        contrast_type = contrast_type,
                                                        idx = idx
    )

    # Adjust figure size based on orientation
    number_of_curves_to_plot = sum([len(i) for i in idx]) if idx is not None else len(data)
    if ax is not None:
        fig = ax.figure
    else:
        if fig_size is None:
            fig_size = (4, 1.3 * number_of_curves_to_plot) if horizontal else (1.3 * number_of_curves_to_plot, 4)
        fig, ax = plt.subplots(figsize=fig_size)

    # Get Kwargs
    violin_kwargs, zeroline_kwargs, es_marker_kwargs, es_errorbar_kwargs = get_kwargs(
                                                                                violin_kwargs = violin_kwargs,
                                                                                zeroline_kwargs = zeroline_kwargs,
                                                                                horizontal = horizontal,
                                                                                es_marker_kwargs = es_marker_kwargs,
                                                                                es_errorbar_kwargs = es_errorbar_kwargs,
                                                                                marker_size = marker_size
    )
                                            
    # Plot the violins and make adjustments
    v = ax.violinplot(
        bootstraps, 
        **violin_kwargs
    )
    halfviolin(
            v, 
            alpha = halfviolin_alpha, 
            half = "bottom" if horizontal else "right",
        )
    
    ## Plotting the effect sizes and confidence intervals
    for k in range(1, number_of_curves_to_plot + 1):
        if horizontal:
            ax.plot(differences[k - 1], k, **es_marker_kwargs)  
            ax.plot([bcalows[k - 1], bcahighs[k - 1]], [k, k], **es_errorbar_kwargs) 
        else:
            ax.plot(k, differences[k - 1], **es_marker_kwargs)
            ax.plot([k, k], [bcalows[k - 1], bcahighs[k - 1]], **es_errorbar_kwargs)
    
    # Aesthetic Adjustments
    ## Handle the custom color palette
    violin_colors = color_palette(
                        custom_palette = custom_palette, 
                        labels = labels, 
                        number_of_curves_to_plot = number_of_curves_to_plot,
                        halfviolin_desat = halfviolin_desat
                    )
    
    for patch, color in zip(v["bodies"], violin_colors):
        patch.set_facecolor(color)

    ## Add a zero line to the plot
    if horizontal:
        ax.plot([0, 0], [0, number_of_curves_to_plot+1], **zeroline_kwargs)   
    else:
        ax.plot([0, number_of_curves_to_plot+1], [0, 0], **zeroline_kwargs)

    ## lims
    ### Indepedent variable
    if horizontal:
        ax.set_ylim([0.7, number_of_curves_to_plot + 0.2])
    else:
        ax.set_xlim([0.7, number_of_curves_to_plot + 0.5])

    ## Depedent variable
    if ylim is not None:
        lim_key = ax.set_xlim if horizontal else ax.set_ylim
        lim_key(ylim)

    ## Ticks
    ### Indepedent variable
    lim_key = ax.set_yticks if horizontal else ax.set_xticks
    lim_key(range(1, number_of_curves_to_plot + 1))

    if labels_rotation == None:
        labels_rotation = 0 if horizontal else 45
    if labels is None:
        labels = [f"Contrast {i}" for i in range(1, number_of_curves_to_plot + 1)]
    lim_key = ax.set_yticklabels if horizontal else ax.set_xticklabels
    lim_key(labels, rotation=labels_rotation, fontsize=labels_fontsize, ha="right")

    ### Depedent variable
    if yticks is not None:
        lim_key = ax.set_xticks if horizontal else ax.set_yticks
        lim_key(yticks)

    if yticklabels is not None:
        lim_key = ax.set_xticklabels if horizontal else ax.set_yticklabels
        lim_key(yticklabels)

    ## y-label 
    if ylabel is None:
        effect_attr_map = {
            "mean_diff": "Mean Difference",
            "hedges_g": "Hedges' g",
            "delta_g": "Deltas' g"
        }
        if contrast_type=='delta2' and idx is None and effect_size == "hedges_g":
            ylabel = "Deltas' g"
        elif contrast_type=='delta2' and idx is not None and (effect_size == "delta_g" or effect_size == "hedges_g"):
            ylabel = "Hedges' g with Deltas' g"
        else:
            ylabel = effect_attr_map[effect_size]
    lim_key = ax.set_xlabel if horizontal else ax.set_ylabel
    lim_key(ylabel, fontsize=ylabel_fontsize)

    ## Setting the title
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)

    ## Adjust Spines
    if remove_spines:
        spines = ["top", "right", "left"] if horizontal else ["top", "right", "bottom"]
        ax.spines[spines].set_visible(False)

    ## Invert Y-axis if horizontal 
    if horizontal:
        ax.invert_yaxis()

    return fig
