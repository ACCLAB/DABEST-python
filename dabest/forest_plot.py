"""Creating forest plots from contrast objects."""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/API/forest_plot.ipynb.

# %% auto 0
__all__ = ['load_plot_data', 'check_for_errors', 'get_kwargs', 'color_palette', 'forest_plot']

# %% ../nbs/API/forest_plot.ipynb 5
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from typing import List, Optional, Union
import numpy as np
import matplotlib.axes as axes
import matplotlib.patches as mpatches

# %% ../nbs/API/forest_plot.ipynb 6
def load_plot_data(
            data: List, 
            effect_size: str = "mean_diff", 
            contrast_type: str = None,
            ci_type: str = "bca",
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
        Type of dabest object to plot ('delta2' or 'mini-meta' or 'delta').
    ci_type: str
        Type of confidence interval to plot ('bca' or 'pct')
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

    # Testing
    if idx is not None:
        bootstraps, differences, bcalows, bcahighs = [], [], [], []
        for current_idx, index_group in enumerate(idx):
            current_contrast = data[current_idx]
            if len(index_group)>0:
                for index in index_group:
                    current_plot_data = getattr(current_contrast, effect_attr)
                    if contrast_type == 'delta2':
                        if index == 2:
                            current_plot_data = getattr(current_plot_data, contrast_attr)
                            bootstrap_name, index_val = "bootstraps_delta_delta", 0
                        elif index == 0 or index == 1:
                            bootstrap_name, index_val = "bootstraps", index
                        else:
                            raise ValueError("The selected indices must be 0, 1, or 2.")
                    elif contrast_type == "mini_meta":
                        num_of_groups = len(getattr(current_contrast, effect_attr).results)
                        if index == num_of_groups:
                            current_plot_data = getattr(getattr(current_contrast, effect_attr), contrast_attr)
                            bootstrap_name, index_val = "bootstraps_weighted_delta", 0
                        elif index < num_of_groups:
                            bootstrap_name, index_val = "bootstraps", index
                        else:
                            msg1 = "There are only {} groups (starting from zero) in this dabest object. ".format(num_of_groups)
                            msg2 = "The idx given is {}.".format(index)
                            raise ValueError(msg1+msg2)        
                    else: # contrast_type == 'delta'
                        bootstrap_name, index_val = "bootstraps", index              

                    bootstraps.append(getattr(current_plot_data.results, bootstrap_name)[index_val])
                    differences.append(current_plot_data.results.difference[index_val])
                    bcalows.append(current_plot_data.results.get(ci_type+'_low')[index_val])
                    bcahighs.append(current_plot_data.results.get(ci_type+'_high')[index_val])    
    else:
        if contrast_type == 'delta':
            contrast_plot_data = [getattr(contrast, effect_attr)  for contrast in data]
            bootstraps_nested = [result.results.bootstraps.to_list() for result in contrast_plot_data]
            differences_nested = [result.results.difference.to_list() for result in contrast_plot_data]
            bcalows_nested = [result.results.get(ci_type+'_low').to_list() for result in contrast_plot_data]
            bcahighs_nested = [result.results.get(ci_type+'_high').to_list() for result in contrast_plot_data]
            
            bootstraps = [element for innerList in bootstraps_nested for element in innerList]
            differences = [element for innerList in differences_nested for element in innerList]
            bcalows = [element for innerList in bcalows_nested for element in innerList]
            bcahighs = [element for innerList in bcahighs_nested for element in innerList]

        else: # contrast_type == 'delta2' or 'mini_meta'
            contrast_plot_data = [getattr(getattr(contrast, effect_attr), contrast_attr) for contrast in data]
            attribute_suffix = "weighted_delta" if contrast_type == "mini_meta" else "delta_delta"

            bootstraps = [getattr(result, f"bootstraps_{attribute_suffix}") for result in contrast_plot_data]
            differences = [result.difference for result in contrast_plot_data]
            bcalows = [result.results.get(ci_type+'_low')[0] for result in contrast_plot_data]
            bcahighs = [result.results.get(ci_type+'_high')[0] for result in contrast_plot_data]

    return bootstraps, differences, bcalows, bcahighs

def check_for_errors(**kwargs):
    data = kwargs.get('data')
    # Contrasts
    if not isinstance(data, list) or not data:
        raise ValueError("The `data` argument must be a non-empty list of dabest objects.")
    
    ## Check if all contrasts are delta-delta or all are mini-meta
    contrast_type = ("delta2" if data[0].delta2 
                     else "mini_meta" if data[0].is_mini_meta
                     else "delta"
                    )

    # contrast_type = "delta2" if data[0].delta2 else "mini_meta"
    for contrast in data:
        check_contrast_type = ("delta2" if contrast.delta2 
                               else "mini_meta" if contrast.is_mini_meta
                               else "delta"
                              )
        if check_contrast_type != contrast_type:
            raise ValueError("Each dabest object supplied must be the same experimental type (mini-meta or delta-delta or neither.)")

    # Idx
    idx = kwargs.get('idx')
    effect_size = kwargs.get('effect_size')
    if idx is not None:
        if not isinstance(idx, (tuple, list)):
            raise TypeError("`idx` must be a tuple or list of integers.")

        msg1 = "The `idx` argument must have the same length as the number of dabest objects. "
        msg2 = "E.g., If two dabest objects are supplied, there should be two lists within `idx`. "
        msg3 = "E.g., `idx` = [[1,2],[0,1]]."
        _total = 0
        for _group in idx:
            if isinstance(_group, int | float):
                raise ValueError(msg1+msg2+msg3)
            else:
                _total += 1
        if _total != len(data):
            raise ValueError(msg1+msg2+msg3)
        
    if idx is not None:
        number_of_curves_to_plot = sum([len(i) for i in idx])
    else:
        if contrast_type == 'delta':
            number_of_curves_to_plot = sum(len(getattr(i, effect_size).results) for i in data)
        else:
            number_of_curves_to_plot = len(data)

    # Axes
    ax = kwargs.get('ax')
    fig_size = kwargs.get('fig_size')
    if ax is not None and not isinstance(ax, plt.Axes):
        raise TypeError("The `ax` must be a `matplotlib.axes.Axes` instance or `None`.")
    
    # Figure size
    if fig_size is not None and not isinstance(fig_size, (tuple, list)):
        raise TypeError("`fig_size` must be a tuple or list of two positive integers.")

    # Effect size
    effect_size_options = ['mean_diff', 'hedges_g', 'delta_g']
    if not isinstance(effect_size, str) or effect_size not in effect_size_options:
        raise TypeError("The `effect_size` argument must be a string and please choose from the following effect sizes: `mean_diff`, `hedges_g`, or `delta_g`.")
    if data[0].is_mini_meta and effect_size != 'mean_diff':
        raise ValueError("The `effect_size` argument must be `mean_diff` for mini-meta analyses.")
    if data[0].delta2 and effect_size not in ['mean_diff', 'hedges_g', 'delta_g']:
        raise ValueError("The `effect_size` argument must be `mean_diff`, `hedges_g`, or `delta_g` for delta-delta analyses.")
    
    # CI type
    ci_type = kwargs.get('ci_type')
    if ci_type not in ('bca', 'pct'):
        raise TypeError("`ci_type` must be either 'bca' or 'pct'.")

    # Horizontal
    horizontal = kwargs.get('horizontal')
    if not isinstance(horizontal, bool):
        raise TypeError("`horizontal` must be a boolean value.")

    # Marker size
    marker_size = kwargs.get('marker_size')
    if not isinstance(marker_size, (int, float)) or marker_size <= 0:
        raise TypeError("`marker_size` must be a positive integer or float.")

    # Custom palette
    custom_palette = kwargs.get('custom_palette')
    labels = kwargs.get('labels')
    if custom_palette is not None and not isinstance(custom_palette, (dict, list, tuple, str, type(None))):
        raise TypeError("The `custom_palette` must be either a dictionary, list, string, or `None`.")
    if isinstance(custom_palette, dict) and labels is None:
        raise ValueError("The `labels` argument must be provided if `custom_palette` is a dictionary.")
    if isinstance(custom_palette, (list, tuple)) and len(custom_palette) < number_of_curves_to_plot:
        raise ValueError("The `custom_palette` list/tuple must have the same length as the number of `data` provided.")

    # Contrast alpha and desat
    contrast_alpha = kwargs.get('contrast_alpha')
    contrast_desat = kwargs.get('contrast_desat')
    if not isinstance(contrast_alpha, float) or not 0 <= contrast_alpha <= 1:
        raise TypeError("`contrast_alpha` must be a float between 0 and 1.")
    
    if not isinstance(contrast_desat, (float, int)) or not 0 <= contrast_desat <= 1:
        raise TypeError("`contrast_desat` must be a float between 0 and 1 or an int (1).")
    
    # Contrast labels
    labels_fontsize = kwargs.get('labels_fontsize')
    labels_rotation = kwargs.get('labels_rotation')
    if labels is not None and not all(isinstance(label, str) for label in labels):
        raise TypeError("The `labels` must be a list of strings or `None`.")
    
    if labels is not None and len(labels) != number_of_curves_to_plot:
        raise ValueError("`labels` must match the number of `data` provided.")
    
    if not isinstance(labels_fontsize, (int, float)):
        raise TypeError("`labels_fontsize` must be an integer or float.")
    
    if labels_rotation is not None and (not isinstance(labels_rotation, (int, float)) or not 0 <= labels_rotation <= 360):
        raise TypeError("`labels_rotation` must be an integer or float between 0 and 360.")   

    # Title
    title = kwargs.get('title')
    title_fontsize = kwargs.get('title_fontsize')
    if title is not None and not isinstance(title, str):
        raise TypeError("The `title` argument must be a string.")
    
    if not isinstance(title_fontsize, (int, float)):
        raise TypeError("`title_fontsize` must be an integer or float.")
    
    # Y-label
    ylabel = kwargs.get('ylabel')
    ylabel_fontsize = kwargs.get('ylabel_fontsize')
    if ylabel is not None and not isinstance(ylabel, str):
        raise TypeError("The `ylabel` argument must be a string.")

    if not isinstance(ylabel_fontsize, (int, float)):
        raise TypeError("`ylabel_fontsize` must be an integer or float.")
    
    # Y-lim
    ylim = kwargs.get('ylim')
    if ylim is not None and not isinstance(ylim, (tuple, list)):
        raise TypeError("`ylim` must be a tuple or list of two floats.")
    if ylim is not None and len(ylim) != 2:
        raise ValueError("`ylim` must be a tuple or list of two floats.")

    # Y-ticks
    yticks = kwargs.get('yticks')
    if yticks is not None and not isinstance(yticks, (tuple, list)):
        raise TypeError("`yticks` must be a tuple or list of floats.")
    
    # Y-ticklabels
    yticklabels = kwargs.get('yticklabels')
    if yticklabels is not None and not isinstance(yticklabels, (tuple, list)):
        raise TypeError("`yticklabels` must be a tuple or list of strings.")
    
    if yticklabels is not None and not all(isinstance(label, str) for label in yticklabels):
        raise TypeError("`yticklabels` must be a list of strings.")
    
    # Remove spines
    remove_spines = kwargs.get('remove_spines')
    if not isinstance(remove_spines, bool):
        raise TypeError("`remove_spines` must be a boolean value.")
    
    # Summary bars
    summary_bars = kwargs.get('summary_bars')
    if summary_bars is not None:
        if not isinstance(summary_bars, list | tuple):
            raise TypeError("`summary_bars` must be a list/tuple of indices (ints).")
        if not all(isinstance(i, int) for i in summary_bars):
            raise TypeError("`summary_bars` must be a list/tuple of indices (ints).")
        if any(i >= number_of_curves_to_plot for i in summary_bars):
            raise ValueError("Index {} chosen is out of range for the contrast objects.".format([i for i in summary_bars if i >= number_of_curves_to_plot]))
    
    # Delta text
    delta_text = kwargs.get('delta_text')
    if delta_text is not None:
        if not isinstance(delta_text, bool):
            raise TypeError("`delta_text` must be a boolean value.")

    # Contrast bars
    contrast_bars = kwargs.get('contrast_bars')
    if contrast_bars is not None:
        if not isinstance(contrast_bars, bool):
            raise TypeError("`contrast_bars` must be a boolean value.")

    return contrast_type    

def get_kwargs(
        violin_kwargs,
        zeroline_kwargs,
        horizontal,
        marker_kwargs,
        errorbar_kwargs,
        delta_text_kwargs,
        contrast_bars_kwargs,
        summary_bars_kwargs,
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
    default_marker_kwargs = {
                'marker': 'o',
                'markersize': marker_size,
                'color': 'black',
                'alpha': 1,
                'zorder': 2,
    }
    if marker_kwargs is None:
        marker_kwargs = default_marker_kwargs
    else:
        marker_kwargs = merge_two_dicts(default_marker_kwargs, marker_kwargs)

    # Effect size error bar kwargs
    default_errorbar_kwargs = {
                'color': 'black',
                'lw': 2.5,
                'linestyle': '-',
                'alpha': 1,
                'zorder': 1,
    }
    if errorbar_kwargs is None:
        errorbar_kwargs = default_errorbar_kwargs
    else:
        errorbar_kwargs = merge_two_dicts(default_errorbar_kwargs, errorbar_kwargs)

    # Delta text kwargs
    default_delta_text_kwargs = {
                "color": None, 
                "alpha": 1,
                "fontsize": 10, 
                "ha": 'center', 
                "va": 'center', 
                "rotation": 0, 
                "x_coordinates": None, 
                "y_coordinates": None,
                "offset": 0
    }
    if delta_text_kwargs is None:
        delta_text_kwargs = default_delta_text_kwargs
    else:
        delta_text_kwargs = merge_two_dicts(default_delta_text_kwargs, delta_text_kwargs)

    # Contrast bars kwargs.
    default_contrast_bars_kwargs = {
                    "color": None, 
                    "zorder":-3,
                    'alpha': 0.15
    }
    if contrast_bars_kwargs is None:
        contrast_bars_kwargs = default_contrast_bars_kwargs
    else:
        contrast_bars_kwargs = merge_two_dicts(default_contrast_bars_kwargs, contrast_bars_kwargs)

    # Summary bars kwargs.
    default_summary_bars_kwargs = {
                    "span_ax": False,
                    "color": None, 
                    "alpha": 0.15,
                    "zorder":-3
    }
    if summary_bars_kwargs is None:
        summary_bars_kwargs = default_summary_bars_kwargs
    else:
        summary_bars_kwargs = merge_two_dicts(default_summary_bars_kwargs, summary_bars_kwargs)

    return (violin_kwargs, zeroline_kwargs, marker_kwargs, errorbar_kwargs, 
            delta_text_kwargs, contrast_bars_kwargs, summary_bars_kwargs)

def color_palette(
        custom_palette, 
        labels, 
        number_of_curves_to_plot,
        contrast_desat
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
    violin_colors = [sns.desaturate(color, contrast_desat) for color in violin_colors]
    return violin_colors

def forest_plot(
    data: list,
    idx: Optional[list[int]] = None,
    ax: Optional[plt.Axes] = None,
    fig_size: tuple[int, int] = None,
    effect_size: str = "mean_diff",
    ci_type='bca',
    horizontal: bool = False, 

    marker_size: int = 10,
    custom_palette: Optional[Union[dict, list, str]] = None,
    contrast_alpha: float = 0.8,
    contrast_desat: float = 1,

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

    delta_text: bool = True,
    delta_text_kwargs: dict = None,

    contrast_bars: bool = True,
    contrast_bars_kwargs: dict = None,
    summary_bars: list|tuple = None,
    summary_bars_kwargs: dict = None,

    violin_kwargs: Optional[dict] = None,
    zeroline_kwargs: Optional[dict] = None,
    marker_kwargs: Optional[dict] = None,
    errorbar_kwargs: Optional[dict] = None,
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
    ci_type : str
        Type of confidence interval to plot (bca' or 'pct')
    horizontal : bool, default=False
        If True, the plot will be horizontal.
    marker_size : int, default=12
        Marker size for plotting effect size dots.
    custom_palette : Optional[Union[dict, list, str]], default=None
        Custom color palette for the plot.
    contrast_alpha : float, default=0.8
        Transparency level for violin plots.
    contrast_desat : float, default=1
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
    delta_text : bool, default=True
        If True, it adds text next to each curve representing the effect size value.
    delta_text_kwargs : dict, default=None
        Additional keyword arguments for the delta_text.
    contrast_bars : bool, default=True
        If True, it adds bars from the zeroline to the effect size curve.
    contrast_bars_kwargs : dict, default=None
        Additional keyword arguments for the contrast_bars.
    summary_bars: list | tuple, default=None,
        If True, it adds summary bars to the relevant effect size curves.
    summary_bars_kwargs : dict, default=None,
        Additional keyword arguments for the summary_bars.
    violin_kwargs : Optional[dict], default=None
        Additional arguments for violin plot customization.
    zeroline_kwargs : Optional[dict], default=None
        Additional arguments for the zero line customization.
    marker_kwargs : Optional[dict], default=None
        Additional arguments for the effect size marker customization.
    errorbar_kwargs : Optional[dict], default=None
        Additional arguments for the effect size error bar customization.

    Returns
    -------
    plt.Figure
        The matplotlib figure object with the generated forest plot.
    """
    from .plot_tools import halfviolin

    # Check for errors in the input arguments
    all_kwargs = locals()
    contrast_type = check_for_errors(**all_kwargs)

    # Load plot data and extract info
    bootstraps, differences, bcalows, bcahighs = load_plot_data(
                                                        data = data, 
                                                        effect_size = effect_size, 
                                                        contrast_type = contrast_type,
                                                        ci_type = ci_type,
                                                        idx = idx
    )
    # Adjust figure size based on orientation
    number_of_curves_to_plot = len(bootstraps)
    if ax is not None:
        fig = ax.figure
    else:
        if fig_size is None:
            fig_size = (4, 1.3 * number_of_curves_to_plot) if horizontal else (1.3 * number_of_curves_to_plot, 4)
        fig, ax = plt.subplots(figsize=fig_size)

    # Get Kwargs
    (violin_kwargs, zeroline_kwargs, marker_kwargs, errorbar_kwargs, 
     delta_text_kwargs, contrast_bars_kwargs, summary_bars_kwargs) = get_kwargs(
                                                                        violin_kwargs = violin_kwargs,
                                                                        zeroline_kwargs = zeroline_kwargs,
                                                                        horizontal = horizontal,
                                                                        marker_kwargs = marker_kwargs,
                                                                        errorbar_kwargs = errorbar_kwargs,
                                                                        delta_text_kwargs = delta_text_kwargs,
                                                                        contrast_bars_kwargs = contrast_bars_kwargs,
                                                                        summary_bars_kwargs = summary_bars_kwargs,
                                                                        marker_size = marker_size
    )
                                            
    # Plot the violins and make adjustments
    v = ax.violinplot(
        bootstraps, 
        **violin_kwargs
    )
    halfviolin(
            v, 
            alpha = contrast_alpha, 
            half = "bottom" if horizontal else "right",
        )
    
    ## Plotting the effect sizes and confidence intervals
    for k in range(1, number_of_curves_to_plot + 1):
        if horizontal:
            ax.plot(differences[k - 1], k, **marker_kwargs)  
            ax.plot([bcalows[k - 1], bcahighs[k - 1]], [k, k], **errorbar_kwargs) 
        else:
            ax.plot(k, differences[k - 1], **marker_kwargs)
            ax.plot([k, k], [bcalows[k - 1], bcahighs[k - 1]], **errorbar_kwargs)
    
    # Aesthetic Adjustments
    ## Handle the custom color palette
    violin_colors = color_palette(
                        custom_palette = custom_palette, 
                        labels = labels, 
                        number_of_curves_to_plot = number_of_curves_to_plot,
                        contrast_desat = contrast_desat
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

    # Delta Text
    if delta_text:
        if delta_text_kwargs.get('color') is not None:
            delta_text_colors = [delta_text_kwargs.pop('color')] * number_of_curves_to_plot
        else:
            delta_text_colors = violin_colors
            delta_text_kwargs.pop('color')

        # Collect the X-coordinates for the delta text
        delta_text_x_coordinates = delta_text_kwargs.pop('x_coordinates')
        delta_text_x_adjustment = delta_text_kwargs.pop('offset')

        if delta_text_x_coordinates is not None:
            if not isinstance(delta_text_x_coordinates, (list, tuple)) or not all(isinstance(x, (int, float)) for x in delta_text_x_coordinates):
                raise TypeError("delta_text_kwargs['x_coordinates'] must be a list of x-coordinates.")
            if len(delta_text_x_coordinates) != number_of_curves_to_plot:
                raise ValueError("delta_text_kwargs['x_coordinates'] must have the same length as the number of ticks to plot.")
        else:
            delta_text_x_coordinates = (np.arange(1, number_of_curves_to_plot + 1) 
                                        + (0.5 if not horizontal else -0.4)
                                        + delta_text_x_adjustment
                                    )

        # Collect the Y-coordinates for the delta text
        delta_text_y_coordinates = delta_text_kwargs.pop('y_coordinates')

        if delta_text_y_coordinates is not None:
            if not isinstance(delta_text_y_coordinates, (list, tuple)) or not all(isinstance(y, (int, float)) for y in delta_text_y_coordinates):
                raise TypeError("delta_text_kwargs['y_coordinates'] must be a list of y-coordinates.")
            if len(delta_text_y_coordinates) != number_of_curves_to_plot:
                raise ValueError("delta_text_kwargs['y_coordinates'] must have the same length as the number of ticks to plot.")
        else:
            delta_text_y_coordinates = differences

        if horizontal:
            delta_text_x_coordinates, delta_text_y_coordinates = delta_text_y_coordinates, delta_text_x_coordinates

        for idx, x, y, delta in zip(np.arange(0, number_of_curves_to_plot, 1), delta_text_x_coordinates, 
                                    delta_text_y_coordinates, differences):
            delta_text = np.format_float_positional(delta, precision=2, sign=True, trim="k", min_digits=2)
            ax.text(x, y, delta_text, color=delta_text_colors[idx], zorder=5, **delta_text_kwargs)

    # Contrast bars
    if contrast_bars:
        _bar_color = contrast_bars_kwargs.pop('color')
        if _bar_color is not None:
            bar_colors = [_bar_color] * number_of_curves_to_plot
        else:
            bar_colors = violin_colors
        for x, y in zip(np.arange(1, number_of_curves_to_plot + 1), differences):
            if horizontal:
                ax.add_patch(mpatches.Rectangle((0, x-0.25), y, 0.25, color=bar_colors[x-1], **contrast_bars_kwargs))
            else:
                ax.add_patch(mpatches.Rectangle((x, 0), 0.25, y, color=bar_colors[x-1], **contrast_bars_kwargs))

    # Summary bars
    if summary_bars:
        _bar_color = summary_bars_kwargs.pop('color')
        if _bar_color is not None:
            bar_colors = [_bar_color] * number_of_curves_to_plot
        else:
            bar_colors = violin_colors

        span_ax = summary_bars_kwargs.pop("span_ax")
        summary_xmin, summary_xmax = ax.get_xlim()
        summary_ymin, summary_ymax = ax.get_ylim()

        for summary_index in summary_bars:
            if span_ax == True:
                starting_location = summary_ymin if horizontal else summary_xmin
            else:
                starting_location = summary_index+1   

            summary_color = bar_colors[summary_index]
            summary_ci_low, summary_ci_high = bcalows[summary_index], bcahighs[summary_index]

            if horizontal:
                ax.add_patch(mpatches.Rectangle(
                    (summary_ci_low, starting_location),
                    summary_ci_high-summary_ci_low, summary_ymax+1, 
                    color=summary_color, 
                    **summary_bars_kwargs)
                    )
            else:
                ax.add_patch(mpatches.Rectangle(
                    (starting_location, summary_ci_low),
                    summary_xmax+1, summary_ci_high-summary_ci_low, 
                    color=summary_color, 
                    **summary_bars_kwargs)
                    )

    ## Invert Y-axis if horizontal 
    if horizontal:
        ax.invert_yaxis()

    return fig
