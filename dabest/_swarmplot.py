import math
import warnings
import pandas as pd
import numpy as np
import matplotlib.axes._subplots as axes
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Iterable, Union
from pandas.api.types import CategoricalDtype
from matplotlib.colors import ListedColormap


def swarmplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    ax: axes.Subplot = None,
    order: List = None,
    hue: str = None,
    palette: Union[Iterable, str] = "black",
    zorder: Union[int, float] = 1,
    size: Union[int, float] = 20,
    side: str = "center",
    jitter: Union[int, float] = 75,
    is_drop_gutter: bool = True,
    gutter_limit: int = 0.5,
    **kwargs,
):
    """
    API to plot a swarm plot.

    Parameters
    ----------
    data : pd.DataFrame
        The input data as a pandas DataFrame.
    x : str
        The column in the DataFrame to be used as the x-axis.
    y : str
        The column in the DataFrame to be used as the y-axis.
    ax : axes.Subplot
        Matplotlib AxesSubplot object for which the plot would be drawn on. Default is None.
    order : List
        The order in which x-axis categories should be displayed. Default is None.
    hue : str
        The column in the DataFrame that determines the grouping for colour.
        If None (by default), it assumes that it is being grouped by x.
    palette : Union[Iterable, str]
        The color palette to be used for plotting. Default is "black".
    zorder : int | float
        The z-order for drawing the swarm plot wrt other matplotlib drawings. Default is 1.
    dot_size : int | float
        The size of the markers in the swarm plot. Default is 20.
    side : str
        The side on which points are swarmed ("center", "left", or "right"). Default is "center".
    jitter : int | float
        Determines the distance between points. Default is 1.
    is_drop_gutter : bool
        If True, drop points that hit the gutters; otherwise, readjust them.
    gutter_limit : int | float
        The limit for points hitting the gutters.
    **kwargs:
        Additional keyword arguments to be passed to the swarm plot.

    Returns
    -------
    axes.Subplot
        Matplotlib AxesSubplot object for which the swarm plot has been drawn on.
    """
    s = SwarmPlot(data, x, y, ax, order, hue, palette, zorder, size, side, jitter)
    ax = s.plot(is_drop_gutter, gutter_limit, ax, **kwargs)
    return ax


class SwarmPlot:
    def __init__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        ax: axes.Subplot = None,
        order: List = None,
        hue: str = None,
        palette: Union[Iterable, str] = "black",
        zorder: Union[int, float] = 1,
        size: Union[int, float] = 20,
        side: str = "center",
        jitter: Union[int, float] = 75,
    ):
        """
        Initialize a SwarmPlot instance.

        Parameters
        ----------
        data : pd.DataFrame
            The input data as a pandas DataFrame.
        x : str
            The column in the DataFrame to be used as the x-axis.
        y : str
            The column in the DataFrame to be used as the y-axis.
        ax : axes.Subplot
            Matplotlib AxesSubplot object for which the plot would be drawn on. Default is None.
        order : List
            The order in which x-axis categories should be displayed. Default is None.
        hue : str
            The column in the DataFrame that determines the grouping for colour.
            If None (by default), it assumes that it is being grouped by x.
        palette : Union[Iterable, str]
            The color palette to be used for plotting. Default is "black".
        zorder : int | float
            The z-order for drawing the swarm plot wrt other matplotlib drawings. Default is 1.
        dot_size : int | float
            The size of the markers in the swarm plot. Default is 20.
        side : str
            The side on which points are swarmed ("center", "left", or "right"). Default is "center".
        jitter : int | float
            Determines the distance between points. Default is 1.

        Returns
        -------
        None
        """
        self.__x = x
        self.__y = y
        self.__order = order
        self.__hue = hue
        self.__zorder = zorder
        self.__palette = palette
        self.__jitter = jitter

        # Input validation
        self._check_errors(data, ax, size, side)

        self.__size = size * 4
        self.__side = side.lower()
        self.__data = data
        self.__color_col = self.__x if self.__hue is None else self.__hue

        # Generate default values
        if order is None:
            self.__order = self._generate_order()
        if ax is None:
            ax = plt.gca()

        # Reformatting
        if not isinstance(self.__palette, dict):
            self.__palette = self._format_palette(self.__palette)
        data_copy = data.copy(deep=True)
        if not isinstance(self.__data[self.__x].dtype, pd.CategoricalDtype):
            # make x column into CategoricalDType to sort by
            data_copy[self.__x] = data_copy[self.__x].astype(
                CategoricalDtype(categories=self.__order, ordered=True)
            )
        data_copy.sort_values(by=[self.__x, self.__y], inplace=True)
        self.__data_copy = data_copy

        x_vals = range(len(self.__order))
        y_vals = self.__data_copy[self.__y]

        x_min = min(x_vals)
        x_max = max(x_vals)
        ax.set_xlim(left=x_min - 0.5, right=x_max + 0.5)

        y_range = max(y_vals) - min(y_vals)
        y_min = min(y_vals) - 0.05 * y_range
        y_max = max(y_vals) + 0.05 * y_range

        # ylim is set manually to override Axes.autoscale if it hasn't already been scaled at least once
        if ax.get_autoscaley_on():
            ax.set_ylim(bottom=y_min, top=y_max)

        figw, figh = ax.get_figure().get_size_inches()
        w = (ax.get_position().xmax - ax.get_position().xmin) * figw
        h = (ax.get_position().ymax - ax.get_position().ymin) * figh
        ax_xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
        ax_yspan = ax.get_ylim()[1] - ax.get_ylim()[0]

        # increases jitter distance based on number of swarms that is going to be drawn
        jitter = jitter * (1 + 0.05 * (math.log(ax_xspan)))

        gsize = (
            math.sqrt(self.__size) * 1.0 / (75 / jitter) * ax_xspan * 1.0 / (w * 0.8)
        )
        dsize = (
            math.sqrt(self.__size) * 1.0 / (75 / jitter) * ax_yspan * 1.0 / (h * 0.8)
        )
        self.__gsize = gsize
        self.__dsize = dsize

    def _check_errors(
        self, data: pd.DataFrame, ax: axes.Subplot, size: Union[int, float], side: str
    ) -> None:
        """
        Check the validity of input parameters. Raises exceptions if detected.

        Parameters
        ----------
        data : pd.Dataframe
            Input data used for generation of the swarmplot.
        ax : axes.Subplot
            Matplotlib AxesSubplot object for which the plot would be drawn on.
        size : Union[int, float]
            scalar value determining size of dots of the swarmplot.
        side: str
            The side on which points are swarmed ("center", "left", or "right"). Default is "center".

        Returns
        -------
        None
        """
        # Type Enforcement
        if not isinstance(data, pd.DataFrame):
            raise ValueError("`data` must be a Pandas Dataframe.")
        if not isinstance(ax, axes.Subplot):
            raise ValueError("`ax` must be a Matplotlib AxesSubplot.")
        if not isinstance(size, (int, float)):
            raise ValueError("`size` must be a scalar or float.")
        if not isinstance(side, str):
            raise ValueError(
                "Invalid `side`. Must be one of 'center', 'right', or 'left'."
            )
        if not isinstance(self.__x, str):
            raise ValueError("`x` must be a string.")
        if not isinstance(self.__y, str):
            raise ValueError("`y` must be a string.")
        if not isinstance(self.__zorder, (int, float)):
            raise ValueError("`zorder` must be a scalar or float.")
        if not isinstance(self.__jitter, (int, float)):
            raise ValueError("`jitter` must be a scalar or float.")
        if not isinstance(self.__palette, (str, dict)):
            raise ValueError("`palette` must be either a string or a dict.")
        if self.__hue is not None and not isinstance(self.__hue, str):
            raise ValueError("`hue` must be either a string or None.")
        if self.__order is not None and not isinstance(self.__order, Iterable):
            raise ValueError("`order` must be either an Iterable or None.")

        # More thorough Input Validation Checks
        if self.__x not in data.columns:
            err = "{0} is not a column in `data`.".format(self.__x)
            raise IndexError(err)
        if self.__y not in data.columns:
            err = "{0} is not a column in `data`.".format(self.__y)
            raise IndexError(err)
        if self.__hue is not None and self.__hue not in data.columns:
            err = "{0} is not a column in `data`.".format(self.__hue)
            raise IndexError(err)

        color_col = self.__x if self.__hue is None else self.__hue
        if self.__order is not None:
            for group_i in self.__order:
                if group_i not in pd.unique(data[self.__x]):
                    err = "{0} in `order` is not in the {1} column of `data`.".format(
                        group_i, self.__x
                    )
                    raise IndexError(err)
        for group_i in self.__palette.keys():
            if group_i not in pd.unique(data[color_col]):
                err = "{0} in `palette` is not in the {1} column of `data`.".format(
                    group_i, color_col
                )
                raise IndexError(err)

        if side.lower() not in ["center", "right", "left"]:
            raise ValueError(
                "Invalid `side`. Must be one of 'center', 'right', or 'left'."
            )

        return None

    def _generate_order(self) -> List:
        """
        Generates order value that determines the order in which x-axis categories should be displayed.

        Parameters
        ----------
        None

        Returns
        -------
        List:
            contains the order in which the x-axis categories should be displayed.
        """
        if isinstance(self.__data[self.__x].dtype, pd.CategoricalDtype):
            order = pd.unique(self.__data[self.__x]).categories.tolist()
        else:
            order = pd.unique(self.__data[self.__x]).tolist()

        return order

    def _format_palette(self, palette: Union[str, List, Tuple]) -> Dict:
        """
        Reformats palette into appropriate Dictionary form for swarm plot

        Parameters
        ----------
        palette: str | List | Tuple
            The color palette used for the swarm plot. Conventions are based on Matplotlib color
            specifications.

            Could be a singular string value - in which case, would be a singular color name.
            In the case of a List or Tuple - it could be a Sequence of color names or RGB(A) values.

        Returns
        -------
        Dict:
            Dictionary mapping unique groupings in the color column (of the data used for the swarm plot)
            to a color name (str) or a RGB(A) value (Tuple[float, float, float] | List[float, float, float]).
        """
        reformatted_palette = dict()
        groups = pd.unique(self.__data[self.__color_col]).tolist()

        if isinstance(palette, str):
            for group_i in groups:
                reformatted_palette[group_i] = palette
        if isinstance(palette, (list, tuple)):
            if len(groups) != len(palette):
                err = "unique values in {0} column in `data` \
                    and `palette` do not have the same length. Number of unique values is {1} \
                    while length of palette is {2}. The assignment of the colours in the \
                    palette will be cycled".format(
                    self.__color_col, len(groups), len(palette)
                )
                warnings.warn(err)
            for i, group_i in enumerate(groups):
                reformatted_palette[group_i] = palette[i % len(palette)]

        return reformatted_palette

    def _swarm(
        self, values: Iterable[float], gsize: float, dsize: float, side: str
    ) -> pd.Series:
        """
        Perform the swarm algorithm to position points without overlap.

        Parameters
        ----------
        values : Iterable[int | float]
            The values to be plotted.
        gsize : int | float
            The size of the gap between points.
        dsize : int | float
            The size of the markers.
        side : str
            The side on which points are swarmed ("center", "left", or "right").

        Returns
        -------
        pd.Series:
            The x-offset values for the swarm plot.
        """
        # Input validation checks
        if not isinstance(values, Iterable):
            raise ValueError("`values` must be an Iterable")
        if not isinstance(gsize, (int, float)):
            raise ValueError("`gsize` must be a scalar or float.")
        if not isinstance(dsize, (int, float)):
            raise ValueError("`dsize` must be a scalar or float.")

        # sorting algorithm based off of: https://github.com/mgymrek/pybeeswarm
        points_data = pd.DataFrame(
            {"y": [yval * 1.0 / dsize for yval in values], "x": [0] * len(values)}
        )
        for i in range(1, points_data.shape[0]):
            y_i = points_data["y"].values[i]
            points_placed = points_data[0:i]
            is_points_overlap = (
                abs(y_i - points_placed["y"]) < 1
            )  # Checks if y_i is overlapping with any points already placed
            if any(is_points_overlap):
                points_placed = points_placed[is_points_overlap]
                x_offsets = points_placed["y"].apply(
                    lambda y_j: math.sqrt(1 - (y_i - y_j) ** 2)
                )
                if side == "center":
                    potential_x_offsets = pd.Series(
                        [0]
                        + (points_placed["x"] + x_offsets).tolist()
                        + (points_placed["x"] - x_offsets).tolist()
                    )
                if side == "right":
                    potential_x_offsets = pd.Series(
                        [0] + (points_placed["x"] + x_offsets).tolist()
                    )
                if side == "left":
                    potential_x_offsets = pd.Series(
                        [0] + (points_placed["x"] - x_offsets).tolist()
                    )
                bad_x_offsets = []
                for x_i in potential_x_offsets:
                    dists = (y_i - points_placed["y"]) ** 2 + (
                        x_i - points_placed["x"]
                    ) ** 2
                    if any([item < 0.999 for item in dists]):
                        bad_x_offsets.append(True)
                    else:
                        bad_x_offsets.append(False)
                potential_x_offsets[bad_x_offsets] = np.infty
                abs_potential_x_offsets = [abs(_) for _ in potential_x_offsets]
                valid_x_offset = potential_x_offsets[
                    abs_potential_x_offsets.index(min(abs_potential_x_offsets))
                ]
                points_data.loc[i, "x"] = valid_x_offset
            else:
                points_data.loc[i, "x"] = 0

        points_data.loc[np.isnan(points_data["y"]), "x"] = np.nan

        return points_data["x"] * gsize

    def _adjust_gutter_points(
        self,
        points_data: pd.DataFrame,
        x_position: float,
        is_drop_gutter: bool,
        gutter_limit: float,
        value_column: str,
    ) -> pd.DataFrame:
        """
        Adjust points that hit the gutters or drop them based on the provided conditions.

        Parameters
        ----------
        points_data: pd.DataFrame
            Data containing coordinates of points for the swarm plot.
        x_position: int | float
            X-coordinate of the center of a singular swarm group of the swarm plot
        is_drop_gutter : bool
            If True, drop points that hit the gutters; otherwise, readjust them.
        gutter_limit : int | float
            The limit for points hitting the gutters.
        value_column : str
            column in points_data that contains the coordinates for the points in the axis against the gutter

        Returns
        -------
        pd.DataFrame:
            DataFrame with adjusted points based on the gutter limit.
        """
        if self.__side == "center":
            gutter_limit = gutter_limit / 2

        hit_gutter = abs(points_data[value_column] - x_position) >= gutter_limit
        total_num_of_points = points_data.shape[0]
        num_of_points_hit_gutter = points_data[hit_gutter].shape[0]
        if any(hit_gutter):
            if is_drop_gutter:
                # Drop points that hit gutter
                points_data.drop(points_data[hit_gutter].index.to_list(), inplace=True)
                err1 = f"""
                    {num_of_points_hit_gutter/total_num_of_points:.1%} of the points cannot be placed.
                    You might want to decrease the size of the markers.
                    """
                warnings.warn(err1)
            else:
                for i in points_data[hit_gutter].index:
                    points_data.loc[i, value_column] = np.sign(
                        points_data.loc[i, value_column]
                    ) * (x_position + gutter_limit)

        return points_data

    def plot(
        self, is_drop_gutter: bool, gutter_limit: float, ax: axes.Subplot, **kwargs
    ) -> axes.Subplot:
        """
        Generate a swarm plot.

        Parameters
        ----------
        is_drop_gutter : bool
            If True, drop points that hit the gutters; otherwise, readjust them.
        gutter_limit : int | float
            The limit for points hitting the gutters.
        ax : axes.Subplot
            The matplotlib figure object to which the swarm plot will be added.
        **kwargs:
            Additional keyword arguments to be passed to the scatter plot.

        Returns
        -------
        axes.Subplot:
            The matplotlib figure containing the swarm plot.
        """
        # Validation Checks
        if not isinstance(is_drop_gutter, bool):
            raise ValueError("`is_drop_gutter` must be a boolean.")
        if not isinstance(gutter_limit, (int, float)):
            raise ValueError("`gutter_limit` must be a scalar or float.")

        # Group by x, then repeat swarm creation algo on the various group in _, groups of the pd.groupby.generic.DataFrameGroupBy object
        # Assumptions are that self.__data is already sorted according to self.__order
        x_position = 0
        x_tick_tabels = []
        for group_i, values_i in self.__data_copy.groupby(self.__x):
            x_new = []
            values_i_y = values_i[self.__y]
            x_offset = self._swarm(
                values=values_i_y,
                gsize=self.__gsize,
                dsize=self.__dsize,
                side=self.__side,
            )
            x_new = [x_position + offset for offset in x_offset]
            values_i["x_new"] = x_new
            values_i = self._adjust_gutter_points(
                values_i, x_position, is_drop_gutter, gutter_limit, "x_new"
            )
            if self.__hue is not None:
                cmap_values, index = np.unique(
                    values_i[self.__hue], return_inverse=True
                )
                cmap = []
                for cmap_group_i in cmap_values:
                    cmap.append(self.__palette[cmap_group_i])
                cmap = ListedColormap(cmap)
                ax.scatter(
                    values_i["x_new"],
                    values_i[self.__y],
                    s=self.__size,
                    c=index,
                    cmap=cmap,
                    zorder=self.__zorder,
                    **kwargs,
                )
            else:
                ax.scatter(
                    values_i["x_new"],
                    values_i[self.__y],
                    s=self.__size,
                    c=self.__palette[group_i],
                    zorder=self.__zorder,
                    **kwargs,
                )
            x_position = x_position + 1
            x_tick_tabels.extend([group_i])

        ax.get_xaxis().set_ticks(np.arange(x_position))
        ax.get_xaxis().set_ticklabels(x_tick_tabels)

        return ax
