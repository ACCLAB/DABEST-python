import math
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from pandas.api.types import CategoricalDtype
from matplotlib.colors import ListedColormap


class SwarmPlot:
    def __init__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        ax: plt.figure,
        order: List[str],
        hue: str,
        palette: Dict[str, Tuple[float, float, float]],
        zorder: int,
        size: int = 20,
        side: str = "center",
        **kwargs,
    ):
        """
        Initialize a SwarmPlot instance.

        Parameters:
        - data (pd.DataFrame): The input data as a pandas DataFrame.
        - x (str): The column in the DataFrame to be used as the x-axis.
        - y (str): The column in the DataFrame to be used as the y-axis.
        - ax (plt.figure): The matplotlib figure object to which the swarm plot will be added.
        - order (List[str]): The order in which x-axis categories should be displayed.
        - hue (str): The column in the DataFrame that determines the grouping for colour.
        - palette (Tuple[str]): The color palette to be used for plotting.
        - zorder (int): The z-order for drawing the swarm plot wrt other matplotlib drawings.
        - dot_size (int, optional): The size of the markers in the swarm plot. Default is 20.
        - side (str, optional): The side on which points are swarmed ("center", "left", or "right"). Default is "center".
        - **kwargs: Additional keyword arguments to be passed to the scatter plot.

        Returns:
        None
        """
        self.__data = data
        self.__x = x
        self.__y = y
        self.__order = order  # if None, generate our own?
        self.__palette = palette
        self.__zorder = zorder
        self.__size = size*6.5
        self.__side = side

        if hue is None:
            self.__colour_col = self.__x
        else:
            self.__colour_col = hue

        # Check validity of input params
        self._check_errors()

        data_copy = data.copy(deep=True)
        # make x column into CategoricalDType to sort by
        data_copy[x] = data_copy[x].astype(
            CategoricalDtype(categories=self.__order, ordered=True)
        )
        data_copy.sort_values(by=[self.__x, self.__y], inplace=True)
        self.__data_copy = data_copy

        # TODO: figure out how the boxed up part below works
        ### START OF BOX
        x_vals = range(len(self.__order))
        y_vals = self.__data_copy[self.__y]

        if ax is None:
            ax = plt.gca()

        x_min = min(x_vals)
        x_max = max(x_vals)
        ax.set_xlim(left=x_min-0.5, right=x_max+0.5)

        y_range = max(y_vals) - min(y_vals)
        y_min = min(y_vals) - 0.05 * y_range
        y_max = max(y_vals) + 0.05 * y_range
        ax.set_ylim(bottom=y_min, top=y_max)

        figw, figh = ax.get_figure().get_size_inches()
        w = (ax.get_position().xmax - ax.get_position().xmin) * figw
        h = (ax.get_position().ymax - ax.get_position().ymin) * figh
        ax_xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
        ax_yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
        self.__ax = ax

        gsize = math.sqrt(self.__size) * 1.0 / 80 * ax_xspan * 1.0 / (w * 0.8)
        dsize = math.sqrt(self.__size) * 1.0 / 80 * ax_yspan * 1.0 / (h * 0.8)
        ### END OF BOX

        self.__gsize = gsize
        self.__dsize = dsize

    def _check_errors(self) -> None:
        # TODO: Check validity of params
        """
        Check the validity of input parameters. Raises exceptions if detected.

        Parameters:
        None

        Returns:
        None
        """
        return None

    def _swarm(self, values: int, gsize: int, dsize: int, side: str) -> pd.Series:
        """
        Perform the swarm algorithm to position points without overlap.

        Parameters:
        - values (int): The values to be plotted.
        - gsize (int): The size of the gap between points.
        - dsize (int): The size of the markers.
        - side (str): The side on which points are swarmed ("center", "left", or "right").

        Returns:
        pd.Series: The x-offset values for the swarm plot.
        """
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
        self, points_data: pd.DataFrame, is_drop_gutter: bool, gutter_limit: int, value_column: str
    ) -> pd.DataFrame:
        """
        Adjust points that hit the gutters or drop them based on the provided conditions.

        Parameters:
        - is_drop_gutter (bool): If True, drop points that hit the gutters; otherwise, readjust them.
        - gutter_limit (int): The limit for points hitting the gutters.

        Returns:
        pd.DataFrame: Adjusted DataFrame with x-position modifications.
        """
        x_position = 0
        if self.__side == "center":
            gutter_limit = gutter_limit / 2
        for _, group_i in points_data.groupby(self.__x):
            hit_gutter = abs(group_i[value_column]) >= x_position + gutter_limit
            total_num_of_points = group_i.shape[0]
            num_of_points_hit_gutter = group_i[hit_gutter].shape[0]
            if any(hit_gutter):
                if is_drop_gutter:
                    # Drop points that hit gutter
                    group_i.drop(group_i[hit_gutter].index.to_list(), inplace=True)
                    err1 = f"""
                        {num_of_points_hit_gutter/total_num_of_points:.1%} of the points cannot be placed.
                        You might want to decrease the size of the markers.
                        """
                    warnings.warn(err1)
                else:
                    for i in group_i[hit_gutter].index:
                        group_i.loc[i, value_column] = np.sign(group_i.loc[i, value_column]) * (
                            x_position + gutter_limit
                        )
            points_data = pd.concat([points_data, group_i])
            x_position = x_position + 1

        return points_data

    def swarmplot(
        self, is_drop_gutter: bool, gutter_limit: int, ax: plt.figure, **kwargs
    ) -> plt.figure:
        """
        Generate a swarm plot.

        Parameters:
        - is_drop_gutter (bool): If True, drop points that hit the gutters; otherwise, readjust them.
        - gutter_limit (int): The limit for points hitting the gutters.
        - **kwargs: Additional keyword arguments to be passed to the scatter plot.

        Returns:
        plt.figure: The matplotlib figure containing the swarm plot.
        """
        # Group by x, then repeat swarm creation algo on the various group in _, groups of the pd.groupby.generic.DataFrameGroupBy object
        # Assumptions are that self.__data is already sorted according to self.__order
        x_final = []
        x_position = 0
        x_tick_tabels = []
        for group_i, values_i in self.__data_copy.groupby(self.__x):
            x_new = []
            values_i_y = values_i[self.__y]
            x_offset = self._swarm(
                values=values_i_y, gsize=self.__gsize, dsize=self.__dsize, side=self.__side
            )
            x_new = [x_position + offset for offset in x_offset]
            values_i["x_new"] = x_new
            values_i = self._adjust_gutter_points(values_i, is_drop_gutter, gutter_limit, "x_new")
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
