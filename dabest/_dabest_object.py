# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/API/dabest_object.ipynb.

# %% auto 0
__all__ = ['Dabest']

# %% ../nbs/API/dabest_object.ipynb 4
# Import standard data science libraries
from numpy import array, repeat, random, issubdtype, number
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import randint

# %% ../nbs/API/dabest_object.ipynb 6
class Dabest(object):

    """
    Class for estimation statistics and plots.
    """

    def __init__(
        self,
        data,
        idx,
        x,
        y,
        paired,
        id_col,
        ci,
        resamples,
        random_seed,
        proportional,
        delta2,
        experiment,
        experiment_label,
        x1_level,
        mini_meta,
    ):
        """
        Parses and stores pandas DataFrames in preparation for estimation
        statistics. You should not be calling this class directly; instead,
        use `dabest.load()` to parse your DataFrame prior to analysis.
        """

        self.__delta2 = delta2
        self.__experiment = experiment
        self.__ci = ci
        self.__input_data = data
        self.__output_data = data.copy()
        self.__id_col = id_col
        self.__is_paired = paired
        self.__resamples = resamples
        self.__random_seed = random_seed
        self.__proportional = proportional
        self.__mini_meta = mini_meta

        # after this call the attributes self.__experiment_label and self.__x1_level are updated
        self._check_errors(x, y, idx, experiment, experiment_label, x1_level)
        

        # Check if there is NaN under any of the paired settings
        if self.__is_paired and self.__output_data.isnull().values.any():
            import warnings
            warn1 = f"NaN values detected under paired setting and removed,"
            warn2 = f" please check your data."
            warnings.warn(warn1 + warn2)
            if x is not None and y is not None:
                rmname = self.__output_data[self.__output_data[y].isnull()][self.__id_col].tolist()
                self.__output_data = self.__output_data[~self.__output_data[self.__id_col].isin(rmname)]
            elif x is None and y is None:
                self.__output_data.dropna(inplace=True)

        # create new x & idx and record the second variable if this is a valid 2x2 ANOVA case
        if idx is None and x is not None and y is not None:
            # Add a length check for unique values in the first element in list x,
            # if the length is greater than 2, force delta2 to be False
            # Should be removed if delta2 for situations other than 2x2 is supported
            if len(self.__output_data[x[0]].unique()) > 2 and self.__x1_level is None:
                self.__delta2 = False
                # stop the loop if delta2 is False

            # add a new column which is a combination of experiment and the first variable
            new_col_name = experiment + x[0]
            while new_col_name in self.__output_data.columns:
                new_col_name += "_"

            self.__output_data[new_col_name] = (
                self.__output_data[x[0]].astype(str)
                + " "
                + self.__output_data[experiment].astype(str)
            )

            # create idx and record the first and second x variable
            idx = []
            for i in list(map(lambda x: str(x), self.__experiment_label)):
                temp = []
                for j in list(map(lambda x: str(x), self.__x1_level)):
                    temp.append(j + " " + i)
                idx.append(temp)

            self.__idx = idx
            self.__x1 = x[0]
            self.__x2 = x[1]
            x = new_col_name
        else:
            self.__idx = idx
            self.__x1 = None
            self.__x2 = None

        # Determine the kind of estimation plot we need to produce.
        if all([isinstance(i, (str, int, float)) for i in idx]):
            # flatten out idx.
            all_plot_groups = pd.unique([t for t in idx]).tolist()
            if len(idx) > len(all_plot_groups):
                err0 = "`idx` contains duplicated groups. Please remove any duplicates and try again."
                raise ValueError(err0)

            # We need to re-wrap this idx inside another tuple so as to
            # easily loop thru each pairwise group later on.
            self.__idx = (idx,)

        elif all([isinstance(i, (tuple, list)) for i in idx]):
            all_plot_groups = pd.unique([tt for t in idx for tt in t]).tolist()

            actual_groups_given = sum([len(i) for i in idx])

            if actual_groups_given > len(all_plot_groups):
                err0 = "Groups are repeated across tuples,"
                err1 = " or a tuple has repeated groups in it."
                err2 = " Please remove any duplicates and try again."
                raise ValueError(err0 + err1 + err2)

        else:  # mix of string and tuple?
            err = "There seems to be a problem with the idx you " "entered--{}.".format(
                idx
            )
            raise ValueError(err)

        # Check if there is a typo on paired
        if self.__is_paired and self.__is_paired not in ("baseline", "sequential"):
            err = "{} assigned for `paired` is not valid.".format(self.__is_paired)
            raise ValueError(err)

        # Determine the type of data: wide or long.
        if x is None and y is not None:
            err = "You have only specified `y`. Please also specify `x`."
            raise ValueError(err)

        if x is not None and y is None:
            err = "You have only specified `x`. Please also specify `y`."
            raise ValueError(err)

        self.__plot_data = self._get_plot_data(x, y, all_plot_groups)
        self.__all_plot_groups = all_plot_groups

        # Check if `id_col` is valid
        if self.__is_paired:
            if id_col is None:
                err = "`id_col` must be specified if `paired` is assigned with a not NoneType value."
                raise IndexError(err)

            if id_col not in self.__plot_data.columns:
                err = "{} is not a column in `data`. ".format(id_col)
                raise IndexError(err)

        self._compute_effectsize_dfs()

    def __repr__(self):
        from .__init__ import __version__
        from .misc_tools import print_greeting

        greeting_header = print_greeting()

        RM_STATUS = {
            "baseline": "for repeated measures against baseline \n",
            "sequential": "for the sequential design of repeated-measures experiment \n",
            "None": "",
        }

        PAIRED_STATUS = {"baseline": "Paired e", "sequential": "Paired e", "None": "E"}

        first_line = {
            "rm_status": RM_STATUS[str(self.__is_paired)],
            "paired_status": PAIRED_STATUS[str(self.__is_paired)],
        }

        s1 = "{paired_status}ffect size(s) {rm_status}".format(**first_line)
        s2 = "with {}% confidence intervals will be computed for:".format(self.__ci)
        desc_line = s1 + s2

        out = [greeting_header + "\n\n" + desc_line]

        comparisons = []

        if self.__is_paired == "sequential":
            for j, current_tuple in enumerate(self.__idx):
                for ix, test_name in enumerate(current_tuple[1:]):
                    control_name = current_tuple[ix]
                    comparisons.append("{} minus {}".format(test_name, control_name))
        else:
            for j, current_tuple in enumerate(self.__idx):
                control_name = current_tuple[0]

                for ix, test_name in enumerate(current_tuple[1:]):
                    comparisons.append("{} minus {}".format(test_name, control_name))

        if self.__delta2:
            comparisons.append(
                "{} minus {} (only for mean difference)".format(
                    self.__experiment_label[1], self.__experiment_label[0]
                )
            )

        if self.__mini_meta:
            comparisons.append("weighted delta (only for mean difference)")

        for j, g in enumerate(comparisons):
            out.append("{}. {}".format(j + 1, g))

        resamples_line1 = "\n{} resamples ".format(self.__resamples)
        resamples_line2 = "will be used to generate the effect size bootstraps."
        out.append(resamples_line1 + resamples_line2)

        return "\n".join(out)

    @property
    def mean_diff(self):
        """
        Returns an :py:class:`EffectSizeDataFrame` for the mean difference, its confidence interval, and relevant statistics, for all comparisons as indicated via the `idx` and `paired` argument in `dabest.load()`

        """
        return self.__mean_diff

    @property
    def median_diff(self):
        """
        Returns an :py:class:`EffectSizeDataFrame` for the median difference, its confidence interval, and relevant statistics, for all comparisons  as indicated via the `idx` and `paired` argument in `dabest.load()`.

        """
        return self.__median_diff

    @property
    def cohens_d(self):
        """
        Returns an :py:class:`EffectSizeDataFrame` for the standardized mean difference Cohen's `d`, its confidence interval, and relevant statistics, for all comparisons as indicated via the `idx` and `paired` argument in `dabest.load()`.

        """
        return self.__cohens_d

    @property
    def cohens_h(self):
        """
        Returns an :py:class:`EffectSizeDataFrame` for the standardized mean difference Cohen's `h`, its confidence interval, and relevant statistics, for all comparisons as indicated via the `idx` and `directional` argument in `dabest.load()`.

        """
        return self.__cohens_h

    @property
    def hedges_g(self):
        """
        Returns an :py:class:`EffectSizeDataFrame` for the standardized mean difference Hedges' `g`, its confidence interval, and relevant statistics, for all comparisons as indicated via the `idx` and `paired` argument in `dabest.load()`.

        """
        return self.__hedges_g

    @property
    def cliffs_delta(self):
        """
        Returns an :py:class:`EffectSizeDataFrame` for Cliff's delta, its confidence interval, and relevant statistics, for all comparisons as indicated via the `idx` and `paired` argument in `dabest.load()`.

        """
        return self.__cliffs_delta

    @property
    def delta_g(self):
        """
        Returns an :py:class:`EffectSizeDataFrame` for deltas' g, its confidence interval, and relevant statistics, for all comparisons as indicated via the `idx` and `paired` argument in `dabest.load()`.
        """
        return self.__delta_g

    @property
    def input_data(self):
        """
        Returns the pandas DataFrame that was passed to `dabest.load()`.
        When `delta2` is True, a new column is added to support the
        function. The name of this new column is indicated by `x`.
        """
        return self.__input_data

    @property
    def idx(self):
        """
        Returns the order of categories that was passed to `dabest.load()`.
        """
        return self.__idx

    @property
    def x1(self):
        """
        Returns the first variable declared in x when it is a delta-delta
        case; returns None otherwise.
        """
        return self.__x1

    @property
    def x1_level(self):
        """
        Returns the levels of first variable declared in x when it is a
        delta-delta case; returns None otherwise.
        """
        return self.__x1_level

    @property
    def x2(self):
        """
        Returns the second variable declared in x when it is a delta-delta
        case; returns None otherwise.
        """
        return self.__x2

    @property
    def experiment(self):
        """
        Returns the column name of experiment labels that was passed to
        `dabest.load()` when it is a delta-delta case; returns None otherwise.
        """
        return self.__experiment

    @property
    def experiment_label(self):
        """
        Returns the experiment labels in order that was passed to `dabest.load()`
        when it is a delta-delta case; returns None otherwise.
        """
        return self.__experiment_label

    @property
    def delta2(self):
        """
        Returns the boolean parameter indicating if this is a delta-delta
        situation.
        """
        return self.__delta2

    @property
    def is_paired(self):
        """
        Returns the type of repeated-measures experiment.
        """
        return self.__is_paired

    @property
    def id_col(self):
        """
        Returns the id column declared to `dabest.load()`.
        """
        return self.__id_col

    @property
    def ci(self):
        """
        The width of the desired confidence interval.
        """
        return self.__ci

    @property
    def resamples(self):
        """
        The number of resamples used to generate the bootstrap.
        """
        return self.__resamples

    @property
    def random_seed(self):
        """
        The number used to initialise the numpy random seed generator, ie.
        `seed_value` from `numpy.random.seed(seed_value)` is returned.
        """
        return self.__random_seed

    @property
    def x(self):
        """
        Returns the x column that was passed to `dabest.load()`, if any.
        When `delta2` is True, `x` returns the name of the new column created
        for the delta-delta situation. To retrieve the 2 variables passed into
        `x` when `delta2` is True, please call `x1` and `x2` instead.
        """
        return self.__x

    @property
    def y(self):
        """
        Returns the y column that was passed to `dabest.load()`, if any.
        """
        return self.__y

    @property
    def _xvar(self):
        """
        Returns the xvar in dabest.plot_data.
        """
        return self.__xvar

    @property
    def _yvar(self):
        """
        Returns the yvar in dabest.plot_data.
        """
        return self.__yvar

    @property
    def _plot_data(self):
        """
        Returns the pandas DataFrame used to produce the estimation stats/plots.
        """
        return self.__plot_data

    @property
    def proportional(self):
        """
        Returns the proportional parameter class.
        """
        return self.__proportional

    @property
    def mini_meta(self):
        """
        Returns the mini_meta boolean parameter.
        """
        return self.__mini_meta

    @property
    def _all_plot_groups(self):
        """
        Returns the all plot groups, as indicated via the `idx` keyword.
        """
        return self.__all_plot_groups

    def _check_errors(self, x, y, idx, experiment, experiment_label, x1_level):
        '''
        Function to check some input parameters and combinations between them.
        At the end of this function these two class attributes are updated
                self.__experiment_label and self.__x1_level
        '''
        # Check if it is a valid mini_meta case
        if self.__mini_meta:
            # Only mini_meta calculation but not proportional and delta-delta function
            if self.__proportional:
                err0 = "`proportional` and `mini_meta` cannot be True at the same time."
                raise ValueError(err0)
            if self.__delta2:
                err0 = "`delta2` and `mini_meta` cannot be True at the same time."
                raise ValueError(err0)

            # Check if the columns stated are valid
            # Initialize a flag to track if any element in idx is neither str nor (tuple, list)
            valid_types = True

            # Initialize variables to track the conditions for str and (tuple, list)
            is_str_condition_met, is_tuple_list_condition_met = False, False

            # Single traversal for optimization
            for item in idx:
                if isinstance(item, str):
                    is_str_condition_met = True
                elif isinstance(item, (tuple, list)) and len(item) == 2:
                    is_tuple_list_condition_met = True
                else:
                    valid_types = False
                    break  # Exit the loop if an invalid type is found

            # Check if all types are valid
            if not valid_types:
                err0 = "`mini_meta` is True, but `idx` ({})".format(idx)
                err1 = "does not contain exactly 2 unique columns."
                raise ValueError(err0 + err1)

            # Handling str type condition
            if is_str_condition_met:
                if len(np.unique(idx).tolist()) != 2:
                    err0 = "`mini_meta` is True, but `idx` ({})".format(idx)
                    err1 = "does not contain exactly 2 unique columns."
                    raise ValueError(err0 + err1)

            # Handling (tuple, list) type condition
            if is_tuple_list_condition_met:
                all_idx_lengths = [len(t) for t in idx]
                if (array(all_idx_lengths) != 2).any():
                    err1 = "`mini_meta` is True, but some elements in idx "
                    err2 = "in {} do not consist only of two groups.".format(idx)
                    raise ValueError(err1 + err2)


        # Check if this is a 2x2 ANOVA case and x & y are valid columns
        # Create experiment_label and x1_level
        elif self.__delta2:
            if x is None:
                error_msg = "If `delta2` is True. `x` parameter cannot be None. String or list expected"
                raise ValueError(error_msg)
                
            if self.__proportional:
                err0 = "`proportional` and `delta2` cannot be True at the same time."
                raise ValueError(err0)

            # idx should not be specified
            if idx:
                err0 = "`idx` should not be specified when `delta2` is True.".format(
                    len(x)
                )
                raise ValueError(err0)

            # Check if x is valid
            if len(x) != 2:
                err0 = "`delta2` is True but the number of variables indicated by `x` is {}.".format(
                    len(x)
                )
                raise ValueError(err0)

            for i in x:
                if i not in self.__output_data.columns:
                    err = "{0} is not a column in `data`. Please check.".format(i)
                    raise IndexError(err)

            # Check if y is valid
            if not y:
                err0 = "`delta2` is True but `y` is not indicated."
                raise ValueError(err0)

            if y not in self.__output_data.columns:
                err = "{0} is not a column in `data`. Please check.".format(y)
                raise IndexError(err)

            # Check if experiment is valid
            if experiment not in self.__output_data.columns:
                err = "{0} is not a column in `data`. Please check.".format(experiment)
                raise IndexError(err)

            # Check if experiment_label is valid and create experiment when needed
            if experiment_label:
                if len(experiment_label) != 2:
                    err0 = "`experiment_label` does not have a length of 2."
                    raise ValueError(err0)

                for i in experiment_label:
                    if i not in self.__output_data[experiment].unique():
                        err = "{0} is not an element in the column `{1}` of `data`. Please check.".format(
                            i, experiment
                        )
                        raise IndexError(err)
            else:
                experiment_label = self.__output_data[experiment].unique()

            # Check if x1_level is valid
            if x1_level:
                if len(x1_level) != 2:
                    err0 = "`x1_level` does not have a length of 2."
                    raise ValueError(err0)

                for i in x1_level:
                    if i not in self.__output_data[x[0]].unique():
                        err = "{0} is not an element in the column `{1}` of `data`. Please check.".format(
                            i, experiment
                        )
                        raise IndexError(err)

            else:
                x1_level = self.__output_data[x[0]].unique()

        elif experiment:
            experiment_label = self.__output_data[experiment].unique()
            x1_level = self.__output_data[x[0]].unique()
        self.__experiment_label = experiment_label
        self.__x1_level = x1_level

    def _get_plot_data(self, x, y, all_plot_groups):
        """
        Function to prepare some attributes for plotting
        """
        # Check if there is NaN under any of the paired settings
        if self.__is_paired is not None and self.__output_data.isnull().values.any():
            print("Nan")
            import warnings
            warn1 = f"NaN values detected under paired setting and removed,"
            warn2 = f" please check your data."
            warnings.warn(warn1 + warn2)
            rmname = self.__output_data[self.__output_data[y].isnull()][self.__id_col].tolist()
            self.__output_data = self.__output_data[~self.__output_data[self.__id_col].isin(rmname)]
                
        # Identify the type of data that was passed in.
        if x is not None and y is not None:
            # Assume we have a long dataset.
            # check both x and y are column names in data.
            if x not in self.__output_data.columns:
                err = "{0} is not a column in `data`. Please check.".format(x)
                raise IndexError(err)
            if y not in self.__output_data.columns:
                err = "{0} is not a column in `data`. Please check.".format(y)
                raise IndexError(err)

            # check y is numeric.
            if not issubdtype(self.__output_data[y].dtype, number):
                err = "{0} is a column in `data`, but it is not numeric.".format(y)
                raise ValueError(err)

            # check all the idx can be found in self.__output_data[x]
            for g in all_plot_groups:
                if g not in self.__output_data[x].unique():
                    err0 = '"{0}" is not a group in the column `{1}`.'.format(g, x)
                    err1 = " Please check `idx` and try again."
                    raise IndexError(err0 + err1)

            # Select only rows where the value in the `x` column
            # is found in `idx`.
            plot_data = self.__output_data[
                self.__output_data.loc[:, x].isin(all_plot_groups)
            ].copy()

            # Assign attributes
            self.__x = x
            self.__y = y
            self.__xvar = x
            self.__yvar = y

        elif x is None and y is None:
            # Assume we have a wide dataset.
            # Assign attributes appropriately.
            self.__x = None
            self.__y = None
            self.__xvar = "group"
            self.__yvar = "value"

            # Check if there is NaN under any of the paired settings
            if self.__is_paired is not None and self.__output_data.isnull().values.any():
                import warnings
                warn1 = f"NaN values detected under paired setting and removed,"
                warn2 = f" please check your data."
                warnings.warn(warn1 + warn2)

            # First, check we have all columns in the dataset.
            for g in all_plot_groups:
                if g not in self.__output_data.columns:
                    err0 = '"{0}" is not a column in `data`.'.format(g)
                    err1 = " Please check `idx` and try again."
                    raise IndexError(err0 + err1)

            set_all_columns = set(self.__output_data.columns.tolist())
            set_all_plot_groups = set(all_plot_groups)
            id_vars = set_all_columns.difference(set_all_plot_groups)

            plot_data = pd.melt(
                self.__output_data,
                id_vars=id_vars,
                value_vars=all_plot_groups,
                value_name=self.__yvar,
                var_name=self.__xvar,
            )

        # Added in v0.2.7.
        plot_data.dropna(axis=0, how="any", subset=[self.__yvar], inplace=True)


        if isinstance(plot_data[self.__xvar].dtype, pd.CategoricalDtype):
            plot_data[self.__xvar].cat.remove_unused_categories(inplace=True)
            plot_data[self.__xvar].cat.reorder_categories(
                all_plot_groups, ordered=True, inplace=True
            )
        else:
            plot_data[self.__xvar] = pd.Categorical(
                plot_data[self.__xvar], categories=all_plot_groups, ordered=True
            )

        return plot_data

    def _compute_effectsize_dfs(self):
        '''
        Function to compute all attributes based on EffectSizeDataFrame.
        It returns nothing.
        '''
        from ._effsize_objects import EffectSizeDataFrame

        effectsize_df_kwargs = dict(
            ci=self.__ci,
            is_paired=self.__is_paired,
            random_seed=self.__random_seed,
            resamples=self.__resamples,
            proportional=self.__proportional,
            delta2=self.__delta2,
            experiment_label=self.__experiment_label,
            x1_level=self.__x1_level,
            x2=self.__x2,
            mini_meta=self.__mini_meta,
        )

        self.__mean_diff = EffectSizeDataFrame(
            self, "mean_diff", **effectsize_df_kwargs
        )

        self.__median_diff = EffectSizeDataFrame(
            self, "median_diff", **effectsize_df_kwargs
        )

        self.__cohens_d = EffectSizeDataFrame(self, "cohens_d", **effectsize_df_kwargs)

        self.__cohens_h = EffectSizeDataFrame(self, "cohens_h", **effectsize_df_kwargs)

        self.__hedges_g = EffectSizeDataFrame(self, "hedges_g", **effectsize_df_kwargs)

        self.__delta_g = EffectSizeDataFrame(self, "delta_g", **effectsize_df_kwargs)

        if not self.__is_paired:
            self.__cliffs_delta = EffectSizeDataFrame(
                self, "cliffs_delta", **effectsize_df_kwargs
            )
        else:
            self.__cliffs_delta = (
                "The data is paired; Cliff's delta is therefore undefined."
            )
