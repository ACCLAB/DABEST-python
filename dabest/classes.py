#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com

class Dabest:
    '''
    Class for estimation statistics and plots.
    '''

    def __init__(self, data, idx, x=None, y=None, paired=False):
        """
        Create a Dabest object.
        This is designed to work with pandas DataFrames.

        Keywords
        --------
        data: pandas DataFrame.

        idx: tuple.
            List of column names (if 'x' is not supplied) or of category names
            (if 'x' is supplied). This can be expressed as a tuple of tuples,
            with each individual tuple producing its own contrast plot.

        x, y: strings, default None.
            Column names for data to be plotted on the x-axis and y-axis.
        """

        # Import standard data science libraries.
        import numpy as np
        import pandas as pd
        import seaborn as sns

        self.__data  = data
        self.__idx   = idx
        self.__is_paired = paired

        # Make a copy of the data, so we don't make alterations to it.
        data_in = data.copy()
        data_in.reset_index(inplace=True)


        # Determine the kind of estimation plot we need to produce.
        if all([isinstance(i, str) for i in idx]):
            self.__plottype = "hubspoke"

            # Set columns and width ratio.
            ncols = 1
            ngroups = len(idx)
            widthratio = [1]

            # flatten out idx.
            all_plot_groups = pd.unique([t for t in idx]).tolist()
            self.__all_plot_groups = all_plot_groups

            self._idx_for_plotting = (idx,)

        elif all([isinstance(i, (tuple, list)) for i in idx]):
            self.__plottype = "multigroup"

            all_plot_groups = np.unique([tt for t in idx for tt in t]).tolist()
            self.__all_plot_groups = all_plot_groups

            widthratio = [len(ii) for ii in idx]
            # Set columns and width ratio.
            ncols = len(idx)
            ngroups = len(all_plot_groups)

            self._idx_for_plotting = idx

        else: # mix of string and tuple?
            err = 'There seems to be a problem with the idx you'
            'entered--{}.'.format(idx)
            raise ValueError(err)


        # Sanity checks.
        if x is None and y is not None:
            err = 'You have only specified `y`. Please also specify `x`.'
            raise ValueError(err)

        elif y is None and x is not None:
            err = 'You have only specified `x`. Please also specify `y`.'
            raise ValueError(err)

        elif x is not None and y is not None:
            # Assume we have a long dataset.
            # check both x and y are column names in data.
            if x not in data_in.columns:
                err = '{0} is not a column in `data`. Please check.'.format(x)
                raise IndexError(err)
            if y not in data_in.columns:
                err = '{0} is not a column in `data`. Please check.'.format(y)
                raise IndexError(err)

            # check y is numeric.
            if not np.issubdtype(data_in[y].dtype, np.number):
                err = '{0} is a column in `data`, but it is not numeric.'.format(y)
                raise ValueError(err)

            # check all the idx can be found in data_in[x]
            for g in self.__all_plot_groups:
                if g not in data_in[x].unique():
                    raise IndexError('{0} is not a group in `{1}`.'.format(g, x))

            plot_data = data_in[data_in.loc[:, x].isin(self.__all_plot_groups)].copy()
            plot_data.drop("index", inplace=True, axis=1)

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

            # First, check we have all columns in the dataset.
            for g in self.__all_plot_groups:
                if g not in data_in.columns:
                    raise IndexError('{0} is not a column in `data`.'.format(g))

            # Extract only the columns being plotted.
            plot_data = data_in.reindex(columns=self.__all_plot_groups).copy()

            plot_data = pd.melt(plot_data,
                                value_vars=self.__all_plot_groups,
                                value_name=self.__yvar,
                                var_name=self.__xvar)

        plot_data.loc[:, self.__xvar] = pd.Categorical(plot_data[self.__xvar],
                                           categories=self.__all_plot_groups,
                                           ordered=True)
        self.__plot_data = plot_data


        self.mean_diff    = EffectSizeDataFrame(self, "mean_diff",
                                             self.__is_paired)

        self.median_diff  = EffectSizeDataFrame(self, "median_diff",
                                               self.__is_paired)

        self.cohens_d     = EffectSizeDataFrame(self, "cohens_d",
                                            self.__is_paired)

        self.hedges_g     = EffectSizeDataFrame(self, "hedges_g",
                                            self.__is_paired)

        self.cliffs_delta = EffectSizeDataFrame(self, "cliffs_delta",
                                                self.__is_paired)




    def __repr__(self):
        repr1 = "UNDER DEVELOPMENT. Ideally, this should return a nice summary"
        repr2 = " of the data, along with the comparisons to be done."

        return repr1 + repr2



    @property
    def data(self):
        """
        Returns the pandas DataFrame that was passed to `dabest.load()`.
        """
        return self.__data

    @property
    def idx(self):
        """
        Returns the order of categories that was passed to `dabest.load()`.
        """
        return self.__idx

    @property
    def is_paired(self):
        """
        Returns True if the dataset was declared as paired to `Dabest.load()`.
        """
        return self.__is_paired

    @property
    def x(self):
        """
        Returns the x column that was passed to `dabest.load()`, if any.
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






class TwoGroupsEffectSize(object):

    """A class to compute and store the results of bootstrapped
    mean differences between two groups."""

    def __init__(self, control, test, effect_size,
                 is_paired=False, ci=95,
                 resamples=5000, random_seed=12345):

        """
        Compute the effect size between two groups.

        Keywords
        --------
            control, test: array-like
                These should be numerical iterables.

            effect_size: string.
                Any one of the following are accepted inputs:
                'mean_diff', 'median_diff', 'cohens_d', 'hedges_g', or 'cliffs_delta'

            is_paired: boolean, default False

            resamples: int, default 5000
                The number of bootstrap resamples to be taken.

            ci: float, default 95
                Denotes the likelihood that the confidence interval produced
                _does not_ include the true imean difference. When alpha = 0.05,
                a 95% confidence interval is produced.

            random_seed: int, default 12345
                `random_seed` is used to seed the random number generator during
                bootstrap resampling. This ensures that the confidence intervals
                reported are replicable.

        Returns
        -------
            A `TwoGroupEffectSize` object.

        Examples
        --------
            >>> import numpy as np
            >>> import scipy as sp
            >>> import dabest
            >>> np.random.seed(12345)
            >>> control = sp.stats.norm.rvs(loc=0, size=30)
            >>> test = sp.stats.norm.rvs(loc=0.5, size=30)
            >>> effsize = dabest.TwoGroupsEffectSize(control, test, "mean_diff")
            >>> effsize
            The unpaired mean difference is -0.253 [95%CI -0.782, 0.241]

            5000 bootstrap samples. The confidence interval is bias-corrected
            and accelerated.
        """

        from numpy import array, isnan
        from numpy import sort as npsort
        from numpy.random import choice, seed

        from .stats_tools import confint_2group_diff as ci2g
        from .stats_tools import effsize as es

        from string import Template
        import warnings

        self.__EFFECT_SIZE_DICT =  {"mean_diff" : "mean difference",
                                    "median_diff" : "median difference",
                                    "cohens_d" : "Cohen's d",
                                    "hedges_g" : "Hedges' g",
                                    "cliffs_delta" : "Cliff's delta"}

        kosher_es = [a for a in self.__EFFECT_SIZE_DICT.keys()]
        if effect_size not in kosher_es:
            err1 = "The effect size '{}'".format(effect_size)
            err2 = "is not one of {}".format(kosher_es)
            raise ValueError(" ".join([err1, err2]))

        # Convert to numpy arrays for speed.
        # NaNs are automatically dropped.
        control = array(control)
        test    = array(test)
        control = control[~isnan(control)]
        test    = test[~isnan(test)]

        self.__effect_size = effect_size
        self.__control     = control
        self.__test        = test
        self.__is_paired   = is_paired
        self.__resamples   = resamples
        self.__random_seed = random_seed
        self.__ci          = ci
        self.__alpha       = ci2g._compute_alpha_from_ci(ci)


        self.__difference = es.two_group_difference(
                                control, test, is_paired, effect_size)

        self.__jackknives = ci2g.compute_meandiff_jackknife(
                                control, test, is_paired, effect_size)

        self.__acceleration_value = ci2g._calc_accel(self.__jackknives)

        bootstraps = ci2g.compute_mean_diff_bootstraps(
                            control, test, is_paired, effect_size,
                            resamples, random_seed)
        self.__bootstraps = npsort(bootstraps)

        self.__bias_correction = ci2g.compute_meandiff_bias_correction(
                                    self.__bootstraps, self.__difference)

        # Compute BCa intervals.
        bca_idx_low, bca_idx_high = ci2g.compute_interval_limits(
            self.__bias_correction, self.__acceleration_value,
            self.__resamples, ci)

        self.__bca_interval_idx = (bca_idx_low, bca_idx_high)

        if ~isnan(bca_idx_low) and ~isnan(bca_idx_high):
            self.__bca_low  = self.__bootstraps[bca_idx_low]
            self.__bca_high = self.__bootstraps[bca_idx_high]

            err1 = "The $lim_type limit of the interval"
            err2 = "was in the $loc 10 values."
            err3 = "The result should be considered unstable."
            err_temp = Template(" ".join([err1, err2, err3]))

            if bca_idx_low <= 10:
                warnings.warn(err_temp.substitute(lim_type="lower",
                                                  loc="bottom"),
                              stacklevel=1)

            if bca_idx_high >= resamples-9:
                warnings.warn(err_temp.substitute(lim_type="upper",
                                                  loc="top"),
                              stacklevel=1)

        else:
            err1 = "The $lim_type limit of the BCa interval cannot be computed."
            err2 = "It is set to the effect size itself."
            err3 = "All bootstrap values were likely all the same."
            err_temp = Template(" ".join([err1, err2, err3]))

            if isnan(bca_idx_low):
                self.__bca_low  = self.__difference
                warnings.warn(err_temp.substitute(lim_type="lower"),
                              stacklevel=0)

            if isnan(bca_idx_high):
                self.__bca_high  = self.__difference
                warnings.warn(err_temp.substitute(lim_type="upper"),
                              stacklevel=0)

        # Compute percentile intervals.
        pct_idx_low  = int((self.__alpha/2)     * resamples)
        pct_idx_high = int((1-(self.__alpha/2)) * resamples)

        self.__pct_interval_idx = (pct_idx_low, pct_idx_high)
        self.__pct_low  = self.__bootstraps[pct_idx_low]
        self.__pct_high = self.__bootstraps[pct_idx_high]



    def __repr__(self, sigfig=3):

        PAIRED_STATUS = {True: 'paired', False: 'unpaired'}

        first_line = {"is_paired": PAIRED_STATUS[self.__is_paired],
                      "es"       : self.__EFFECT_SIZE_DICT[self.__effect_size]}

        out1 = "The {is_paired} {es}".format(**first_line)

        base_string_fmt = "{:." + str(sigfig) + "}"
        if "." in str(self.__ci):
            ci_width = base_string_fmt.format(self.__ci)
        else:
            ci_width = str(self.__ci)

        ci_out = {"es"       : base_string_fmt.format(self.__difference),
                  "ci"       : ci_width,
                  "bca_low"  : base_string_fmt.format(self.__bca_low),
                  "bca_high" : base_string_fmt.format(self.__bca_high)}

        out2 = "is {es} [{ci}%CI {bca_low}, {bca_high}].".format(**ci_out)

        out3 = "{} bootstrap samples were taken;".format(self.__resamples)

        out4 = "the confidence interval is bias-corrected and accelerated."

        return "{} {}\n{} {}".format(out1, out2, out3, out4)



    def to_dict(self):
        """
        Returns the attributes of the `dabest.TwoGroupEffectSize` object as a
        dictionary.
        """
        # Only get public (user-facing) attributes.
        attrs = [a for a in dir(self)
                 if not a.startswith(("_", "to_dict"))]
        out = {}
        for a in attrs:
            out[a] = getattr(self, a)
        return out




    @property
    def difference(self):
        """
        Returns the difference between the control and the test.
        """
        return self.__difference

    @property
    def effect_size(self):
        """
        Returns the type of effect size reported.
        """
        return self.__EFFECT_SIZE_DICT[self.__effect_size]

    @property
    def is_paired(self):
        return self.__is_paired

    @property
    def ci(self):
        """
        Returns the width of the confidence interval.
        """
        return self.__ci

    @property
    def resamples(self):
        """
        The number of resamples performed during the bootstrap procedure.
        """
        return self.__resamples

    @property
    def bootstraps(self):
        """
        The generated bootstraps of the effect size.
        """
        return self.__bootstraps

    @property
    def random_seed(self):
        """
        The number used to initialise the numpy random seed generator, ie.
        `seed_value` from `numpy.random.seed(seed_value)` is returned.
        """
        return self.__random_seed

    @property
    def bca_interval_idx(self):
        return self.__bca_interval_idx

    @property
    def bca_low(self):
        """
        The bias-corrected and accelerated confidence interval lower limit.
        """
        return self.__bca_low

    @property
    def bca_high(self):
        """
        The bias-corrected and accelerated confidence interval upper limit.
        """
        return self.__bca_high

    @property
    def pct_interval_idx(self):
        return self.__pct_interval_idx

    @property
    def pct_low(self):
        """
        The percentile confidence interval lower limit.
        """
        return self.__pct_low

    @property
    def pct_high(self):
        """
        The percentile confidence interval lower limit.
        """
        return self.__pct_high






class EffectSizeDataFrame(object):
    """A class that generates and stores the results of bootstrapped effect
    sizes for several comparisons."""

    def __init__(self, dabest, effect_size,
                 is_paired, ci=95,
                 resamples=5000, random_seed=12345):
        """
        Takes the data parsed from a Dabest object, and gives a plotting and
        printing capability.

        Keywords
        --------

        Examples
        --------
        """

        self.__idx         = dabest._idx_for_plotting
        self.__plot_data   = dabest._plot_data
        self.__xvar        = dabest._xvar
        self.__yvar        = dabest._yvar
        self.__effect_size = effect_size
        self.__is_paired   = is_paired
        self.__ci          = ci
        self.__resamples   = resamples
        self.__random_seed = random_seed



    def __pre_calc(self):
        import pandas as pd

        dat = self.__plot_data
        out = []
        reprs = []

        for j, current_tuple in enumerate(self.__idx):

            cname = current_tuple[0]
            control = dat[dat[self.__xvar] == cname][self.__yvar].copy()

            for ix, tname in enumerate(current_tuple[1:]):
                test = dat[dat[self.__xvar] == tname][self.__yvar].copy()

                result = TwoGroupsEffectSize(control, test,
                                             self.__effect_size,
                                             self.__is_paired,
                                             self.__ci,
                                             self.__resamples,
                                             self.__random_seed)
                r_dict = result.to_dict()

                r_dict["control"] = cname
                r_dict["test"] = tname
                out.append(r_dict)

                to_replace = "between {} and {} is".format(cname, tname)
                text_repr = result.__repr__().replace("is", to_replace, 1)
                reprs.append(text_repr)

        self.__for_print = "\n\n".join(reprs)

        out_             = pd.DataFrame(out)

        columns_in_order = ['control', 'test', 'effect_size', 'is_paired',
                            'difference', 'ci',

                            'bca_low', 'bca_high', 'bca_interval_idx',
                            'pct_low', 'pct_high', 'pct_interval_idx',

                            'bootstraps', 'resamples', 'random_seed']

        self.__results   = out_.reindex(columns=columns_in_order)


    def __repr__(self):
        try:
            return self.__for_print
        except AttributeError:
            self.__pre_calc()
            return self.__for_print




    def plot(self):
        if hasattr(self, "results"):
            return "Plotting functions under DEVELOPMENT."
        else:
            return "first run precalc, then plot."




    @property
    def results(self):
        """Prints all pairwise comparisons nicely."""
        try:
            return self.__results
        except AttributeError:
            self.__pre_calc()
            return self.__results



    @property
    def _for_print(self):
        return self.__for_print

    @property
    def _plot_data(self):
        return self.__plot_data

    @property
    def _idx(self):
        return self.__idx

    @property
    def _xvar(self):
        return self.__xvar

    @property
    def _yvar(self):
        return self.__yvar

    @property
    def _is_paired(self):
        return self.__is_paired

    @property
    def _ci(self):
        return self.__ci

    @property
    def _resamples(self):
        return self.__resamples

    @property
    def _random_seed(self):
        return self.__random_seed
