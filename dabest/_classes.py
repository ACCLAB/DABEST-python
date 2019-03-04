#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com

class Dabest:

    """
    Class for estimation statistics and plots.
    """

    def __init__(self, data, idx, x, y, paired, id_col, ci, resamples,
                random_seed):

        """
        Parses and stores pandas DataFrames in preparation for estimation
        statistics.
        """

        # Import standard data science libraries.
        import numpy as np
        import pandas as pd
        import seaborn as sns

        self.__ci          = ci
        self.__data        = data
        self.__idx         = idx
        self.__id_col      = id_col
        self.__is_paired   = paired
        self.__resamples   = resamples
        self.__random_seed = random_seed

        # Make a copy of the data, so we don't make alterations to it.
        data_in = data.copy()
        # data_in.reset_index(inplace=True)
        # data_in_index_name = data_in.index.name



        # Determine the kind of estimation plot we need to produce.
        if all([isinstance(i, str) for i in idx]):
            # flatten out idx.
            all_plot_groups = pd.unique([t for t in idx]).tolist()
            # We need to re-wrap this idx inside another tuple so as to
            # easily loop thru each pairwise group later on.
            self.__idx = (idx,)

        elif all([isinstance(i, (tuple, list)) for i in idx]):
            all_plot_groups = pd.unique([tt for t in idx for tt in t]).tolist()

        else: # mix of string and tuple?
            err = 'There seems to be a problem with the idx you'
            'entered--{}.'.format(idx)
            raise ValueError(err)

        # Having parsed the idx, check if it is a kosher paired plot,
        # if so stated.
        if paired is True:
            all_idx_lengths = [len(t) for t in self.__idx]
            if (np.array(all_idx_lengths) != 2).any():
                err1 = "`is_paired` is True, but some idx "
                err2 = "in {} does not consist only of two groups.".format(idx)
                raise ValueError(err1 + err2)


        # Determine the type of data: wide or long.
        if x is None and y is not None:
            err = 'You have only specified `y`. Please also specify `x`.'
            raise ValueError(err)

        elif y is None and x is not None:
            err = 'You have only specified `x`. Please also specify `y`.'
            raise ValueError(err)

        # Identify the type of data that was passed in.
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
            for g in all_plot_groups:
                if g not in data_in[x].unique():
                    raise IndexError('{0} is not a group in `{1}`.'.format(g, x))

            plot_data = data_in[data_in.loc[:, x].isin(all_plot_groups)].copy()
            # plot_data.drop("index", inplace=True, axis=1)

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
            for g in all_plot_groups:
                if g not in data_in.columns:
                    raise IndexError('{0} is not a column in `data`.'.format(g))

            set_all_columns     = set(data_in.columns.tolist())
            set_all_plot_groups = set(all_plot_groups)
            id_vars = set_all_columns.difference(set_all_plot_groups)

            plot_data = pd.melt(data_in,
                                id_vars=id_vars,
                                value_vars=all_plot_groups,
                                value_name=self.__yvar,
                                var_name=self.__xvar)

        plot_data.loc[:, self.__xvar] = pd.Categorical(plot_data[self.__xvar],
                                           categories=all_plot_groups,
                                           ordered=True)

        self.__plot_data = plot_data

        self.__all_plot_groups = all_plot_groups

        # Sanity check that all idxs are paired, if so desired.
        if paired is True:
            if id_col is None:
                err = "`id_col` must be specified if `is_paired` is set to True."
                raise IndexError(err)
            elif id_col not in plot_data.columns:
                err = "{} is not a column in `data`. ".format(id_col)
                raise IndexError(err)


        EffectSizeDataFrame_kwargs = dict(ci=ci, is_paired=paired,
                                          random_seed=random_seed,
                                          resamples=resamples)

        self.mean_diff    = EffectSizeDataFrame(self, "mean_diff",
                                                **EffectSizeDataFrame_kwargs)

        self.median_diff  = EffectSizeDataFrame(self, "median_diff",
                                               **EffectSizeDataFrame_kwargs)

        self.cohens_d     = EffectSizeDataFrame(self, "cohens_d",
                                                **EffectSizeDataFrame_kwargs)

        self.hedges_g     = EffectSizeDataFrame(self, "hedges_g",
                                                **EffectSizeDataFrame_kwargs)

        self.cliffs_delta = EffectSizeDataFrame(self, "cliffs_delta",
                                                **EffectSizeDataFrame_kwargs)




    def __repr__(self):
        from .__init__ import __version__
        import datetime as dt
        import numpy as np

        from .misc_tools import print_greeting

        if self.__is_paired:
            es = "Paired e"
        else:
            es = "E"

        greeting_header = print_greeting()

        s1 = "{}ffect size(s) ".format(es)
        s2 = "with {}% confidence intervals will be computed for:".format(self.__ci)
        desc_line = s1 + s2

        out = [greeting_header + "\n\n" + desc_line]

        comparisons = []

        for j, current_tuple in enumerate(self.__idx):
            control_name = current_tuple[0]

            for ix, test_name in enumerate(current_tuple[1:]):
                comparisons.append("{} minus {}".format(test_name, control_name))

        for j, g in enumerate(comparisons):
            out.append("{}. {}".format(j+1, g))

        resamples_line1 = "\n{} resamples ".format(self.__resamples)
        resamples_line2 = "will be used to generate the effect size bootstraps."
        out.append(resamples_line1 + resamples_line2)

        return "\n".join(out)




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
        Returns True if the dataset was declared as paired to `dabest.load()`.
        """
        return self.__is_paired

    @property
    def id_col(self):
        """
        Returns the ic column declared to `dabest.load()`.
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
    def _all_plot_groups(self):
        """
        Returns the all plot groups, as indicated via the `idx` keyword.
        """
        return self.__all_plot_groups






class TwoGroupsEffectSize(object):

    """
    A class to compute and store the results of bootstrapped
    mean differences between two groups.
    """

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
            The confidence interval width. The default of 95 produces 95%
            confidence intervals.

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

        from ._stats_tools import confint_2group_diff as ci2g
        from ._stats_tools import effsize as es

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

        bootstraps = ci2g.compute_bootstrapped_diff(
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



    def __repr__(self, show_resample_count=True, sigfig=3):

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

        if show_resample_count:
            return "{} {}\n\n{} {}".format(out1, out2, out3, out4)
        else:
            return "{} {}".format(out1, out2)

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
        Returns the width of the confidence interval, in percent
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
        Parses the data from a Dabest object, enabling plotting and printing
        capability for the effect size of interest.
        """

        self.__dabest_obj = dabest
        self.__effect_size  = effect_size
        self.__is_paired    = is_paired
        self.__ci           = ci
        self.__resamples    = resamples
        self.__random_seed  = random_seed



    def __pre_calc(self):
        import pandas as pd
        from .misc_tools import print_greeting

        idx  = self.__dabest_obj.idx
        dat  = self.__dabest_obj._plot_data
        xvar = self.__dabest_obj._xvar
        yvar = self.__dabest_obj._yvar

        out = []
        reprs = []

        for j, current_tuple in enumerate(idx):

            cname = current_tuple[0]
            control = dat[dat[xvar] == cname][yvar].copy()

            for ix, tname in enumerate(current_tuple[1:]):
                test = dat[dat[xvar] == tname][yvar].copy()

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

                if j == len(idx)-1 and ix == len(current_tuple)-2:
                    resamp_count = True
                else:
                    resamp_count = False

                text_repr = result.__repr__(show_resample_count=resamp_count)

                to_replace = "between {} and {} is".format(cname, tname)
                text_repr = text_repr.replace("is", to_replace, 1)

                reprs.append(text_repr)

        reprs.insert(0, print_greeting())

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



    def plot(self, color_col=None,

            raw_marker_size=6, es_marker_size=9,

            swarm_label=None, contrast_label=None,
            swarm_ylim=None, contrast_ylim=None,

            custom_palette=None,
            float_contrast=True,
            show_pairs=True,
            group_summaries=None,

            fig_size=None,
            dpi=100,
            halfviolin_alpha=0.8,

            swarmplot_kwargs=None,
            violinplot_kwargs=None,
            slopegraph_kwargs=None,
            reflines_kwargs=None,
            group_summary_kwargs=None,
            legend_kwargs=None):
        """
        Creates an estimation plot for the effect size of interest.

        Keywords
        --------
        color_col: string, default None
            Column to be used for colors.

        raw_marker_size: float, default 6
            The diameter (in points) of the marker dots plotted in the
            swarmplot.

        es_marker_size: float, default 9
            The size (in points) of the effect size points on the difference
            axes.

        swarm_label, contrast_label: strings, default None
            Set labels for the y-axis of the swarmplot and the contrast plot,
            respectively. If `swarm_label` is not specified, it defaults to
            "value", unless a column name was passed to `y`. If
            `contrast_label` is not specified, it defaults to the effect size
            being plotted.

        swarm_ylim, contrast_ylim: tuples, default None
            The desired y-limits of the raw data (swarmplot) axes and the
            difference axes respectively, as a tuple. These will be autoscaled
            to sensible values if they are not specified.

        custom_palette: dict, list, or matplotlib color palette, default None
            This keyword accepts a dictionary with {'group':'color'} pairings,
            a list of RGB colors, or a specified matplotlib palette. This
            palette will be used to color the swarmplot. If `color_col` is not
            specified, then each group will be colored in sequence according
            to the default palette currently used by matplotlib.

            Please take a look at the seaborn commands `color_palette`
            and `cubehelix_palette` to generate a custom palette. Both
            these functions generate a list of RGB colors.

            See:
            https://seaborn.pydata.org/generated/seaborn.color_palette.html
            https://seaborn.pydata.org/generated/seaborn.cubehelix_palette.html

            The named colors of matplotlib can be found here:
            https://matplotlib.org/examples/color/named_colors.html

        float_contrast: boolean, default True
            Whether or not to display the halfviolin bootstrapped difference
            distribution alongside the raw data.

        show_pairs: boolean, default True
            If the data is paired, whether or not to show the raw data as a
            swarmplot, or as slopegraph, with a line joining each pair of
            observations.

        group_summaries: ['mean_sd', 'median_quartiles', 'None'], default None.
            Plots the summary statistics for each group. If 'mean_sd', then
            the mean and standard deviation of each group is plotted as a
            notched line beside each group. If 'median_quantiles', then the
            median and 25th and 75th percentiles of each group is plotted
            instead. If 'None', the summaries are not shown.

        fig_size: tuple, default None
            The desired dimensions of the figure as a (length, width) tuple.

        dpi: int, default 100
            The dots per inch of the resulting figure.

        halfviolin_alpha: float, default 0.8
            The alpha level (transparency) of the half-violinplot used to
            depict the bootstrapped distribution of the effect size(s).

        swarmplot_kwargs: dict, default None
            Pass any keyword arguments accepted by the seaborn `swarmplot`
            command here, as a dict. If None, the following keywords are
            passed to sns.swarmplot: {'size':`raw_marker_size`}.

        violinplot_kwargs: dict, default None
            Pass any keyword arguments accepted by the matplotlib `
            pyplot.violinplot` command here, as a dict. If None, the following
            keywords are passed to violinplot: {'widths':0.5, 'vert':True,
            'showextrema':False, 'showmedians':False}.

        reflines_kwargs: dict, default None
            This will change the appearance of the zero reference lines. Pass
            any keyword arguments accepted by the matplotlib Axes `hlines`
            command here, as a dict. If None, the following keywords are
            passed to Axes.hlines: {'linestyle':'solid', 'linewidth':0.75,
            'zorder':2, 'color': default y-tick color}.

        group_summary_kwargs: dict, default None
            Pass any keyword arguments accepted by the matplotlib.lines.Line2D
            command here, as a dict. This will change the appearance of the
            vertical summary lines for each group, if `group_summaries` is not
            'None'. If None, the following keywords are passed to
            matplotlib.lines.Line2D: {'lw':2, 'alpha':1, 'zorder':3}.

        legend_kwargs: dict, default None
            Pass any keyword arguments accepted by the matplotlib Axes
            `legend` command here, as a dict. If None, the following keywords
            are passed to matplotlib.Axes.legend: {'loc':'upper left',
            'frameon':False}.


        Returns
        -------
        A matplotlib Figure :class:`matplotlib.figure.Figure` with 2 Axes.

        The first axes (accessible with ``FigName.axes()[0]``) contains the
        rawdata swarmplot; the second axes (accessible with
        ``FigName.axes()[1]``) has the bootstrap distributions and effect sizes
        (with confidence intervals) plotted on it.

        Examples
        --------
        Create a Gardner-Altman estimation plot for the mean difference.

        >>> my_data = dabest.load(df, idx=("Control 1", "Test 1"))
        >>> fig1 = my_data.mean_diff.plot()

        Create a Gardner-Altman plot for the Hedges' g effect size.

        >>> fig2 = my_data.hedges_g.plot()

        Create a Cumming estimation plot for the mean difference.

        >>> fig3 = my_data.mean_diff.plot(float_contrast=True)

        Create a paired Gardner-Altman plot.

        >>> my_data_paired = dabest.load(df, idx=("Control 1", "Test 1"),
        ...                              paired=True)
        >>> fig4 = my_data_paired.mean_diff.plot()

        Create a multi-group Cumming plot.

        >>> my_multi_groups = dabest.load(df, idx=(("Control 1", "Test 1"),
        ...                                        ("Control 2", "Test 2"))
        ...                               )
        >>> fig5 = my_multi_groups.mean_diff.plot()

        Create a shared control Cumming plot.

        >>> my_shared_control = dabest.load(df, idx=("Control 1", "Test 1",
        ...                                          "Test 2", "Test 3")
        ...                                 )
        >>> fig6 = my_shared_control.mean_diff.plot()

        """

        from .plotter import EffectSizeDataFramePlotter

        if hasattr(self, "results") is False:
            self.__pre_calc()

        all_kwargs = locals()
        del all_kwargs["self"]

        out = EffectSizeDataFramePlotter(self, **all_kwargs)

        return out


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
        return self.__dabest_obj._plot_data

    @property
    def idx(self):
        return self.__dabest_obj.idx

    @property
    def xvar(self):
        return self.__dabest_obj._xvar

    @property
    def yvar(self):
        return self.__dabest_obj._yvar

    @property
    def is_paired(self):
        return self.__is_paired

    @property
    def ci(self):
        """
        The width of the confidence interval being produced, in percent.
        """
        return self.__ci

    @property
    def resamples(self):
        """
        The number of resamples (with replacement) during bootstrap resampling."
        """
        return self.__resamples

    @property
    def random_seed(self):
        """
        The seed used by `numpy.seed()` for bootstrap resampling.
        """
        return self.__random_seed

    @property
    def effect_size(self):
        """The type of effect size being computed."""
        return self.__effect_size

    @property
    def dabest_obj(self):
        """
        Returns the `dabest` object that invoked the current EffectSizeDataFrame
        class.
        """
        return self.__dabest_obj
