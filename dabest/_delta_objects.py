# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/API/delta_objects.ipynb.

# %% auto 0
__all__ = ['DeltaDelta', 'MiniMetaDelta']

# %% ../nbs/API/delta_objects.ipynb 5
from scipy.stats import norm
import pandas as pd
import numpy as np
from numpy import sort as npsort
from numpy import isnan
from string import Template
import warnings
import datetime as dt

# %% ../nbs/API/delta_objects.ipynb 6
class DeltaDelta(object):
    """
    A class to compute and store the delta-delta statistics for experiments with a 2-by-2 arrangement where two independent variables, A and B, each have two categorical values, 1 and 2. The data is divided into two pairs of two groups, and a primary delta is first calculated as the mean difference between each of the pairs:


    $$\Delta_{1} = \overline{X}_{A_{2}, B_{1}} - \overline{X}_{A_{1}, B_{1}}$$

    $$\Delta_{2} = \overline{X}_{A_{2}, B_{2}} - \overline{X}_{A_{1}, B_{2}}$$


    where $\overline{X}_{A_{i}, B_{j}}$ is the mean of the sample with A = i and B = j, $\Delta$ is the mean difference between two samples.

    A delta-delta value is then calculated as the mean difference between the two primary deltas:


    $$\Delta_{\Delta} = \Delta_{2} - \Delta_{1}$$

    and a deltas' g value is calculated as the mean difference between the two primary deltas divided by
    the standard deviation of the delta-delta value, which is calculated from a pooled variance of the 4 samples:

    $$\Delta_{g} = \frac{\Delta_{\Delta}}{s_{\Delta_{\Delta}}}$$

    $$s_{\Delta_{\Delta}} = \sqrt{\frac{(n_{A_{2}, B_{1}}-1)s_{A_{2}, B_{1}}^2+(n_{A_{1}, B_{1}}-1)s_{A_{1}, B_{1}}^2+(n_{A_{2}, B_{2}}-1)s_{A_{2}, B_{2}}^2+(n_{A_{1}, B_{2}}-1)s_{A_{1}, B_{2}}^2}{(n_{A_{2}, B_{1}} - 1) + (n_{A_{1}, B_{1}} - 1) + (n_{A_{2}, B_{2}} - 1) + (n_{A_{1}, B_{2}} - 1)}}$$

    where $s$ is the standard deviation and $n$ is the sample size.


    """

    def __init__(
        self, effectsizedataframe, permutation_count, bootstraps_delta_delta, ci=95
    ):
        from ._stats_tools import effsize as es
        from ._stats_tools import confint_1group as ci1g
        from ._stats_tools import confint_2group_diff as ci2g

        self.__effsizedf = effectsizedataframe.results
        self.__dabest_obj = effectsizedataframe.dabest_obj
        self.__ci = ci
        self.__resamples = effectsizedataframe.resamples
        self.__effect_size = effectsizedataframe.effect_size
        self.__alpha = ci2g._compute_alpha_from_ci(ci)
        self.__permutation_count = permutation_count
        self.__bootstraps = np.array(self.__effsizedf["bootstraps"])
        self.__control = self.__dabest_obj.experiment_label[0]
        self.__test = self.__dabest_obj.experiment_label[1]

        # Compute the bootstrap delta-delta or deltas' g and the true dela-delta based on the raw data
        if self.__effect_size == "mean_diff":
            self.__bootstraps_delta_delta = bootstraps_delta_delta[2]
            self.__difference = (
                self.__effsizedf["difference"][1] - self.__effsizedf["difference"][0]
            )
        else:
            self.__bootstraps_delta_delta = bootstraps_delta_delta[0]
            self.__difference = bootstraps_delta_delta[1]

        sorted_delta_delta = npsort(self.__bootstraps_delta_delta)

        self.__bias_correction = ci2g.compute_meandiff_bias_correction(
            self.__bootstraps_delta_delta, self.__difference
        )

        self.__jackknives = np.array(
            ci1g.compute_1group_jackknife(self.__bootstraps_delta_delta, np.mean)
        )

        self.__acceleration_value = ci2g._calc_accel(self.__jackknives)

        # Compute BCa intervals.
        bca_idx_low, bca_idx_high = ci2g.compute_interval_limits(
            self.__bias_correction, self.__acceleration_value, self.__resamples, ci
        )

        self.__bca_interval_idx = (bca_idx_low, bca_idx_high)

        if ~isnan(bca_idx_low) and ~isnan(bca_idx_high):
            self.__bca_low = sorted_delta_delta[bca_idx_low]
            self.__bca_high = sorted_delta_delta[bca_idx_high]

            err1 = "The $lim_type limit of the interval"
            err2 = "was in the $loc 10 values."
            err3 = "The result should be considered unstable."
            err_temp = Template(" ".join([err1, err2, err3]))

            if bca_idx_low <= 10:
                warnings.warn(
                    err_temp.substitute(lim_type="lower", loc="bottom"), stacklevel=1
                )

            if bca_idx_high >= self.__resamples - 9:
                warnings.warn(
                    err_temp.substitute(lim_type="upper", loc="top"), stacklevel=1
                )

        else:
            err1 = "The $lim_type limit of the BCa interval cannot be computed."
            err2 = "It is set to the effect size itself."
            err3 = "All bootstrap values were likely all the same."
            err_temp = Template(" ".join([err1, err2, err3]))

            if isnan(bca_idx_low):
                self.__bca_low = self.__difference
                warnings.warn(err_temp.substitute(lim_type="lower"), stacklevel=0)

            if isnan(bca_idx_high):
                self.__bca_high = self.__difference
                warnings.warn(err_temp.substitute(lim_type="upper"), stacklevel=0)

        # Compute percentile intervals.
        pct_idx_low = int((self.__alpha / 2) * self.__resamples)
        pct_idx_high = int((1 - (self.__alpha / 2)) * self.__resamples)

        self.__pct_interval_idx = (pct_idx_low, pct_idx_high)
        self.__pct_low = sorted_delta_delta[pct_idx_low]
        self.__pct_high = sorted_delta_delta[pct_idx_high]

    def __permutation_test(self):
        """
        Perform a permutation test and obtain the permutation p-value
        based on the permutation data.
        """
        self.__permutations = np.array(self.__effsizedf["permutations"])

        THRESHOLD = np.abs(self.__difference)

        self.__permutations_delta_delta = np.array(
            self.__permutations[1] - self.__permutations[0]
        )

        count = sum(np.abs(self.__permutations_delta_delta) > THRESHOLD)
        self.__pvalue_permutation = count / self.__permutation_count

    def __repr__(self, header=True, sigfig=3):
        from .misc_tools import print_greeting

        first_line = {"control": self.__control, "test": self.__test}

        if self.__effect_size == "mean_diff":
            out1 = "The delta-delta between {control} and {test} ".format(**first_line)
        else:
            out1 = "The deltas' g between {control} and {test} ".format(**first_line)

        base_string_fmt = "{:." + str(sigfig) + "}"
        if "." in str(self.__ci):
            ci_width = base_string_fmt.format(self.__ci)
        else:
            ci_width = str(self.__ci)

        ci_out = {
            "es": base_string_fmt.format(self.__difference),
            "ci": ci_width,
            "bca_low": base_string_fmt.format(self.__bca_low),
            "bca_high": base_string_fmt.format(self.__bca_high),
        }

        out2 = "is {es} [{ci}%CI {bca_low}, {bca_high}].".format(**ci_out)
        out = out1 + out2

        if header is True:
            out = print_greeting() + "\n" + "\n" + out

        pval_rounded = base_string_fmt.format(self.pvalue_permutation)

        p1 = "The p-value of the two-sided permutation t-test is {}, ".format(
            pval_rounded
        )
        p2 = "calculated for legacy purposes only. "
        pvalue = p1 + p2

        bs1 = "{} bootstrap samples were taken; ".format(self.__resamples)
        bs2 = "the confidence interval is bias-corrected and accelerated."
        bs = bs1 + bs2

        pval_def1 = (
            "Any p-value reported is the probability of observing the "
            + "effect size (or greater),\nassuming the null hypothesis of "
            + "zero difference is true."
        )
        pval_def2 = (
            "\nFor each p-value, 5000 reshuffles of the "
            + "control and test labels were performed."
        )
        pval_def = pval_def1 + pval_def2

        return "{}\n{}\n\n{}\n{}".format(out, pvalue, bs, pval_def)

    def to_dict(self):
        """
        Returns the attributes of the `DeltaDelta` object as a
        dictionary.
        """
        # Only get public (user-facing) attributes.
        attrs = [a for a in dir(self) if not a.startswith(("_", "to_dict"))]
        out = {}
        for a in attrs:
            out[a] = getattr(self, a)
        return out

    @property
    def ci(self):
        """
        Returns the width of the confidence interval, in percent.
        """
        return self.__ci

    @property
    def alpha(self):
        """
        Returns the significance level of the statistical test as a float
        between 0 and 1.
        """
        return self.__alpha

    @property
    def bias_correction(self):
        return self.__bias_correction

    @property
    def bootstraps(self):
        """
        Return the bootstrapped deltas from all the experiment groups.
        """
        return self.__bootstraps

    @property
    def jackknives(self):
        return self.__jackknives

    @property
    def acceleration_value(self):
        return self.__acceleration_value

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
    def bca_interval_idx(self):
        return self.__bca_interval_idx

    @property
    def control(self):
        """
        Return the name of the control experiment group.
        """
        return self.__control

    @property
    def test(self):
        """
        Return the name of the test experiment group.
        """
        return self.__test

    @property
    def bootstraps_delta_delta(self):
        """
        Return the delta-delta values calculated from the bootstrapped
        deltas.
        """
        return self.__bootstraps_delta_delta

    @property
    def difference(self):
        """
        Return the delta-delta value calculated based on the raw data.
        """
        return self.__difference

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

    @property
    def pvalue_permutation(self):
        try:
            return self.__pvalue_permutation
        except AttributeError:
            self.__permutation_test()
            return self.__pvalue_permutation

    @property
    def permutation_count(self):
        """
        The number of permuations taken.
        """
        return self.__permutation_count

    @property
    def permutations(self):
        """
        Return the mean differences of permutations obtained during
        the permutation test for each experiment group.
        """
        try:
            return self.__permutations
        except AttributeError:
            self.__permutation_test()
            return self.__permutations

    @property
    def permutations_delta_delta(self):
        """
        Return the delta-delta values of permutations obtained
        during the permutation test.
        """
        try:
            return self.__permutations_delta_delta
        except AttributeError:
            self.__permutation_test()
            return self.__permutations_delta_delta

# %% ../nbs/API/delta_objects.ipynb 10
class MiniMetaDelta(object):
    """
    A class to compute and store the weighted delta.
    A weighted delta is calculated if the argument ``mini_meta=True`` is passed during ``dabest.load()``.
    
    """

    def __init__(self, effectsizedataframe, permutation_count,
                ci=95):
        from ._stats_tools import effsize as es
        from ._stats_tools import confint_1group as ci1g
        from ._stats_tools import confint_2group_diff as ci2g
        
        self.__effsizedf         = effectsizedataframe.results
        self.__dabest_obj        = effectsizedataframe.dabest_obj
        self.__ci                = ci
        self.__resamples         = effectsizedataframe.resamples
        self.__alpha             = ci2g._compute_alpha_from_ci(ci)
        self.__permutation_count = permutation_count
        self.__bootstraps        = np.array(self.__effsizedf["bootstraps"])
        self.__control           = np.array(self.__effsizedf["control"])
        self.__test              = np.array(self.__effsizedf["test"])
        self.__control_N         = np.array(self.__effsizedf["control_N"])
        self.__test_N            = np.array(self.__effsizedf["test_N"])


        idx  = self.__dabest_obj.idx
        dat  = self.__dabest_obj._plot_data
        xvar = self.__dabest_obj._xvar
        yvar = self.__dabest_obj._yvar

        # compute the variances of each control group and each test group
        control_var=[]
        test_var=[]
        grouped_data = {name: group[yvar].copy() for name, group in dat.groupby(xvar)}
        for j, current_tuple in enumerate(idx):
            cname = current_tuple[0]
            control = grouped_data[cname]
            control_var.append(np.var(control, ddof=1))

            tname = current_tuple[1]
            test = grouped_data[tname]
            test_var.append(np.var(test, ddof=1))
        self.__control_var = np.array(control_var)
        self.__test_var    = np.array(test_var)

        # Compute pooled group variances for each pair of experiment groups
        # based on the raw data
        self.__group_var   = ci2g.calculate_group_var(self.__control_var, 
                                                 self.__control_N,
                                                 self.__test_var, 
                                                 self.__test_N)

        # Compute the weighted average mean differences of the bootstrap data
        # using the pooled group variances of the raw data as the inverse of 
        # weights
        self.__bootstraps_weighted_delta = ci2g.calculate_weighted_delta(
                                                          self.__group_var, 
                                                          self.__bootstraps)

        # Compute the weighted average mean difference based on the raw data
        self.__difference = es.weighted_delta(self.__effsizedf["difference"],
                                                   self.__group_var)

        sorted_weighted_deltas = npsort(self.__bootstraps_weighted_delta)


        self.__bias_correction = ci2g.compute_meandiff_bias_correction(
                                    self.__bootstraps_weighted_delta, self.__difference)
        
        self.__jackknives = np.array(ci1g.compute_1group_jackknife(
                                                self.__bootstraps_weighted_delta, 
                                                np.mean))

        self.__acceleration_value = ci2g._calc_accel(self.__jackknives)

        # Compute BCa intervals.
        bca_idx_low, bca_idx_high = ci2g.compute_interval_limits(
            self.__bias_correction, self.__acceleration_value,
            self.__resamples, ci)
        
        self.__bca_interval_idx = (bca_idx_low, bca_idx_high)

        if ~isnan(bca_idx_low) and ~isnan(bca_idx_high):
            self.__bca_low  = sorted_weighted_deltas[bca_idx_low]
            self.__bca_high = sorted_weighted_deltas[bca_idx_high]

            err1 = "The $lim_type limit of the interval"
            err2 = "was in the $loc 10 values."
            err3 = "The result should be considered unstable."
            err_temp = Template(" ".join([err1, err2, err3]))

            if bca_idx_low <= 10:
                warnings.warn(err_temp.substitute(lim_type="lower",
                                                  loc="bottom"),
                              stacklevel=1)

            if bca_idx_high >= self.__resamples-9:
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
        pct_idx_low  = int((self.__alpha/2)     * self.__resamples)
        pct_idx_high = int((1-(self.__alpha/2)) * self.__resamples)

        self.__pct_interval_idx = (pct_idx_low, pct_idx_high)
        self.__pct_low          = sorted_weighted_deltas[pct_idx_low]
        self.__pct_high         = sorted_weighted_deltas[pct_idx_high]
        
    

    def __permutation_test(self):
        """
        Perform a permutation test and obtain the permutation p-value
        based on the permutation data.
        """
        self.__permutations     = np.array(self.__effsizedf["permutations"])
        self.__permutations_var = np.array(self.__effsizedf["permutations_var"])

        THRESHOLD = np.abs(self.__difference)

        all_num = []
        all_denom = []

        groups = len(self.__permutations)
        for i in range(0, len(self.__permutations[0])):
            weight = [1/self.__permutations_var[j][i] for j in range(0, groups)]
            all_num.append(np.sum([weight[j]*self.__permutations[j][i] for j in range(0, groups)]))
            all_denom.append(np.sum(weight))
        
        output=[]
        for i in range(0, len(all_num)):
            output.append(all_num[i]/all_denom[i])
        
        self.__permutations_weighted_delta = np.array(output)

        count = sum(np.abs(self.__permutations_weighted_delta)>THRESHOLD)
        self.__pvalue_permutation = count/self.__permutation_count



    def __repr__(self, header=True, sigfig=3):
        from .misc_tools import print_greeting
        
        is_paired = self.__dabest_obj.is_paired

        PAIRED_STATUS = {'baseline'   : 'paired', 
                         'sequential' : 'paired',
                         'None'       : 'unpaired'
        }

        first_line = {"paired_status": PAIRED_STATUS[str(is_paired)]}
        

        out1 = "The weighted-average {paired_status} mean differences ".format(**first_line)
        
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
        out = out1 + out2

        if header is True:
            out = print_greeting() + "\n" + "\n" + out


        pval_rounded = base_string_fmt.format(self.pvalue_permutation)

        
        p1 = "The p-value of the two-sided permutation t-test is {}, ".format(pval_rounded)
        p2 = "calculated for legacy purposes only. "
        pvalue = p1 + p2


        bs1 = "{} bootstrap samples were taken; ".format(self.__resamples)
        bs2 = "the confidence interval is bias-corrected and accelerated."
        bs = bs1 + bs2

        pval_def1 = "Any p-value reported is the probability of observing the" + \
                    "effect size (or greater),\nassuming the null hypothesis of " + \
                    "zero difference is true."
        pval_def2 = "\nFor each p-value, 5000 reshuffles of the " + \
                    "control and test labels were performed."
        pval_def = pval_def1 + pval_def2


        return "{}\n{}\n\n{}\n{}".format(out, pvalue, bs, pval_def)


    def to_dict(self):
        """
        Returns all attributes of the `dabest.MiniMetaDelta` object as a
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
    def ci(self):
        """
        Returns the width of the confidence interval, in percent.
        """
        return self.__ci


    @property
    def alpha(self):
        """
        Returns the significance level of the statistical test as a float
        between 0 and 1.
        """
        return self.__alpha


    @property
    def bias_correction(self):
        return self.__bias_correction


    @property
    def bootstraps(self):
        '''
        Return the bootstrapped differences from all the experiment groups.
        '''
        return self.__bootstraps


    @property
    def jackknives(self):
        return self.__jackknives


    @property
    def acceleration_value(self):
        return self.__acceleration_value


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
    def bca_interval_idx(self):
        return self.__bca_interval_idx


    @property
    def control(self):
        '''
        Return the names of the control groups from all the experiment 
        groups in order.
        '''
        return self.__control


    @property
    def test(self):
        '''
        Return the names of the test groups from all the experiment 
        groups in order.
        '''
        return self.__test
    
    @property
    def control_N(self):
        '''
        Return the sizes of the control groups from all the experiment 
        groups in order.
        '''
        return self.__control_N


    @property
    def test_N(self):
        '''
        Return the sizes of the test groups from all the experiment 
        groups in order.
        '''
        return self.__test_N


    @property
    def control_var(self):
        '''
        Return the estimated population variances of the control groups 
        from all the experiment groups in order. Here the population 
        variance is estimated from the sample variance. 
        '''
        return self.__control_var


    @property
    def test_var(self):
        '''
        Return the estimated population variances of the control groups 
        from all the experiment groups in order. Here the population 
        variance is estimated from the sample variance. 
        '''
        return self.__test_var

    
    @property
    def group_var(self):
        '''
        Return the pooled group variances of all the experiment groups 
        in order. 
        '''
        return self.__group_var


    @property
    def bootstraps_weighted_delta(self):
        '''
        Return the weighted-average mean differences calculated from the bootstrapped 
        deltas and weights across the experiment groups, where the weights are 
        the inverse of the pooled group variances.
        '''
        return self.__bootstraps_weighted_delta


    @property
    def difference(self):
        '''
        Return the weighted-average delta calculated from the raw data.
        '''
        return self.__difference


    @property
    def pct_interval_idx (self):
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


    @property
    def pvalue_permutation(self):
        try:
            return self.__pvalue_permutation
        except AttributeError:
            self.__permutation_test()
            return self.__pvalue_permutation
    

    @property
    def permutation_count(self):
        """
        The number of permuations taken.
        """
        return self.__permutation_count

    
    @property
    def permutations(self):
        '''
        Return the mean differences of permutations obtained during
        the permutation test for each experiment group.
        '''
        try:
            return self.__permutations
        except AttributeError:
            self.__permutation_test()
            return self.__permutations


    @property
    def permutations_var(self):
        '''
        Return the pooled group variances of permutations obtained during
        the permutation test for each experiment group.
        '''
        try:
            return self.__permutations_var
        except AttributeError:
            self.__permutation_test()
            return self.__permutations_var

    
    @property
    def permutations_weighted_delta(self):
        '''
        Return the weighted-average deltas of permutations obtained 
        during the permutation test.
        '''
        try:
            return self.__permutations_weighted_delta
        except AttributeError:
            self.__permutation_test()
            return self.__permutations_weighted_delta


