#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com



def create_jackknife_indexes(data):
    """
    Given an array-like, creates a jackknife bootstrap.

    For a given set of data Y, the jackknife bootstrap sample J[i]
    is defined as the data set Y with the ith data point deleted.

    Keywords
    --------
    data: array-like

    Returns
    -------
    Generator that yields all jackknife bootstrap samples.
    """
    from numpy import arange, delete

    index_range = arange(0, len(data))
    return (delete(index_range, i) for i in index_range)



def create_repeated_indexes(data):
    """
    Convenience function. Given an array-like with length N,
    returns a generator that yields N indexes [0, 1, ..., N].
    """
    from numpy import arange

    index_range = arange(0, len(data))
    return (index_range for i in index_range)



def _create_two_group_jackknife_indexes(x0, x1, is_paired):
    """Creates the jackknife bootstrap for 2 groups."""

    if is_paired and len(x0) == len(x1):
        out = list(zip([j for j in create_jackknife_indexes(x0)],
                       [i for i in create_jackknife_indexes(x1)]
                       )
                   )
    else:
        jackknife_c = list(zip([j for j in create_jackknife_indexes(x0)],
                               [i for i in create_repeated_indexes(x1)]
                              )
                          )

        jackknife_t = list(zip([i for i in create_repeated_indexes(x0)],
                               [j for j in create_jackknife_indexes(x1)]
                              )
                          )
        out = jackknife_c + jackknife_t
        del jackknife_c
        del jackknife_t

    return out



def compute_meandiff_jackknife(x0, x1, is_paired, effect_size):
    """
    Given two arrays, returns the jackknife for their effect size.
    """
    from . import effsize as __es

    jackknives = _create_two_group_jackknife_indexes(x0, x1, is_paired)

    out = []

    for j in jackknives:
        x0_shuffled = x0[j[0]]
        x1_shuffled = x1[j[1]]

        es = __es.two_group_difference(x0_shuffled, x1_shuffled,
                                       is_paired, effect_size)
        out.append(es)

    return out



def _calc_accel(jack_dist):
    from numpy import mean as npmean
    from numpy import sum as npsum
    from numpy import errstate

    jack_mean = npmean(jack_dist)

    numer = npsum((jack_mean - jack_dist)**3)
    denom = 6.0 * (npsum((jack_mean - jack_dist)**2) ** 1.5)

    with errstate(invalid='ignore'):
        # does not raise warning if invalid division encountered.
        return numer / denom



def compute_mean_diff_bootstraps(x0, x1, is_paired, effect_size,
                                resamples=5000, random_seed=12345):
    """Bootstraps the effect_size for 2 groups."""
    from . import effsize as __es
    import numpy as np

    np.random.seed(random_seed)

    out = np.repeat(np.nan, resamples)
    x0_len = len(x0)
    x1_len = len(x1)

    for i in range(int(resamples)):
        x0_boot = np.random.choice(x0, x0_len, replace=True)
        x1_boot = np.random.choice(x1, x1_len, replace=True)
        out[i] = __es.two_group_difference(x0_boot, x1_boot,
                                          is_paired, effect_size)

    # reset seed
    np.random.seed()

    return out



def compute_meandiff_bias_correction(bootstraps, effsize):
    """
    Computes the bias correction required for the BCa method
    of confidence interval construction.

    Keywords
    --------
    bootstraps: array-like
        An numerical iterable, comprising bootstrap resamples
        of the effect size.

    effsize: numeric
        The effect size for the original sample.


    Returns
    -------
    bias: numeric
        The bias correction value for the given bootstraps
        and effect size.

    """
    from scipy.stats import norm
    from numpy import array

    B = array(bootstraps)
    prop_less_than_es = sum(B < effsize) / len(B)

    return norm.ppf(prop_less_than_es)



def _compute_alpha_from_ci(ci):
    if ci < 0 or ci > 100:
        raise ValueError("`ci` must be a number between 0 and 100.")

    return (100. - ci) / 100.



def _compute_quantile(z, bias, acceleration):
    numer = bias + z
    denom = 1 - (acceleration * numer)

    return bias + (numer / denom)



def compute_interval_limits(bias, acceleration, n_boots, ci=95):
    """
    Returns the indexes of the interval limits for a given bootstrap.

    Supply the bias, acceleration factor, and number of bootstraps.
    """
    from scipy.stats import norm
    from numpy import isnan, nan

    alpha = _compute_alpha_from_ci(ci)

    alpha_low = alpha / 2
    alpha_high = 1 - (alpha / 2)

    z_low = norm.ppf(alpha_low)
    z_high = norm.ppf(alpha_high)

    kws = {'bias': bias, 'acceleration': acceleration}
    low = _compute_quantile(z_low, **kws)
    high = _compute_quantile(z_high, **kws)

    if isnan(low) or isnan(high):
        return nan, nan

    else:
        low = int(norm.cdf(low) * n_boots)
        high = int(norm.cdf(high) * n_boots)
        return low, high



# def __get_true_intervals(xx):
#     from numpy import sort
#     low  = sort(xx.bootstraps)[xx["index_low"]]
#     high = sort(xx.bootstraps)[xx["index_high"]]
#     return low, high


#
# def difference_ci(x0, x1, effect_size,
#                 is_paired=False, resamples=5000, alpha=0.05,
#                 random_seed=12345):
#     """
#     Given an two array-likes x0 and x1, returns the effect size along with bootstrap confidence intervals. Available effect sizes are:
#     - unstandardized mean difference
#     - median difference
#     - standardized mean differences
#         - Cohen's d
#         - Hedges' g
#     - Cliff's delta
#
#     Keywords
#     --------
#     x0, x1: array-like
#         These should be numerical iterables.
#
#     effect_size: string.
#         Any one of the following are accepted inputs:
#         'mean_diff', 'median_diff', 'cohens_d', 'hedges_g', or 'cliffs_delta'
#
#     is_paired: boolean, default False
#
#     resamples: int, default 5000
#         The number of bootstrap resamples to be taken.
#
#     alpha: float, default 0.05
#         Denotes the likelihood that the confidence interval produced
#         _does not_ include the true imean difference. When alpha = 0.05,
#         a 95% confidence interval is produced.
#
#     random_seed: int, default 12345
#         `random_seed` is used to seed the random number generator during
#         bootstrap resampling. This ensures that the confidence intervals
#         reported are replicable.
#
#
#     Returns
#     -------
#     A pandas DataFrame with the various effect sizes as the index, and with the
#     following columns:
#         effect_size:    The effect size of interest.
#         bias:           Bias as computed according to the method of Efron and
#                         Tibshirani.
#         acceleration:   Accelration as computed according to the method of
#                         Efron and Tibshirani.
#         index_low:      The index of the sorted bootstrap array corresponding to
#                         the lower confidence interval bound.
#         index_high:     The index of the sorted bootstrap array corresponding to
#                         the upper confidence interval bound.
#         bca_ci_low:     The bias corrected and accelerated lower bound of the
#                         confidence interval.
#         bca_ci_high:    The bias corrected and accelerated upper bound of the
#                         confidence interval.
#         bootstraps:     The unsorted of resamples used to compute the confidence
#                         interval.
#
#
#     References
#     ----------
#     Efron, B., & Tibshirani, R. J. (1993). An introduction to the bootstrap.
#     New York: Chapman & Hall.
#
#     """
#     from numpy import sort as npsort
#     from numpy import array, vectorize, isnan
#     from pandas import Series, DataFrame, merge
#     from . import effsize
#
#     EFFECT_SIZES = ['mean_diff', 'median_diff',
#                     'cohens_d', 'hedges_g', 'cliffs_delta']
#     if effect_size not in EFFECT_SIZES:
#         err1 = "{} is not a recognized effect size.\n".format(effect_size)
#         err2 = "Select from "
#         err3 = "'mean_diff', 'median_diff', 'cohens_d', 'hedges_g', 'cliffs_delta'"
#         raise ValueError(err1 + err2 + err3)
#
#     es_ = effsize.two_group_difference(x0, x1, is_paired)
#     es  =  es_[effect_size]
#
#     jackknives = compute_meandiff_jackknife(x0, x1, effect_size, is_paired)
#     acceleration_value = _calc_accel(jackknives)
#     del jackknives # for memory management
#
#     bootstraps = compute_mean_diff_bootstraps(x0, x1, is_paired, effect_size,
#                                               resamples, random_seed)
#     bias_correction = compute_meandiff_bias_correction(bootstraps, es)
#
#
#
#     effsizes = DataFrame([es, bias_correction, acceleration_value],
#                          index=["effect_size", "bias", "acceleration"]).T
#
#
#     # First, apply `compute_interval_limits` to the "bias"
#     # and "acceleration" columns.
#     # This produces a tuple.
#     # Then, apply `pandas.Series` to split each tuple into 2 columns.
#     interval_lims = effsizes.apply(lambda x: compute_interval_limits(
#                                         x["bias"], x["acceleration"],
#                                         resamples),
#                                    axis=1)\
#                             .apply(Series)
#
#     # Rename the columns
#     interval_lims.columns = ["index_low", "index_high"]
#
#     # Merge effect sizes with interval limits.
#     effsizes = merge(effsizes, interval_lims,
#                        left_index=True, right_index=True)
#
#     # Join with bootstraps.
#     boots_df = DataFrame([[v] for v in bootstraps.values()],
#                          index=bootstraps.keys(), columns=['bootstraps'])
#     del bootstraps # for memory management
#     effsizes = merge(effsizes, boots_df, left_index=True, right_index=True)
#
#     # Get intervals from `effsizes`
#     intervals = effsizes.apply(lambda x: __get_true_intervals(x), axis=1)\
#                         .apply(Series)
#     intervals.columns = ["bca_ci_low", "bca_ci_high"]
#     effsizes = merge(effsizes, intervals,
#                        left_index=True, right_index=True)
#
#     return effsizes
