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



def compute_bootstrapped_diff(x0, x1, is_paired, effect_size,
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
        return low, high

    else:
        low = int(norm.cdf(low) * n_boots)
        high = int(norm.cdf(high) * n_boots)
        return low, high
