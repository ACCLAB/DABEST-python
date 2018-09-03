#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com


from . import effect_sizes as __es
__effect_sizes = __es.__effect_sizes


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
    from numpy import arange, delete

    index_range = arange(0, len(data))
    return (index_range for i in index_range)



def __create_two_group_jackknife_indexes(x0, x1):
    """Creates the jackknife bootstrap for 2 groups."""

    jackknife_c = list(zip([j for j in create_jackknife_indexes(x0)],
                           [i for i in create_repeated_indexes(x1)]
                          )
                      )

    jackknife_t = list(zip([i for i in create_repeated_indexes(x0)],
                           [j for j in create_jackknife_indexes(x1)]
                          )
                      )
    return jackknife_c + jackknife_t



def compute_meandiff_jackknife(x0, x1, paired=False):
    """
    Returns the jackknife for the mean difference, Cohen's d, and Hedges' g.
    """

    jack_dict = {}
    for a in __effect_sizes:
        jack_dict[a] = []

    jackknives = __create_two_group_jackknife_indexes(x0, x1)

    for j in jackknives:
        x0_shuffled = x0[j[0]]
        x1_shuffled = x1[j[1]]

        e = __es.mean_difference(x0_shuffled, x1_shuffled,
                                 paired=paired)

        for a in __effect_sizes:
            jackknife = e[a]
            jack_dict[a].append(jackknife)

    # Create namedtuple for easy output
    return jack_dict



def __calc_accel(jack_dist):
    from numpy import mean as npmean
    from numpy import sum as npsum

    jack_mean = npmean(jack_dist)

    numer = npsum((jack_mean - jack_dist)**3)
    denom = 6.0 * (npsum((jack_mean - jack_dist)**2) ** 1.5)

    return numer / denom



def compute_acceleration(jack_dist):

    acc = dict()
    for a in __effect_sizes:
        acc[a] = []

    for a in __effect_sizes:
        effsize_jack = jack_dist[a]
        acc[a] = __calc_accel(effsize_jack)

    return acc



def __create_bootstrap_indexes(size, resamples=5000, random_seed=12345):
    from numpy.random import choice, seed

    # Set seed.
    seed(random_seed)

    for i in range(int(resamples)):
        yield choice(size, size=size, replace=True)

    # reset seed
    seed()



def compute_mean_diff_bootstraps(x0, x1, paired=False,
                                 resamples=5000, random_seed=12345):
    """Bootstraps the mean difference, Cohen's d, and Hedges' g for 2 groups."""

    bs_index_kwargs = {'resamples': resamples,
                       'random_seed': random_seed}

    boots_dict = {}
    for a in __effect_sizes:
        boots_dict[a] = []

    x0_bs_idx = __create_bootstrap_indexes(len(x0), **bs_index_kwargs)
    x1_bs_idx = __create_bootstrap_indexes(len(x1), **bs_index_kwargs)

    boot_indexes = list(zip(x0_bs_idx, x1_bs_idx))

    for b in boot_indexes:
        x0_boot = x0[b[0]]
        x1_boot = x1[b[1]]

        e = __es.mean_difference(x0_boot, x1_boot,
                               paired=paired)

        for a in __effect_sizes:
            bootstrap = e[a]
            boots_dict[a].append(bootstrap)

    return boots_dict



def compute_bias_correction(bootstraps, effect_size):
    """
    Computes the bias correction required for the BCa method
    of confidence interval construction.

    Keywords
    --------
    bootstraps: array-like
        An numerical iterable, comprising bootstrap resamples
        of the effect size.

    effect_size: numeric
        The effect size for the original sample.


    Returns
    -------
    bias_dict: dictionary
        The bias correction value for the given bootstraps
        and effect size.

    """
    from scipy.stats import norm

    bias_dict = {}

    for a in __effect_sizes:
        B = bootstraps[a]
        e = effect_size[a]

        prop_less_than_es = sum(B < e) / len(B)

        bias_dict[a] = norm.ppf(prop_less_than_es)

    return bias_dict



def __compute_quantile(z, bias, acceleration):
    numer = bias + z
    denom = 1 - (acceleration * numer)

    return bias + (numer / denom)



def compute_interval_limits(bias, acceleration, n_boots, alpha=0.05):
    """
    Returns the indexes of the interval limits for a given bootstrap.
    Supply the bias, acceleration factor, and array of bootstraps.
    """
    from scipy.stats import norm

    alpha_low = alpha / 2
    alpha_high = 1 - (alpha / 2)

    z_low = norm.ppf(alpha_low)
    z_high = norm.ppf(alpha_high)

    quant = {'bias': bias, 'acceleration': acceleration}
    low = __compute_quantile(z_low, **quant)
    high = __compute_quantile(z_high, **quant)

    low = int(norm.cdf(low) * n_boots)
    high = int(norm.cdf(high) * n_boots)

    return low, high
