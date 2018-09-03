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



def create_two_group_jackknife_indexes(x0, x1):
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



def compute_meandiff_jackknife(x0, x1, jackknives, paired=False):
    """
    Returns the jackknife for the mean difference, Cohen's d, and Hedges' g.
    """
    import effectsizes as es
    from collections import namedtuple

    jack_dict = {}
    for a in __effect_sizes:
        jack_dict[a] = []

    for j in jackknives:
        x0_shuffled = x0[j[0]]
        x1_shuffled = x1[j[1]]

        e = es.mean_difference(x0_shuffled, x1_shuffled,
                               paired=paired)
        for a in __effect_sizes:
            jackknife = getattr(e, a)
            jack_dict[a].append(jackknife)

    # Create namedtuple for easy output
    jk = namedtuple('JackKnives', __effect_sizes)
    return jk(*jack_dict.values())



def __calc_accel(jack_dist):
    from numpy import mean as npmean
    from numpy import sum as npsum

    jack_mean = npmean(jack_dist)

    numer = npsum((jack_mean - jack_dist)**3)
    denom = 6.0 * (npsum((jack_mean - jack_dist)**2) ** 1.5)

    return numer / denom



def compute_acceleration(jack_dist):

    from collections import namedtuple

    # Create namedtuple for easy output
    acc = namedtuple('Acceleration', __effect_sizes)

    results = []
    for a in __effect_sizes:
        effsize = getattr(jack_dist, a)
        results.append(__calc_accel(effsize))

    return acc(*results)



def create_bootstrap_indexes(size, resamples=5000, random_seed=12345):
    from numpy.random import choice, seed

    # Set seed.
    seed(random_seed)

    for i in range(int(resamples)):
        yield choice(size, size=size, replace=True)

    # reset seed
    seed()



def compute_mean_diff_bootstraps(x0, x1, paired=False):
    """Bootstraps the mean difference, Cohen's d, and Hedges' g for 2 groups."""
    import effectsizes as es
    from collections import namedtuple

    boots_dict = {}
    for a in __effect_sizes:
        boots_dict[a] = []

    boot_indexes = list(zip(create_bootstrap_indexes(len(x0)),
                            create_bootstrap_indexes(len(x1))))

    for b in boot_indexes:
        x0_boot = x0[b[0]]
        x1_boot = x1[b[1]]

        e = es.mean_difference(x0_boot, x1_boot,
                               paired=paired)

        for a in __effect_sizes:
            bootstrap = getattr(e, a)
            boots_dict[a].append(bootstrap)

    # Create namedtuple for easy output
    B = namedtuple('bootstraps', __effect_sizes)
    return B(*boots_dict.values())



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
    z0: float
        The bias correction value for the given bootstraps
        and effect size.

    """
    from scipy.stats import norm
    from collections import namedtuple

    # Create namedtuple for easy output
    attributes = effect_size._fields

    bias_list = []

    for a in attributes:
        B = getattr(bootstraps, a)
        e = getattr(effect_size, a)

        prop_less_than_es = sum(B < e) / len(B)

        bias_list.append(norm.ppf(prop_less_than_es))

    out = namedtuple('Bias', attributes)
    return out(*bias_list)
