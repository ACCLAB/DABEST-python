#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com
"""
A range of functions to compute bootstraps for a single sample.
"""

def create_bootstrap_indexes(array, resamples=5000, random_seed=12345):
    """Given an array-like, returns a generator of bootstrap indexes
    to be used for resampling.
    """
    import numpy as np
    
    # Set seed.
    np.random.seed(random_seed)
    
    indexes = range(0, len(array))

    out = (np.random.choice(indexes, len(indexes), replace=True)
            for i in range(0, resamples))
    
    # Reset seed
    np.random.seed()
    
    return out

def compute_1group_jackknife(x, func, *args, **kwargs):
    """
    Returns the jackknife bootstraps for func(x).
    """
    from . import confint_2group_diff as ci_2g
    jackknives = [i for i in ci_2g.create_jackknife_indexes(x)]
    out = [func(x[j], *args, **kwargs) for j in jackknives]
    del jackknives # memory management.
    return out



def compute_1group_acceleration(jack_dist):
    from . import confint_2group_diff as ci_2g
    return ci_2g._calc_accel(jack_dist)



def compute_1group_bootstraps(x, func, resamples=5000, random_seed=12345,
                             *args, **kwargs):
    """Bootstraps func(x), with the number of specified resamples."""

    import numpy as np
    
    # Instantiate random seed.
    np.random.seed(random_seed)
    
    # Create bootstrap indexes.
    boot_indexes = create_bootstrap_indexes(x, resamples=resamples,
                                            random_seed=random_seed)

    out = [func(x[b], *args, **kwargs) for b in boot_indexes]
    
    del boot_indexes
    
    np.random.seed()
    
    return out



def compute_1group_bias_correction(x, bootstraps, func, *args, **kwargs):
    from scipy.stats import norm
    metric = func(x, *args, **kwargs)
    prop_boots_less_than_metric = sum(bootstraps < metric) / len(bootstraps)

    return norm.ppf(prop_boots_less_than_metric)



def summary_ci_1group(x, func, resamples=5000, alpha=0.05, random_seed=12345,
                      sort_bootstraps=True, *args, **kwargs):
    """
    Given an array-like x, returns func(x), and a bootstrap confidence
    interval of func(x).

    Keywords
    --------
    x: array-like
        An numerical iterable.

    func: function
        The function to be applied to x.

    resamples: int, default 5000
        The number of bootstrap resamples to be taken of func(x).

    alpha: float, default 0.05
        Denotes the likelihood that the confidence interval produced
        _does not_ include the true summary statistic. When alpha = 0.05,
        a 95% confidence interval is produced.
    random_seed: int, default 12345
        `random_seed` is used to seed the random number generator during
        bootstrap resampling. This ensures that the confidence intervals
        reported are replicable.
        
    sort_bootstraps: boolean, default True
    


    Returns
    -------
    A dictionary with the following five keys:
        'summary': float.
            The outcome of func(x).

        'func': function.
            The function applied to x.

        'bca_ci_low': float
        'bca_ci_high': float.
            The bias-corrected and accelerated confidence interval, for the
            given alpha.

        'bootstraps': array.
            The bootstraps used to generate the confidence interval.
            These will be sorted in ascending order if `sort_bootstraps`
            was True.

    """
    from . import confint_2group_diff as ci2g
    from numpy import sort as npsort

    boots = compute_1group_bootstraps(x, func, resamples=resamples,
                                      random_seed=random_seed,
                                      *args, **kwargs)
    bias = compute_1group_bias_correction(x, boots, func)

    jk = compute_1group_jackknife(x, func, *args, **kwargs)
    accel = compute_1group_acceleration(jk)
    del jk

    ci_idx = ci2g.compute_interval_limits(bias, accel, resamples, alpha)

    boots_sorted = npsort(boots)

    low = boots_sorted[ci_idx[0]]
    high = boots_sorted[ci_idx[1]]

    if sort_bootstraps:
        B = boots_sorted
    else:
        B = boots
    del boots
    del boots_sorted

    out = {'summary': func(x), 'func': func,
            'bca_ci_low': low, 'bca_ci_high': high,
            'bootstraps': B}

    del B
    return out
