#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com


# from . import effsize as __es
# __effect_sizes = __es.__effect_sizes


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



def _create_two_group_jackknife_indexes(x0, x1, paired):
    """Creates the jackknife bootstrap for 2 groups."""

    if paired and len(x0) == len(x1):
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



def compute_meandiff_jackknife(x0, x1, paired):
    """
    Returns the jackknife for the mean difference, Cohen's d, and Hedges' g.
    """
    from . import effsize as __es
    jack_dict = {}
    # Introspection: compute a throwaway mean difference to get the correct
    # effect sizes. This allows us to alter the standardized mean differences
    # computed in effsize.mean_difference(), and still gain access to it.
    _temp_md = __es.two_group_difference(x0, x1, paired)
    for eff_s in _temp_md.keys():
        jack_dict[eff_s] = []
    del _temp_md

    jackknives = _create_two_group_jackknife_indexes(x0, x1, paired)

    for j in jackknives:
        x0_shuffled = x0[j[0]]
        x1_shuffled = x1[j[1]]

        e = __es.two_group_difference(x0_shuffled, x1_shuffled,
                                 paired=paired)

        for eff_size in e.keys():
            jackknife = e[eff_size]
            jack_dict[eff_size].append(jackknife)

    del jackknives

    return jack_dict



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



def compute_meandiff_acceleration(jack_dist, mean_diff):
    """
    Given a jackknife distribution for mean_differences, computes the
    acceleration value.
    """
    acc = dict()

    for eff_size in mean_diff.keys():
        effsize_jack = jack_dist[eff_size]
        acc[eff_size] = _calc_accel(effsize_jack)

    return acc



def _create_bootstrap_indexes(size, resamples=5000, random_seed=12345):
    from numpy.random import choice, seed

    # Set seed.
    seed(random_seed)

    for i in range(int(resamples)):
        yield choice(size, size=size, replace=True)

    # reset seed
    seed()



def compute_mean_diff_bootstraps(x0, x1, paired, resamples=5000,
                                 random_seed=12345):
    """Bootstraps the mean difference, Cohen's d, Hedges' g, and Cliff's delta
     for 2 groups."""
    from . import effsize as __es

    bs_index_kwargs = {'resamples': resamples,
                       'random_seed': random_seed}

    boots_dict = {}

    # Introspection: compute a throwaway mean difference to get the correct
    # effect sizes. This allows us to alter the standardized mean differences
    # computed in effsize.mean_difference(), and still gain access to it.
    _temp_md = __es.two_group_difference(x0, x1, paired)
    for eff_s in _temp_md.keys():
        boots_dict[eff_s] = []
    del _temp_md

    x0_bs_idx = _create_bootstrap_indexes(len(x0), **bs_index_kwargs)
    x1_bs_idx = _create_bootstrap_indexes(len(x1), **bs_index_kwargs)

    boot_indexes = list(zip(x0_bs_idx, x1_bs_idx))

    del x0_bs_idx
    del x1_bs_idx

    for b in boot_indexes:
        x0_boot = x0[b[0]]
        x1_boot = x1[b[1]]

        e = __es.two_group_difference(x0_boot, x1_boot, paired)

        for eff_size in e.keys():
            bootstrap = e[eff_size]
            boots_dict[eff_size].append(bootstrap)

    del boot_indexes
    return boots_dict



def compute_meandiff_bias_correction(bootstraps, effect_size):
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
    from numpy import array

    bias_dict = {}

    for effsize in effect_size.keys():
        B = array(bootstraps[effsize])
        e = effect_size[effsize]

        prop_less_than_es = sum(B < e) / len(B)

        bias_dict[effsize] = norm.ppf(prop_less_than_es)

    return bias_dict



def _compute_quantile(z, bias, acceleration):
    numer = bias + z
    denom = 1 - (acceleration * numer)

    return bias + (numer / denom)



def compute_interval_limits(bias, acceleration, n_boots, alpha=0.05):
    """
    Returns the indexes of the interval limits for a given bootstrap.

    Supply the bias, acceleration factor, and number of bootstraps.
    """
    from scipy.stats import norm
    from numpy import isnan, nan

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



def difference_ci(x0, x1, paired=False, resamples=5000, alpha=0.05,
                 random_seed=12345):
    """
    Given an two array-likes x0 and x1, returns the unstandardized mean
    difference, standardized mean differences (Cohen's d and Hedges' g),
    and Glass' delta, along with bootstrap confidence intervals for each of
    the above.

    Keywords
    --------
    x0, x1: array-like
        These should be numerical iterables.

    paired: boolean, default False

    resamples: int, default 5000
        The number of bootstrap resamples to be taken.

    alpha: float, default 0.05
        Denotes the likelihood that the confidence interval produced
        _does not_ include the true imean difference. When alpha = 0.05,
        a 95% confidence interval is produced.

    random_seed: int, default 12345
        `random_seed` is used to seed the random number generator during
        bootstrap resampling. This ensures that the confidence intervals
        reported are replicable.


    Returns
    -------
    A pandas DataFrame with the various effect sizes as the index, and with the
    following columns:
        effect_size:    The effect size of interest.
        bias:           Bias as computed according to the method of Efron and
                        Tibshirani.
        acceleration:   Accelration as computed according to the method of
                        Efron and Tibshirani.
        index_low:      The index of the sorted bootstrap array corresponding to
                        the lower confidence interval bound.
        index_high:     The index of the sorted bootstrap array corresponding to
                        the upper confidence interval bound.
        bca_ci_low:     The bias corrected and accelerated lower bound of the
                        confidence interval.
        bca_ci_high:    The bias corrected and accelerated upper bound of the
                        confidence interval.
        bootstraps:     The unsorted of resamples used to compute the confidence
                        interval.


    References
    ----------
    Efron, B., & Tibshirani, R. J. (1993). An introduction to the bootstrap.
    New York: Chapman & Hall.

    """
    from numpy import sort as npsort
    from numpy import array, vectorize, isnan
    from pandas import DataFrame, merge
    from . import effsize

    md = effsize.two_group_difference(x0, x1, paired)

    jackknives = compute_meandiff_jackknife(x0, x1, paired)
    acceleration_value = compute_meandiff_acceleration(jackknives, md)
    del jackknives # for memory management

    bootstraps = compute_mean_diff_bootstraps(x0, x1, paired)
    bias_correction = compute_meandiff_bias_correction(bootstraps, md)

    effsizes = DataFrame([md, bias_correction, acceleration_value],
                         index=["effect_size", "bias", "acceleration"]).T

    interval_kwargs = [effsizes.bias, effsizes.acceleration, resamples, alpha]
    l, h = vectorize(compute_interval_limits, otypes=[float, float])(*interval_kwargs)
    effsizes['index_low'], effsizes['index_high'] = l, h

    for e in bootstraps.keys():
        bootstraps[e] = array(bootstraps[e])

    for e in effsizes.index:
        B_sorted = npsort(bootstraps[e])

        for ci_suffix in ['low', 'high']:
            bca_name = 'bca_ci_' + ci_suffix
            idx_name = 'index_' + ci_suffix
            idxx = effsizes.loc[e, idx_name]
            if isnan(idxx):
                effsizes.loc[e, bca_name] = md[e]
            else:
                effsizes.loc[e, bca_name] = B_sorted[int(idxx)]

    boots_df = DataFrame([[v] for v in bootstraps.values()],
                         index=bootstraps.keys(), columns=['bootstraps'])

    del bootstraps # for memory management

    return merge(effsizes, boots_df, left_index=True, right_index=True)
