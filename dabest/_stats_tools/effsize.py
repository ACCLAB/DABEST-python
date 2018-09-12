#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com
"""
A range of functions to compute various effect sizes.

    two_group_difference
    cohens_d
    hedges_g
    cliffs_delta
    func_difference
"""


def two_group_difference(control, test, paired=False):
    """
    Computes the following metrics for control and test:
        - Unstandardized mean difference
        - Standardized mean differences (paired or unpaired)
            * Cohen's d
            * Hedges' g
        - Median difference
        - Cliff's Delta

    See the Wikipedia entry here: https://bit.ly/2LzWokf

    Keywords
    --------
    control, test: list, tuple, or ndarray.
        Accepts lists, tuples, or numpy ndarrays of numeric types.

    paired: boolean, default False.
        If True, returns the paired Cohen's d.

    Returns
    -------
    results: dictionary with the following keys and values.

        mean_diff:      This is simply the mean of `control` subtracted from
                        the mean of `test`.

        cohens_d:       This is the mean of control subtracted from the
                        mean of test, divided by the pooled standard deviation
                        of control and test. The pooled SD is computed as:

                          ------------------------------------------
                         / (n1 - 1) * var(control) + (n2 - 1) * var (test)
                        /  ----------------------------------------
                       V                (n1 + n2 - 2)

                        where n1 and n2 are the sizes of control and test
                        respectively.

        hedges_g:       This is Cohen's d corrected for bias via multiplication
                         with the following correction factor:

                                        gamma(n/2)
                        J(n) = ------------------------------
                               sqrt(n/2) * gamma((n - 1) / 2)

                        where n = (n1 + n2 -2).

        median_diff:    This is the median of `control` subtracted from the
                        median of `test`.
    """

    from numpy import array, mean, median

    # Create dict for output.
    es_dict = {}

    # es_dict["is_paired"] = paired

    es_dict['mean_diff'] = func_difference(control, test, mean)
    es_dict['median_diff'] = func_difference(control, test, median)
    es_dict['cohens_d'] = cohens_d(control, test, paired)
    es_dict['hedges_g'] = hedges_g(control, test, paired)

    if paired is False:
        es_dict['cliffs_delta'] = cliffs_delta(control, test)

    return es_dict



def _compute_standardizers(control, test):
    from numpy import mean, var, sqrt, nan
    # For calculation of correlation; not currently used.
    # from scipy.stats import pearsonr

    control_n = len(control)
    test_n = len(test)

    control_mean = mean(control)
    test_mean = mean(test)

    control_var = var(control, ddof=1) # use N-1 to compute the variance.
    test_var = var(test, ddof=1)

    control_std = sqrt(control_var)
    test_std = sqrt(test_var)

    # For unpaired 2-groups standardized mean difference.
    pooled = sqrt(((control_n - 1) * control_var + (test_n - 1) * test_var) /
               (control_n + test_n - 2)
               )

    # For paired standardized mean difference.
    average = sqrt((control_var + test_var) / 2)

    # if len(control) == len(test):
    #     corr = pearsonr(control, test)[0]
    #     std_diff = sqrt(control_var + test_var - (2 * corr * control_std * test_std))
    #     std_diff_corrected = std_diff / (sqrt(2 * (1 - corr)))
    #     return pooled, average, std_diff_corrected
    #
    # else:
    return pooled, average # indent if you implement above code chunk.



def cohens_d(control, test, paired=False):
    """
    Computes Cohen's d for test v.s. control.
    See https://en.wikipedia.org/wiki/Effect_size#Cohen's_d

    Keywords
    --------
    control, test: List, tuple, or array.

    paired: boolean, default False
        If True, the paired Cohen's d is returned.

    Returns
    -------
        d: float.
            If paired is False, this is equivalent to:
            (numpy.mean(test) - numpy.mean(control))  / pooled StDev

            If paired is True, returns
            (numpy.mean(test) - numpy.mean(control))  / average StDev

            The pooled standard deviation is equal to:

              ------------------------------------------------
             / (n1 - 1) * var(control) + (n2 - 1) * var (test)
            /  ----------------------------------------
           V                (n1 + n2 - 2)


            The average standard deviation is equal to:

              ---------------------------
             / var(control) + var(test)
            / -------------------------
           V              2

    Notes
    -----
    The sample variance (and standard deviation) uses N-1 degrees of freedoms.
    This is an application of Bessel's correction, and yields the unbiased
    sample variance.

    References:
        https://en.wikipedia.org/wiki/Bessel%27s_correction
        https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation
    """
    from numpy import array, mean

    # Convert to numpy arrays for speed
    control = array(control)
    test = array(test)

    pooled_sd, average_sd = _compute_standardizers(control, test)
    # pooled SD is used for Cohen's d of two independant groups.
    # average SD is used for Cohen's d of two paired groups
    # (aka repeated measures).
    # NOT IMPLEMENTED YET: Correlation adjusted SD is used for Cohen's d of
    # two paired groups but accounting for the correlation between
    # the two groups.

    if paired:
        # Check control and test are same length.
        if len(control) != len(test):
            raise ValueError("`control` and `test` are not the same length.")
        # assume the two arrays are ordered already.
        delta = test - control
        M = mean(delta)
        return M / average_sd

    else:
        M = mean(test) - mean(control)
        return M / pooled_sd



def _compute_hedges_correction_factor(n1, n2):
    """
    Computes the bias correction factor for Hedges' g.

    See https://en.wikipedia.org/wiki/Effect_size#Hedges'_g

    Returns
    -------
        j: float

    References
    ----------
    Larry V. Hedges & Ingram Olkin (1985).
    Statistical Methods for Meta-Analysis. Orlando: Academic Press.
    ISBN 0-12-336380-2.
    """

    from scipy.special import gamma
    from numpy import sqrt, isinf
    import warnings

    df = n1 + n2 - 2
    numer = gamma(df / 2)
    denom0 = gamma((df - 1) / 2)
    denom = sqrt(df / 2) * denom0

    if isinf(numer) or isinf(denom):
        # occurs when df is too large.
        # Apply Hedges and Olkin's approximation.
        df_sum = n1 + n2
        denom = (4 * df_sum) - 9
        out = 1 - (3 / denom)

    else:
        out = numer / denom

    return out



def hedges_g(control, test, paired=False):
    """
    Computes Hedges' g for  for test v.s. control.
    It first computes Cohen's d, then calulates a correction factor based on
    the total degress of freedom using the gamma function.

    See https://en.wikipedia.org/wiki/Effect_size#Hedges'_g

    Keywords
    --------
    control, test: numeric iterables.
        These can be lists, tuples, or arrays of numeric types.

    Returns
    -------
        g: float.
    """
    from numpy import array

    control = array(control)
    test = array(test)

    d = cohens_d(control, test, paired)
    len_c = len(control)
    len_t = len(test)
    correction_factor = _compute_hedges_correction_factor(len_c, len_t)
    return correction_factor * d



def cliffs_delta(control, test):
    """
    Computes Cliff's delta for 2 samples.
    See https://en.wikipedia.org/wiki/Effect_size#Effect_size_for_ordinal_data

    Keywords
    --------
    control, test: numeric iterables.
        These can be lists, tuples, or arrays of numeric types.

    Returns
    -------
        delta: float.
    """
    from numpy import array, isnan

    control = array(control)
    test = array(test)

    control = control[~isnan(control)]
    test = test[~isnan(test)]

    control_n = len(control)
    test_n = len(test)

    more = 0
    less = 0

    for i, c in enumerate(control):
        for j, t in enumerate(test):
            if t > c:
                more += 1
            elif t < c:
                less += 1

    cliffs_delta = (more - less) / (control_n * test_n)

    return cliffs_delta



def func_difference(control, test, func):
    """
    Applies func to `control` and `test`, and then returns the difference.

    Keywords:
    --------
        control, test: List, tuple, or array.

    Returns:
    --------
        paired = False:
            func(test) - func(control)
        paired = True:
            func(test - control)
    """
    from numpy import ndarray, array

    # Convert to numpy arrays for speed
    control = array(control)
    test = array(test)
    return func(test) - func(control)
