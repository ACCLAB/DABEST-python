#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com


__effect_sizes = ['mean_diff', 'cohens_d', 'hedges_g']



def mean_difference(x1, x2=None, paired=False):
    """
    Computes the unstandardized mean difference, and the standardized Cohen's d
    (for a single sample) or both Cohen's d and Hedges' g for 2 samples.
    See the Wikipedia entry here: https://bit.ly/2LzWokf

    Keywords
    --------
    x1: list, tuple, or ndarray.
        Accepts lists, tuples, or numpy ndarrays of numeric types.

    x2: float or numeric iterable, default None.
        The effect size will be computed as x2 - x1.
        If x2 is a float, the reported mean difference and Cohen's d is the
        difference between the mean of x1, and the value x2. If None, x2 = 0.

    paired: boolean, default False.
        If True, computed the paired Cohen's d.

    Returns
    -------
    results: dict
        mean_diff:      float. This is simply the mean of x1 subtracted from the
                        mean of x2.

        cohens_d:       float. This is the mean of x1 subtracted from the
                        mean of x2, divided by the pooled standard deviation
                        of x1 and x2. The pooled SD is computed as:

                          ------------------------------------------
                         / (n1 - 1) * var(x1) + (n2 - 1) * var (x2)
                        /  ----------------------------------------
                       V                (n1 + n2 - 2)

                       where n1 and n2 are the sizes of x1 and x2 respectively.

        hedges_g:       float or Nan. This is Cohen's d corrected for bias via
                        multiplication with the following correction factor:

                                        gamma(n/2)
                        J(n) = ------------------------------
                               sqrt(n/2) * gamma((n - 1) / 2)

                        where n = (n1 + n2 -2).
                        This will have a value of NaN if `x2` is None.
    """

    from numpy import ndarray, isnan, array, mean, std, nan

    # Create dict
    es_dict = {}

    # Convert to numpy arrays for speed,
    if isinstance(x1, (list, tuple, ndarray)):
        x1 = array(x1)

    if x2 is not None:
        if isinstance(x2, (float, int)):
            is_single_samp = True
            paired = False
        else:
            is_single_samp = False
            x2 = array(x2)
    else:
        is_single_samp = True
        paired = False
        x2 = 0

    if is_single_samp or paired:
        if paired:
            # Check x1 and x2 are same length.
            if len(x1) != len(x2):
                raise ValueError("x1 and x2 are not the same length.")
            delta = x2 - x1
            M = mean(delta)
            STD = std(delta)

        elif is_single_samp:
            M = mean(x1) - x2
            STD = std(x1)

        d = M / STD
        g = nan

    else:
        M = mean(x2) - mean(x1)
        pooled_sd = __pooled_sd(x1, x2)

        d = M / pooled_sd

        # Compute Hedges' g
        correction_factor = __hedges_correction_factor(len(x1), len(x2))
        g = correction_factor * d

    es_dict['mean_diff'] = M
    es_dict['cohens_d'] = d
    es_dict['hedges_g'] = g

    return es_dict



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
        A single numeric float.
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



def __hedges_correction_factor(n1, n2):
    """
    Computes the bias correction factor for Hedges' g.
    Uses the gamma function.

    See https://en.wikipedia.org/wiki/Effect_size#Hedges'_g
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



def __pooled_sd(x1, x2):
    from numpy import mean, var, sqrt

    x1_n = len(x1)
    x2_n = len(x2)

    x1_mean = mean(x1)
    x2_mean = mean(x2)

    x1_var = var(x1)
    x2_var = var(x2)

    psd = sqrt(((x1_n - 1) * x1_var + (x2_n - 1) * x2_var) /
               (x1_n + x2_n - 2)
               )

    return psd
