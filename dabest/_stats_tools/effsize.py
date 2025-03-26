"""A range of functions to compute various effect sizes."""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/API/effsize.ipynb.

# %% ../../nbs/API/effsize.ipynb 4
from __future__ import annotations
import numpy as np
from numba import njit
import warnings
from scipy.special import gamma
from scipy.stats import mannwhitneyu


# %% auto 0
__all__ = ['two_group_difference', 'func_difference', 'cohens_d', 'cohens_h', 'hedges_g', 'cliffs_delta', 'weighted_delta']

# %% ../../nbs/API/effsize.ipynb 5
def two_group_difference(control:list|tuple|np.ndarray, #Accepts lists, tuples, or numpy ndarrays of numeric types.
                         test:list|tuple|np.ndarray, #Accepts lists, tuples, or numpy ndarrays of numeric types.
                         is_paired=None, #If not None, returns the paired Cohen's d
                         effect_size:str="mean_diff" # Any one of the following effect sizes: ["mean_diff", "median_diff", "cohens_d", "hedges_g", "cliffs_delta"]
                        )->float: #  The desired effect size.
    """
    Computes the following metrics for control and test:
    
        - Unstandardized mean difference
        - Standardized mean differences (paired or unpaired)
            * Cohen's d
            * Hedges' g
        - Median difference
        - Cliff's Delta
        - Cohen's h (distance between two proportions)

    See the Wikipedia entry [here](https://bit.ly/2LzWokf)
    
    `effect_size`:
    
        mean_diff:      This is simply the mean of `control` subtracted from
                        the mean of `test`.

        cohens_d:       This is the mean of control subtracted from the
                        mean of test, divided by the pooled standard deviation
                        of control and test. The pooled SD is the square as:


                               (n1 - 1) * var(control) + (n2 - 1) * var(test)
                        sqrt (   -------------------------------------------  )
                                                 (n1 + n2 - 2)

                        where n1 and n2 are the sizes of control and test
                        respectively.

        hedges_g:       This is Cohen's d corrected for bias via multiplication
                         with the following correction factor:

                                        gamma(n/2)
                        J(n) = ------------------------------
                               sqrt(n/2) * gamma((n - 1) / 2)

                        where n = (n1 + n2 - 2).

        median_diff:    This is the median of `control` subtracted from the
                        median of `test`.

    """

    if not isinstance(control, np.ndarray):
        control = np.array(control)
    if not isinstance(test, np.ndarray):
        test    = np.array(test)

    if effect_size == "mean_diff":
        return func_difference(control, test, np.mean, is_paired)

    if effect_size == "median_diff":
        mes1 = "Using median as the statistic in bootstrapping may " + \
                "result in a biased estimate and cause problems with " + \
                "BCa confidence intervals. Consider using a different statistic, such as the mean.\n"
        mes2 = "When plotting, please consider using percetile confidence intervals " + \
                "by specifying `ci_type='pct'`. For detailed information, " + \
                "refer to https://github.com/ACCLAB/DABEST-python/issues/129 \n"
        warnings.warn(message=mes1+mes2, category=UserWarning)
        return func_difference(control, test, np.median, is_paired)

    if effect_size == "cohens_d":
        return cohens_d(control, test, is_paired)

    if effect_size == "cohens_h":
        return cohens_h(control, test)

    if effect_size == "hedges_g":
        return hedges_g(control, test, is_paired)

    if effect_size == "cliffs_delta":
        if is_paired:
            err1 = "`is_paired` is not None; therefore Cliff's delta is not defined."
            raise ValueError(err1)
        
        return cliffs_delta(control, test)


# %% ../../nbs/API/effsize.ipynb 6
def func_difference(control:list|tuple|np.ndarray, # NaNs are automatically discarded.
                    test:list|tuple|np.ndarray, # NaNs are automatically discarded.
                    func, # Summary function to apply.
                    is_paired:str # If not None, computes func(test - control). If None, computes func(test) - func(control).
                   )->float:
    """
    
    Applies func to `control` and `test`, and then returns the difference.
    
    """

    # Convert to numpy arrays for speed.
    # NaNs are automatically dropped.
    if not isinstance(control, np.ndarray):
        control = np.array(control)
    if not isinstance(test, np.ndarray):
        test    = np.array(test)

    if is_paired:
        if len(control) != len(test):
            err = "The two arrays supplied do not have the same length."
            raise ValueError(err)

        non_nan_mask = ~np.isnan(control) & ~np.isnan(test)
        control_non_nan = control[non_nan_mask]
        test_non_nan = test[non_nan_mask]

        return func(test_non_nan - control_non_nan)

    
    control = control[~np.isnan(control)]
    test    = test[~np.isnan(test)]
    return func(test) - func(control)


# %% ../../nbs/API/effsize.ipynb 7
@njit(cache=True)
def cohens_d(control:list|tuple|np.ndarray,
             test:list|tuple|np.ndarray,
             is_paired:str=None # If not None, the paired Cohen's d is returned.
            )->float:
    """
    Computes Cohen's d for test v.s. control.
    See [here](https://en.wikipedia.org/wiki/Effect_size#Cohen's_d)

    If `is_paired` is None, returns:
    
    $$
    \\frac{\\bar{X}_2 - \\bar{X}_1}{s_{pooled}}
    $$
    
    where
    
    $$
    s_{pooled} = \\sqrt{\\frac{(n_1 - 1) s_1^2 + (n_2 - 1) s_2^2}{n_1 + n_2 - 2}}
    $$
    
    If `is_paired` is not None, returns:
    
    $$
    \\frac{\\bar{X}_2 - \\bar{X}_1}{s_{avg}}
    $$
    
    where
    
    $$
    s_{avg} = \\sqrt{\\frac{s_1^2 + s_2^2}{2}}
    $$
    
    `Notes`:

    - The sample variance (and standard deviation) uses N-1 degrees of freedoms.
    This is an application of Bessel's correction, and yields the unbiased
    sample variance.

    `References`:
    
        - https://en.wikipedia.org/wiki/Bessel%27s_correction
        - https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation
    """

    control = control[~np.isnan(control)]
    test    = test[~np.isnan(test)]

    pooled_sd, average_sd = _compute_standardizers(control, test)
    # pooled SD is used for Cohen's d of two independant groups.
    # average SD is used for Cohen's d of two paired groups
    # (aka repeated measures).
    # NOT IMPLEMENTED YET: Correlation adjusted SD is used for Cohen's d of
    # two paired groups but accounting for the correlation between
    # the two groups.

    if is_paired:
        # Check control and test are same length.
        if len(control) != len(test):
            raise ValueError("`control` and `test` are not the same length.")
        # assume the two arrays are ordered already.
        delta = test - control
        M = np.mean(delta)
        divisor = average_sd

    else:
        M = np.mean(test) - np.mean(control)
        divisor = pooled_sd
    
    if divisor == 0:
        raise ValueError("The divisor is zero, indicating no variability in the data.")

    return M / divisor

# %% ../../nbs/API/effsize.ipynb 8
# @njit(cache=True) # It uses np.seterr which is not supported by Numba
def cohens_h(control:list|tuple|np.ndarray, 
             test:list|tuple|np.ndarray
            )->float:
    '''
    Computes Cohen's h for test v.s. control.
    See [here](https://en.wikipedia.org/wiki/Cohen%27s_h) for reference.
    
    `Notes`:
    
    - Assuming the input data type is binary, i.e. a series of 0s and 1s,
    and a dict for mapping the 0s and 1s to the actual labels, e.g.{1: "Smoker", 0: "Non-smoker"}
    '''

    np.seterr(divide='ignore', invalid='ignore')

    # Check whether dataframe contains only 0s and 1s.
    if np.isin(control, [0, 1]).all() == False or np.isin(test, [0, 1]).all() == False:
        raise ValueError("Input data must be binary.")

    # Convert to numpy arrays for speed.
    # NaNs are automatically dropped.
    # Aligned with cohens_d calculation.
    control = control[~np.isnan(control)]
    test = test[~np.isnan(test)]

    prop_control = sum(control)/len(control)
    prop_test = sum(test)/len(test)

    # Arcsine transformation
    phi_control = 2 * np.arcsin(np.sqrt(prop_control))
    phi_test = 2 * np.arcsin(np.sqrt(prop_test))

    return phi_test - phi_control


# %% ../../nbs/API/effsize.ipynb 9
def hedges_g(control:list|tuple|np.ndarray, 
             test:list|tuple|np.ndarray, 
             is_paired:str=None)->float:
    """
    Computes Hedges' g for  for test v.s. control.
    It first computes Cohen's d, then calulates a correction factor based on
    the total degress of freedom using the gamma function.

    See [here](https://en.wikipedia.org/wiki/Effect_size#Hedges'_g)

    """

    # Convert to numpy arrays for speed.
    # NaNs are automatically dropped.
    control = control[~np.isnan(control)]
    test    = test[~np.isnan(test)]

    d = cohens_d(control, test, is_paired)
    len_c = len(control)
    len_t = len(test)
    correction_factor = _compute_hedges_correction_factor(len_c, len_t)
    return correction_factor * d

# %% ../../nbs/API/effsize.ipynb 10
@njit(cache=True)
def _mann_whitney_u(x, y):
    """Numba-optimized Mann-Whitney U calculation"""
    n1, n2 = len(x), len(y)
    combined = np.concatenate((x, y))
    
    # Use numpy broadcasting for comparison
    less_than = (combined.reshape(-1, 1) > combined).sum(axis=1)
    equal_to = (combined.reshape(-1, 1) == combined).sum(axis=1)
    
    # Calculate ranks directly
    ranks = less_than + (equal_to + 1) / 2
    
    R1 = np.sum(ranks[:n1])
    U1 = R1 - (n1 * (n1 + 1)) / 2
    return U1

@njit(cache=True)
def _cliffs_delta_core(control, test):
    """Numba-optimized Cliff's delta calculation"""
    U = _mann_whitney_u(test, control)
    return ((2 * U) / (len(control) * len(test))) - 1

def cliffs_delta(control:list|tuple|np.ndarray, 
                 test:list|tuple|np.ndarray
                )->float:
    """
    Computes Cliff's delta for 2 samples.
    See [here](https://en.wikipedia.org/wiki/Effect_size#Effect_size_for_ordinal_data)
    """
    c = control[~np.isnan(control)]
    t = test[~np.isnan(test)]
    return _cliffs_delta_core(c, t)


# %% ../../nbs/API/effsize.ipynb 11
@njit(cache=True)
def _compute_standardizers(control, test):
    """
    Computes the pooled and average standard deviations for two datasets.

    This function is useful in the context of statistical analysis, particularly
    when calculating standardized mean differences between two groups. It supports
    both unpaired and paired data scenarios.

    Parameters:
    control (array-like): A numeric array representing the control group data.
    test (array-like): A numeric array representing the test group data.

    Returns:
    tuple: A tuple containing two elements:
        - pooled (float): The pooled standard deviation, calculated for unpaired two-group 
                          scenarios. It is computed using the sample variances of the 
                          control and test groups, weighted by their sample sizes.
        - average (float): The average standard deviation, calculated for paired data 
                           scenarios. It is the average of the sample standard deviations 
                           of the control and test groups.

    Note:
    The function assumes that the input arrays are independent samples and calculates
    the sample variances using N-1 degrees of freedom.

    For calculation of correlation; not currently used.

    """
    # from scipy.stats import pearsonr

    control_n = len(control)
    test_n = len(test)

    # ddof parameter is not supported by numba.
    control_var = np.var(control)*control_n/(control_n-1) # use N-1 to compute the variance.
    test_var = np.var(test)*test_n/(test_n-1)

    # For unpaired 2-groups standardized mean difference.
    pooled = np.sqrt(((control_n - 1) * control_var + (test_n - 1) * test_var) /
               (control_n + test_n - 2)
               )

    # For paired standardized mean difference.
    average = np.sqrt((control_var + test_var) / 2)

    return pooled, average 

# %% ../../nbs/API/effsize.ipynb 12
def _compute_hedges_correction_factor(n1, 
                                      n2
                                     )->float:
    """
    Computes the bias correction factor for Hedges' g.

    See [here](https://en.wikipedia.org/wiki/Effect_size#Hedges'_g)

    `References`:
    
    - Larry V. Hedges & Ingram Olkin (1985).
    Statistical Methods for Meta-Analysis. Orlando: Academic Press.
    ISBN 0-12-336380-2.
    """

    df = n1 + n2 - 2
    # gamma function is not supported by numba.
    numer = gamma(df / 2)
    denom0 = gamma((df - 1) / 2)
    denom = np.sqrt(df / 2) * denom0

    if np.isinf(numer) or np.isinf(denom):
        # occurs when df is too large.
        # Apply Hedges and Olkin's approximation.
        df_sum = n1 + n2
        denom = (4 * df_sum) - 9
        out = 1 - (3 / denom)

    else:
        out = numer / denom

    return out

# %% ../../nbs/API/effsize.ipynb 13
@njit(cache=True)
def weighted_delta(difference, group_var):
    '''
    Compute the weighted deltas where the weight is the inverse of the
    pooled group difference.
    '''

    weight = np.true_divide(1, group_var)
    return np.sum(difference*weight)/np.sum(weight)
