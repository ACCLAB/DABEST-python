# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/API/bootstrap.ipynb.

# %% auto 0
__all__ = ['bootstrap', 'jackknife_indexes', 'bca']

# %% ../nbs/API/bootstrap.ipynb 3
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel
from scipy.stats import mannwhitneyu, wilcoxon, norm
import warnings

# %% ../nbs/API/bootstrap.ipynb 4
class bootstrap:
    """
    Computes the summary statistic and a bootstrapped confidence interval.

    Returns
    -------
    An `bootstrap` object reporting the summary statistics, percentile CIs, bias-corrected and accelerated (BCa) CIs, and the settings used:
        `summary`: float.
            The summary statistic.
        `is_difference`: boolean.
             Whether or not the summary is the difference between two groups. If False, only x1 was supplied.
        `is_paired`:  string, default None
            The type of the experiment under which the data are obtained
        `statistic`: callable
            The function used to compute the summary.
        `reps`: int
            The number of bootstrap iterations performed.
        `stat_array`:array
            A sorted array of values obtained by bootstrapping the input arrays.
        `ci`:float
            The size of the confidence interval reported (in percentage).
        `pct_ci_low,pct_ci_high`:floats
            The upper and lower bounds of the confidence interval as computed by taking the percentage bounds.
        `pct_low_high_indices`:array
            An array with the indices in `stat_array` corresponding to the percentage confidence interval bounds.
        `bca_ci_low, bca_ci_high`: floats
            The upper and lower bounds of the bias-corrected and accelerated(BCa) confidence interval. See Efron 1977.
        `bca_low_high_indices`: array
            An array with the indices in `stat_array` corresponding to the BCa confidence interval bounds.
        `pvalue_1samp_ttest`: float
            P-value obtained from scipy.stats.ttest_1samp. If 2 arrays were passed (x1 and x2), returns 'NIL'. See <https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.stats.ttest_1samp.html>
        `pvalue_2samp_ind_ttest`: float
            P-value obtained from scipy.stats.ttest_ind. If a single array was given (x1 only), or if `paired` is not None, returns 'NIL'. See <https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.stats.ttest_ind.html>
        `pvalue_2samp_related_ttest`: float
            P-value obtained from scipy.stats.ttest_rel. If a single array was given (x1 only), or if `paired` is None, returns 'NIL'. See <https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.stats.ttest_rel.html>
        `pvalue_wilcoxon`: float
            P-value obtained from scipy.stats.wilcoxon. If a single array was given (x1 only), or if `paired` is None, returns 'NIL'. The Wilcoxons signed-rank test is a nonparametric paired test of the null hypothesis that the related samples x1 and x2 are from the same distribution. See <https://docs.scipy.org/doc/scipy-1.0.0/reference/scipy.stats.wilcoxon.html>
        `pvalue_mann_whitney`: float
            Two-sided p-value obtained from scipy.stats.mannwhitneyu. If a single array was given (x1 only), returns 'NIL'. The Mann-Whitney U-test is a nonparametric unpaired test of the null hypothesis that x1 and x2 are from the same distribution. See <https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.stats.mannwhitneyu.html>

    """

    def __init__(
        self,
        x1: np.array,  # The data in a one-dimensional array form. Only x1 is required. If x2 is given, the bootstrapped summary difference between the two groups (x2-x1) is computed. NaNs are automatically discarded.
        x2: np.array = None,  # The data in a one-dimensional array form. Only x1 is required. If x2 is given, the bootstrapped summary difference between the two groups (x2-x1) is computed. NaNs are automatically discarded.
        paired: bool = False,  # Whether or not x1 and x2 are paired samples. If 'paired' is None then the data will not be treated as paired data in the subsequent calculations. If 'paired' is 'baseline', then in each tuple of x, other groups will be paired up with the first group (as control). If 'paired' is 'sequential', then in each tuple of x, each group will be paired up with the previous group (as control).
        stat_function: callable = np.mean,  # The summary statistic called on data.
        smoothboot: bool = False,  # Taken from seaborn.algorithms.bootstrap. If True, performs a smoothed bootstrap (draws samples from a kernel destiny estimate).
        alpha_level: float = 0.05,  # Denotes the likelihood that the confidence interval produced does not include the true summary statistic. When alpha = 0.05, a 95% confidence interval is produced.
        reps: int = 5000,  # Number of bootstrap iterations to perform.
    ):
        # Turn to pandas series.
        # x1 = pd.Series(x1).dropna()
        x1 = x1[~np.isnan(x1)]

        diff = False

        # Initialise stat_function
        if stat_function is None:
            stat_function = np.mean

        # Compute two-sided alphas.
        if alpha_level > 1.0 or alpha_level < 0.0:
            raise ValueError("alpha_level must be between 0 and 1.")
        alphas = np.array([alpha_level / 2.0, 1 - alpha_level / 2.0])

        sns_bootstrap_kwargs = {
            "func": stat_function,
            "n_boot": reps,
            "smooth": smoothboot,
        }

        if paired:
            # check x2 is not None:
            if x2 is None:
                raise ValueError("Please specify x2.")
            
            # x2 = pd.Series(x2).dropna()
            x2 = x1[~np.isnan(x2)]

            if len(x1) != len(x2):
                raise ValueError("x1 and x2 are not the same length.")

        if (x2 is None) or (paired is not None):
            if x2 is None:
                tx = x1
                paired = False
                ttest_single = ttest_1samp(x1, 0)[1]
                ttest_2_ind = "NIL"
                ttest_2_paired = "NIL"
                wilcoxonresult = "NIL"

            else:  # only two options to enter here
                diff = True
                tx = x2 - x1
                ttest_single = "NIL"
                ttest_2_ind = "NIL"
                ttest_2_paired = ttest_rel(x1, x2)[1]

                try:
                    wilcoxonresult = wilcoxon(x1, x2)[1]
                except ValueError as e:
                    warnings.warn("Wilcoxon test could not be performed. This might be due "
                    "to no variability in the difference of the paired groups. \n"
                    "Error: {}\n"
                    "For detailed information, please refer to https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html "
                    .format(e))
            mannwhitneyresult = "NIL"

            # Turns data into array, then tuple.
            tdata = (tx,)

            # The value of the statistic function applied
            # just to the actual data.
            summ_stat = stat_function(*tdata)
            statarray = sns.algorithms.bootstrap(tx, **sns_bootstrap_kwargs)
            statarray.sort()

            # Get Percentile indices
            pct_low_high = np.round((reps - 1) * alphas)
            pct_low_high = np.nan_to_num(pct_low_high).astype("int")

        elif x2 is not None and paired is None:
            diff = True
            # x2 = pd.Series(x2).dropna()
            x2 = x2[~np.isnan(x2)]
            # Generate statarrays for both arrays.
            ref_statarray = sns.algorithms.bootstrap(x1, **sns_bootstrap_kwargs)
            exp_statarray = sns.algorithms.bootstrap(x2, **sns_bootstrap_kwargs)

            tdata = exp_statarray - ref_statarray
            statarray = tdata.copy()
            statarray.sort()
            tdata = (tdata,)  # Note tuple form.

            # The difference as one would calculate it.
            summ_stat = stat_function(x2) - stat_function(x1)

            # Get Percentile indices
            pct_low_high = np.round((reps - 1) * alphas)
            pct_low_high = np.nan_to_num(pct_low_high).astype("int")

            # Statistical tests.
            ttest_single = "NIL"
            ttest_2_ind = ttest_ind(x1, x2)[1]
            ttest_2_paired = "NIL"
            mannwhitneyresult = mannwhitneyu(x1, x2, alternative="two-sided")[1]
            wilcoxonresult = "NIL"

        # Get Bias-Corrected Accelerated indices convenience function invoked.
        bca_low_high = bca(tdata, alphas, statarray, stat_function, summ_stat, reps)

        # Warnings for unstable or extreme indices.
        for ind in [pct_low_high, bca_low_high]:
            if np.any(ind == 0) or np.any(ind == reps - 1):
                warnings.warn(
                    "Some values used extremal samples;"
                    " results are probably unstable."
                )
            elif np.any(ind < 10) or np.any(ind >= reps - 10):
                warnings.warn(
                    "Some values used top 10 low/high samples;"
                    " results may be unstable."
                )

        self.summary = summ_stat
        self.is_paired = paired
        self.is_difference = diff
        self.statistic = str(stat_function)
        self.n_reps = reps

        self.ci = (1 - alpha_level) * 100
        self.stat_array = np.array(statarray)

        self.pct_ci_low = statarray[pct_low_high[0]]
        self.pct_ci_high = statarray[pct_low_high[1]]
        self.pct_low_high_indices = pct_low_high

        self.bca_ci_low = statarray[bca_low_high[0]]
        self.bca_ci_high = statarray[bca_low_high[1]]
        self.bca_low_high_indices = bca_low_high

        self.pvalue_1samp_ttest = ttest_single
        self.pvalue_2samp_ind_ttest = ttest_2_ind
        self.pvalue_2samp_paired_ttest = ttest_2_paired
        self.pvalue_wilcoxon = wilcoxonresult
        self.pvalue_mann_whitney = mannwhitneyresult

        self.results = {
            "stat_summary": self.summary,
            "is_difference": diff,
            "is_paired": paired,
            "bca_ci_low": self.bca_ci_low,
            "bca_ci_high": self.bca_ci_high,
            "ci": self.ci,
        }

    def __repr__(self):
        if "mean" in self.statistic:
            stat = "mean"
        elif "median" in self.statistic:
            stat = "median"
        else:
            stat = self.statistic

        diff_types = {"sequential": "paired", "baseline": "paired", None: "unpaired"}
        if self.is_difference:
            a = "The {} {} difference is {}.".format(
                diff_types[self.is_paired], stat, self.summary
            )
        else:
            a = "The {} is {}.".format(stat, self.summary)

        b = "[{} CI: {}, {}]".format(self.ci, self.bca_ci_low, self.bca_ci_high)
        return "\n".join([a, b])

# %% ../nbs/API/bootstrap.ipynb 5
def jackknife_indexes(data):
    # Taken without modification from scikits.bootstrap package.
    """
    From the scikits.bootstrap package.
    Given an array, returns a list of arrays where each array is a set of
    jackknife indexes.

    For a given set of data Y, the jackknife sample J[i] is defined as the
    data set Y with the ith data point deleted.
    """

    base = np.arange(0, len(data))
    return (np.delete(base, i) for i in base)


def bca(data, alphas, stat_array, stat_function, ostat, reps):
    """
    Subroutine called to calculate the BCa statistics.
    Borrowed heavily from scikits.bootstrap code.
    """

    # The bias correction value.
    z0 = norm.ppf((1.0 * np.sum(stat_array < ostat, axis=0)) / reps)

    # Statistics of the jackknife distribution
    jack_indexes = jackknife_indexes(data[0])
    jstat = [stat_function(*(x[indexes] for x in data)) for indexes in jack_indexes]
    jmean = np.mean(jstat, axis=0)

    # Acceleration value
    a = np.divide(
        np.sum((jmean - jstat) ** 3, axis=0),
        (6.0 * np.sum((jmean - jstat) ** 2, axis=0) ** 1.5),
    )
    if np.any(np.isnan(a)):
        nanind = np.nonzero(np.isnan(a))
        warnings.warn(
            "Some acceleration values were undefined."
            "This is almost certainly because all values"
            "for the statistic were equal. Affected"
            "confidence intervals will have zero width and"
            "may be inaccurate (indexes: {})".format(nanind)
        )
    zs = z0 + norm.ppf(alphas).reshape(alphas.shape + (1,) * z0.ndim)
    avals = norm.cdf(z0 + zs / (1 - a * zs))
    nvals = np.round((reps - 1) * avals)
    nvals = np.nan_to_num(nvals).astype("int")

    return nvals
