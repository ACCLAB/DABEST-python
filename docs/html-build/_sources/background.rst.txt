.. _background:

==========
Background
==========

------------------------------
What is estimation statistics?
------------------------------

`Estimation statistics <https://en.wikipedia.org/wiki/Estimation_statistics>`_ is a simple framework that—while avoiding the pitfalls of significance testing—uses familiar statistical concepts: means, mean differences, and error bars.

------------------------------
Why use estimation statistics?
------------------------------
For each of the most routine significance tests, there is an estimation replacement:

* Unpaired Student’s t-test → Two-groups Gardner-Altman comparison plot
* Paired Student’s t-test → Paired comparison
* One-way ANOVA + multiple comparisons → Cumming multiple-groups plot
* Repeated measures ANOVA → Multi-paired comparison
* Ordered groups ANOVA → Shared-control comparison

*image of all five types of plot*

All of these plots enable you to graphically inspect the mean difference and its confidence interval. When there are multiple groups, the side-by-side plotting allows the visual comparison of effect sizes.

.. csv-table:: Benefits of Estimation Plot
  :header: " ", "Bars-and-Stars", "Boxplot & P", "Estimation Plot"
  :widths: 30, 15, 15, 15

  "Avoid false dichotomy", "✘", "✘", "✔"
  "Display all observed values", "✘", "✘", "✔"
  "Focus on intervention effect size", "✘", "✘", "✔"
  "Visualize estimate precision", "✘", "✘", "✔"
  "Show mean difference distribution", "✘", "✘", "✔"

**1. Avoid false dichotomy.** Is there really a great difference between probabilities of 0.04999 and 0.05001? One of the many problems with significance testing is the application of an α-threshold creates the illusion of a dichotomy. In significance testing, the former is ‘significant’ and the latter is ‘non-significant.’

The graphical method of showing this test result with clusters of stars amplifies this false dichotomy; since the average reader is primed to look for *P* < 0.05, presenting *P*values is almost as bad.

Estimation plots present the significance test result innocuously: as the presence or absence of a gap between the mean-difference zero line and the closest confidence interval bound.

**2. Display all observed values.** `Bar charts <https://doi.org/10.1038/nmeth.2837>`_ often show means, error, and significance stars only. `Boxplots <https://www.nature.com/articles/nmeth.2811>`_ generally show just medians, quartiles, maybe a few outliers, and *P* values. For observed values, estimation plots follow best practices by presenting `each and every datapoint <https://doi.org/10.1371/journal.pbio.1002128>`_.

Presenting all observed values means that nothing is hidden: range, normality, skew, kurtosis, outliers, bounds, modality, and sample size are all clearly visible.

**3. Focus on intervention effect size.** Estimation comparison plots include an entirely new axis for the mean difference of two groups (or paired data), and a whole panel for the mean differences of multiple groups. This serves to draw attention to something that deserves it, the answer to the question: “What is the magnitude of the effect of the intervention?”

**4. Visualize estimate precision.** Unlike a significance test result, the narrowness of a confidence interval gives a clear impression of effect size precision. The 95% confidence interval provides the range of the the population mean difference values that are the most plausible.

This 95% plausible interval also serves as an 83% prediction interval for replications (`Cumming and Calin-Jageman 2016 <https://www.amazon.com/Introduction-New-Statistics-Estimation-Science/dp/1138825522/>`_), i.e. predicts future replication effect sizes (assuming no change in protocol) with 83% accuracy.

**5. Show mean difference distribution.** The distribution of mean differences can be estimated using bootstrap resamples of the available data. As an approximation of the `Bayes posterior distribution <https://web.stanford.edu/~hastie/Papers/ESLII.pdf>`_, this curve allows the analyst to weigh plausibility over an effect likelihood size range. Plotting this distribution also discourages dichotomous thinking—engendered by *P* values and hard-edged confidence intervals (`Kruschke and Liddell 2017 <https://www.ncbi.nlm.nih.gov/pubmed/28176294>`_)—by drawing attention to the distribution’s graded nature.

------------------------------------------------------------
Why are these plots named after Gardner, Altman, and Cumming?
------------------------------------------------------------

To our knowledge, mean difference comparison plots were first described by Martin Gardner and Douglas Altman (`Gardner and Altman 1986 <https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/3082422/>`_), while the multiple-comparison design was devised by Geoff Cumming (`Cumming 2012 <https://www.amazon.com/Introduction-New-Statistics-Estimation-Science/dp/1138825522/>`_). We propose calling the two-groups plot Gardner-Altman comparison plots and the multi-group plots Cumming comparison plots.
