# Release notes

<!-- do not remove -->

## 2024.03.29

### New Features

- **Forest Plot**: With the new forest plot functionality, DABEST users can now generate forest plots, a mainstay in meta-analysis. This plot type is invaluable for visualizing the results of multiple studies or analyses simultaneously, showing not just the estimated effect sizes but also their confidence intervals. It is built upon the calculation and existing DABEST feature of delta-delta and mini-meta analysis and facilitates systematic reviews by generating a summary plot that compares delta-delta or mini-meta effect size plots from different studies or analysis. It's an essential tool for researchers conducting systematic reviews or wanting to summarize a body of evidence in a single, comprehensive visual.

- **Standardized Delta-delta Effect Size**: We added a new metric akin to a Hedges’ g for delta-delta effect size, which allows comparisons between delta-delta effects generated from metrics with different units. 

- **New Paired Proportion Plot**: This feature builds upon the existing proportional analysis capabilities by introducing advanced aesthetics and clearer visualization of changes in proportions between different groups, inspired by the informative nature of Sankey Diagrams. It's particularly useful for studies that require detailed examination of how proportions shift in paired observations.

- **Customizable Swarm Plot**: Enhancements allow for tailored swarm plot aesthetics, notably the adjustment of swarm sides to produce asymmetric swarm plots. This customization enhances data representation, making visual distinctions more pronounced and interpretations clearer.

### Enhancement

- **Miscellaneous Improvements**: This version also encompasses a broad range of miscellaneous enhancements, including bug fixes, Bootstrapping speed improvements, new templates for raising issues, and updated unit tests. These improvements are designed to streamline the user experience, increase the software's stability, and expand its versatility. By addressing user feedback and identified issues, DABEST continues to refine its functionality and reliability.



## 2023.03.29

### New Features
- **Repeated measures**: Augments the prior function for plotting (independent) multiple test groups versus a shared control; it can now do the same for repeated-measures experimental designs. Thus, together, these two methods can be used to replace both flavors of the 1-way ANOVA with an estimation analysis.

- **Proportional data**: Generates proportional bar plots, proportional differences, and calculates Cohen’s h. Also enables plotting Sankey diagrams for paired binary data. This is the estimation equivalent to a bar chart with Fischer’s exact test.

- **The ∆∆ plot**: Calculates the delta-delta (∆∆) for 2 × 2 experimental designs and plots the four groups with their relevant effect sizes. This design can be used as a replacement for the 2 × 2 ANOVA.

- **Mini-meta**: Calculates and plots a weighted delta (∆) for meta-analysis of experimental replicates. Useful for summarizing data from multiple replicated experiments, for example by different scientists in the same lab, or the same scientist at different times. When the observed values are known (and share a common metric), this makes meta-analysis available as a routinely accessible tool.