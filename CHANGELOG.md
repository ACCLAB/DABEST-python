# Release notes

<!-- do not remove -->

## v2025.03.14

### New Features

- **Horizontal Plots**: This new feature allows users to create horizontal plots instead of the regular vertical plots, providing a more compact visualization of data. This can be utilized by setting `horizontal=True` in the `plot()` method.

- **Forest Plots**: This new feature allows users to create forest plots! Forest plots provide a simple and intuitive way to visualize many delta-delta (or Deltas' g) or mini-meta effect sizes at once from multiple different dabest objects without presenting the raw data.

- **Gridkey**: This new feature allows users to create a gridkey to represent the labels of the groups in the plot. This can be utilized with the `gridkey_rows` argument in the `plot()` method.

- **Aesthetic Updates**: We have made several aesthetic improvements to the plots, including:
   - **Swarm, Contrast, and Summary bars**: We have added bars to better highlight the various groups and their differences. These bars can be customized to suit the user's needs. The swarm and contrast bars are provided by default, while the summary bars can be added by the user.
  
   - **Delta-Delta Plots**: We have modified the delta-delta plot format to be more compact and easier to read. The new format brings the delta-delta (or Deltas' g) effect size closer to the regular effect sizes. In addition, a gap has been added to the zeroline to separate the delta-delta and regular effect sizes.

   - **Delta-delta Effect Sizes for Proportion Plots**: Delta-delta effect sizes for proportion plots are now available.
  
   - **Mini-Meta Plots**: We have modified the mini-meta plot format to be more compact and easier to read. The new format brings the mini-meta effect size closer to the regular effect sizes.

   - **Proportion Plots Sample Sizes**: We have updated the proportion plots to show sample sizes for each group. These can be toggled on or off via the `prop_sample_counts` parameter.

   - **Effect Size Lines for Paired Plots**: Effect size lines for paired plots are now available. These can be toggled on or off via the `es_paired_lines` parameter.

   - **Baseline Error Curves**: Plots now include a baseline error dot and curve to show the error of the baseline/control group. By default, the dot is shown, while the curve can be added by the user (via the `show_baseline_ec` parameter).

   - **Delta Text**: There is now an option to display delta text on the contrast axes. It displays the effect size of the contrast group relative to the reference group. This can be toggled on or off via the `delta_text` parameter. It is on by default.

   - **Empty Circle Color Palette**: A new swarmplot color palette modification is available for unpaired plots via the `empty_circle` parameter in the `plot()` method. This option modifies the two-group swarmplots to have empty circles for the control group and filled circles for the experimental group.

### Enhancement

- **Python 3.13 Support**: DABEST now supports Python 3.10-3.13.

- **Numba for Speed Improvements**: We have included Numba to speed up the various calculations in DABEST. This should make the calculations faster and more efficient. Importing DABEST may take a little longer than before, and a progress bar will appear during the import process to show the calculations being performed. Once imported, loading and plotting data should now be faster.

- **Terminology Updates**: We have made several updates to the documentation and terminology to improve clarity and consistency. For example:
    - The method to utilise the Deltas' g effect size is now via the `.hedges_g.plot()` method now rather than creating a whole new `Delta_g` object as before. The functionality remains the same, it plots hedges_g effect sizes and then the Deltas' g effect size alongside these (if a delta-delta experiment was loaded correctly).

- **Updated Tutorial Pages**: We have updated the tutorial pages to reflect the new features and changes. The tutorial pages are now more comprehensive and hopefully more intuitive!.

- **Results Dataframe for Delta-delta and Mini-meta Plots**: A results dataframe can now be extracted for both the delta-delta and mini-meta effect size data (similar to the results dataframe for the regular effect sizes). These can be found via the `.results` attribute of the `.delta_delta` or `.mini_meta` object.



## v2024.03.29

### New Features

- **Standardized Delta-delta Effect Size**: We added a new metric akin to a Hedges’ g for delta-delta effect size, which allows comparisons between delta-delta effects generated from metrics with different units. 

- **New Paired Proportion Plot**: This feature builds upon the existing proportional analysis capabilities by introducing advanced aesthetics and clearer visualization of changes in proportions between different groups, inspired by the informative nature of Sankey Diagrams. It's particularly useful for studies that require detailed examination of how proportions shift in paired observations.

- **Customizable Swarm Plot**: Enhancements allow for tailored swarm plot aesthetics, notably the adjustment of swarm sides to produce asymmetric swarm plots. This customization enhances data representation, making visual distinctions more pronounced and interpretations clearer.

### Enhancement

- **Miscellaneous Improvements**: This version also encompasses a broad range of miscellaneous enhancements, including bug fixes, Bootstrapping speed improvements, new templates for raising issues, and updated unit tests. These improvements are designed to streamline the user experience, increase the software's stability, and expand its versatility. By addressing user feedback and identified issues, DABEST continues to refine its functionality and reliability.



## v2023.03.29

### New Features
- **Repeated measures**: Augments the prior function for plotting (independent) multiple test groups versus a shared control; it can now do the same for repeated-measures experimental designs. Thus, together, these two methods can be used to replace both flavors of the 1-way ANOVA with an estimation analysis.

- **Proportional data**: Generates proportional bar plots, proportional differences, and calculates Cohen’s h. Also enables plotting Sankey diagrams for paired binary data. This is the estimation equivalent to a bar chart with Fischer’s exact test.

- **The ∆∆ plot**: Calculates the delta-delta (∆∆) for 2 × 2 experimental designs and plots the four groups with their relevant effect sizes. This design can be used as a replacement for the 2 × 2 ANOVA.

- **Mini-meta**: Calculates and plots a weighted delta (∆) for meta-analysis of experimental replicates. Useful for summarizing data from multiple replicated experiments, for example by different scientists in the same lab, or the same scientist at different times. When the observed values are known (and share a common metric), this makes meta-analysis available as a routinely accessible tool.