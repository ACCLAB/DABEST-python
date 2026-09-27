# Release notes

<!-- do not remove -->

## Unreleased

### New Features
1. **Cluster-aware bootstrap and permutation tests**: `dabest.load()` accepts a new `cluster_col` argument naming the column that identifies the independent sampling unit (cluster) each observation belongs to, such as a participant who contributes several observations or several pairs of paired observations. When it is set, the bootstrap resamples whole clusters with replacement (a cluster bootstrap, stratified by the pattern of groups each cluster appears in) and the permutation test reshuffles labels at the cluster level, so that confidence intervals and permutation p-values account for the correlation between observations from the same cluster. This works for unpaired data, for paired data (`paired` with `id_col`, where `id_col` identifies the pairs and `cluster_col` the units the pairs are nested in), for shared-control and multi-group `idx`, and for delta-delta and mini-meta analyses. The results table gains an `n_clusters` column and `TwoGroupsEffectSize`/`PermutationTest` accept `control_clusters`/`test_clusters` directly. The parametric and rank-based tests in `statistical_tests` are unchanged and still ignore clustering.
2. **Cluster count on plots**: when `cluster_col` is set, each group's axis label reports the number of distinct clusters on a second line below the number of observations, `(N=<observations>,` / ` n=<clusters>)` instead of the usual `(N=<observations>)`, since the cluster count is what the bootstrap actually resamples. To make room, the gap between the raw-data and contrast axes of vertical Cumming plots is then sized to the height of the labels (including the taller labels of two-column Sankey plots), so they clear the contrast axes. Unclustered plots are unchanged.
3. **Small-sample expansion of cluster-bootstrap intervals**: bootstrap intervals are too narrow when there are few independent units, so with `cluster_col` set, confidence intervals are now expanded for the number of clusters by default, using the expanded percentile method of Hesterberg (2015). Each interval is read further into the tails of the same bootstrap distribution, at a level derived from a t distribution with degrees of freedom based on the number of clusters (for nested or mixed designs, whose groups of clusters are resampled independently, combined with the Welch-Satterthwaite approximation). This applies to the percentile and bias-corrected and accelerated intervals, and to those of the baseline error curve, delta-delta and mini-meta analyses. The reported confidence level (`ci`) is unchanged, and the level actually read is reported as `ci_expanded` in the results and in the printed summary, with the unexpanded limits alongside (`bca_low_unexpanded` and so on). Because the plotted bootstrap distribution is unchanged, estimation plots and forest plots draw an expanded interval in two parts: the unexpanded interval as the usual thick bar, and the expansion beyond it as a thinner line, which can be styled with `contrast_expanded_errorbar_kwargs` in `.plot()` (or `expanded_errorbar_kwargs` in `forest_plot()`). In simulations, expanded intervals came within about 3 percentage points of nominal coverage for every effect size once there were at least 6 clusters, and at least 4 in each independently resampled group of clusters; below that, a warning is given. Pass `cluster_ci_expansion=False` to `dabest.load()` to turn the expansion off. `cluster_col` can also no longer be the same column as `x`.

### Documentation
1. **Baseline error curve, explained**: the [Plot Aesthetics tutorial](nbs/tutorials/08-plot_aesthetics.ipynb) and the `show_baseline_ec` docstring now spell out what the baseline error curve (`show_baseline_ec=True`) actually computes, and call out that it is always an *unpaired* self-comparison of the control group, regardless of `paired`. This matters with `cluster_col`: paired real comparisons largely cancel between-cluster variation, but the always-unpaired baseline curve does not, so it can become much wider than the real contrasts once clustering is on. A worked example with and without `cluster_col` is included.
2. **New tutorial: [Cluster-Robust Bootstrap for Repeated Measures](nbs/tutorials/11-cluster_robust_bootstrap.ipynb)**: a self-contained, simulation-based worked example of `cluster_col` for the common case of a participant contributing several sets of paired observations. Its central example is tuned so that, for the identical data and point estimate, the naive dummy-ID bootstrap's 95% interval lies entirely above zero while the cluster-aware interval spans it, making the practical stakes of pseudoreplication concrete rather than abstract. A 200-dataset coverage simulation then shows this is systematic, not a fluke of one dataset: against a known true effect, the naive interval covers the truth only about 82% of the time, the unexpanded cluster-aware interval about 90%, and the default expanded cluster-aware interval about 95%. A section on the small-sample expansion explains when and why it applies.

## v2025.10.20

### New Features
1. **Whorlmap Visualization**: Introducing a new way to visualize effect sizes from multiple comparisons in a grid of whorled square cells. This design condenses information from full bootstrap distributions of an array of contrast objects into a compact visual representation. It optimizes visual real estate by presenting a clear impression of the whole dataset at a glance while retaining nuanced distributional information for further scrutiny. Whorlmaps are a space-efficient alternative to stacked forest plots when working with multi-dimensional DABEST objects from large scale experiments. 
   
2. **Slopegraphs have a new look**: Slopegraphs for paired continuous data now show summary statistics for each group. By default, a thick trend line connects group means, with vertical bars indicating standard deviation. Users can choose the summary type via the `group_summaries` argument in .plot() — options include `'mean_sd'`, `'median_quartiles'`, or `None`. Appearance can be customized using `group_summaries_kwargs`. See the group summaries section in the Plot Aesthetics tutorial for more details.
   
3. **Fixed Mini-meta Weighted Delta Calculation**: The weighted delta in mini-meta plots has been updated to ensure accurate calculation and reporting of the weighted delta.
   
4. **Expanded custom_palette functionality**:
     - **Barplots (unpaired, proportional)**: The custom_palette dict can now take 0 and 1 as keys to color the filled and unfilled portions of the plots. See the custom palette section in the Plot Aesthetics tutorial for more details.
  
     - **Slopegraphs (paired, non proportional)**: The custom_palette can now be used to color the contrast bars and effect-size curves. See the custom palette section in the Plot Aesthetics tutorial for more details.



## v2025.03.27

### New Features

1. **Python 3.13 Support**: DABEST now supports Python 3.10—3.13.

2. **Horizontal Plots**: Users can now create horizontal layout plots, providing compact data visualization. This can be achieved by setting `horizontal=True` in the `.plot()` method.

3. **Forest Plots**: Forest plots provide a simple and intuitive way to visualize many delta-delta (or delta *g*), mini-meta, or regular delta effect sizes at once from multiple different dabest objects without presenting the raw data.

4. **Gridkey**: Users can now represent experimental labels in a ‘gridkey’ table. This can be accessed with the `gridkey` parameter in the `.plot()` method.

5. **Other Visualization Improvements**:
   - **Comparing means and effect sizes**: The estimation plots now include three types of customizable visual features to enhance contextualization and comparison of means and effect sizes:
     - **Bars for the mean of the observed values (`raw_bars`)**: Colored rectangles that extend from the zero line to the mean of each group's raw data. These bars visually highlight the central tendency of the raw data.
     - **Bars for effect size/s (`contrast_bars`)**: Similar to raw bars, these highlight the effect-size difference between two groups (typically test and control) in the contrast axis. They provide a visual representation of the differences between groups.
     - **Summary bands (`reference_band`)**: An optional band or ribbon that can be added to emphasize a specific effect size’s confidence interval that is used as a reference range across the entire contrast axis. Unlike raw and contrast bars, these span horizontally (or vertically if `horizontal=True`) and are not displayed by default.

          Raw and contrast bars are shown by default. Users can customize these bars and add summary bands as needed.

   - **Tighter spacing in delta-delta and mini-meta plots**: We have adjusted the spacing of delta-delta and mini-meta plots to reduce whitespace. The new format brings the overall effect size closer to the two-groups effect sizes. In addition, delta-delta plots now have a gap in the zero line to separate the delta-delta from the ∆ effect sizes.

   - **Delta-delta effect sizes for proportion plots**: In addition to continuous data, delta-delta plots now support binary data (proportions). This means that 2-way designs for binary outcomes can be analyzed with DABEST.

   - **Proportion plots sample sizes**: The sample size of each binary option for each group can now be displayed. These can be toggled on/off via the `prop_sample_counts` parameter.

   - **Effect size lines for paired plots**: Along with lines connecting paired observed values, the paired plots now also display lines linking the effect sizes within a group in the contrast axes. These lines can be toggled on/off via the `contrast_paired_lines` parameter.

   - **Baseline error curves**: To represent the baseline/control group in the contrast axes, it is now possible to plot the baseline dot and the baseline error curve. The dot is shown by default, while the curve can be toggled on/off via the `show_baseline_ec` parameter. This dot helps make it clear where the baseline comes from i.e. the control minus itself. The baseline error curve can be used to show that the baseline itself is an estimate inferred from the observed values of the control data. 

   - **Delta text**: Effect-size deltas (e.g. mean differences) are now displayed as numerals next to their respective effect size. This can be toggled on/off via the `delta_text` parameter.

   - **Empty circle color palette**: A new swarmplot color palette modification is available for unpaired plots via the `empty_circle` parameter in the `.plot()` method. This option modifies the two-group swarmplots to have empty circles for the control group and filled circles for the experimental group.

6. **Miscellaneous Improvements & Adjustments**
    - **Numba for speed improvements**: We have added [Numba](https://numba.pydata.org/) to speed up the various calculations in DABEST. Precalculations will be performed during import, which will help speed up the subsequent loading and plotting of data.
  
    - **Terminology/naming updates**: During the refactoring of the code, we have made several updates to the documentation and terminology to improve clarity and consistency. For example:
      - Plot arguments have been adjusted to bring more clarity and consistency in naming. Arguments relating to the rawdata plot axis will now be typically referred to with `raw` while arguments relating to the contrast axis will be referred to with `contrast`. For example, `raw_label` replaces `swarm_label` and `bar_label`. The various kwargs relating to each different type of plot (e.g., `swarmplot_kwargs`) remain unchanged.
  
      - The method to utilise the Delta *g* effect size is now via the .hedges_g.plot() method rather than creating a whole new Delta_g object as before. The functionality remains the same, it plots hedges_g effect sizes and then the Delta *g* effect size alongside these (if a delta-delta experiment was loaded correctly).

    - **Updated tutorial pages**: We have updated the tutorial pages to reflect the new features and changes. The tutorial pages are now more comprehensive and (hopefully!) more intuitive!

    - **Results dataframe for delta-delta and mini-meta plots**: A results dataframe can now be extracted for both the delta-delta and mini-meta effect size data (similar to the results dataframe for the regular effect sizes). These can be found via the `.results` attribute of the `.delta_delta` or `.mini_meta` object.



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