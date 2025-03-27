# DABEST-Python


<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

[![minimal Python
version](https://img.shields.io/badge/Python%3E%3D-3.10-6666ff.svg)](https://www.anaconda.com/distribution/)
[![PyPI
version](https://badge.fury.io/py/dabest.svg)](https://badge.fury.io/py/dabest)
[![Downloads](https://img.shields.io/pepy/dt/dabest.svg)](https://pepy.tech/project/dabest)
[![Free-to-view
citation](https://zenodo.org/badge/DOI/10.1038/s41592-019-0470-3.svg)](https://rdcu.be/bHhJ4)
[![License](https://img.shields.io/badge/License-BSD%203--Clause--Clear-orange.svg)](https://spdx.org/licenses/BSD-3-Clause-Clear.html)

## Recent Version Update

We are proud to announce **DABEST Version Dadar (v2025.03.27)** This new
version of the DABEST Python library includes several new features and
performance improvements. It’s a big one!

1.  **Python 3.13 Support**: DABEST now supports Python 3.10—3.13.

2.  **Horizontal Plots**: Users can now create horizontal layout plots,
    providing compact data visualization. This can be achieved by
    setting `horizontal=True` in the `.plot()` method. See the
    [Horizontal Plots
    tutorial](../nbs/tutorials/08-horizontal_plot.ipynb) for more
    details.

3.  **Forest Plots**: Forest plots provide a simple and intuitive way to
    visualize many delta-delta (or delta *g*), mini-meta, or regular
    delta effect sizes at once from multiple different dabest objects
    without presenting the raw data. See the [Forest Plots
    tutorial](../nbs/tutorials/07-forest_plot.ipynb) for more details.

4.  **Gridkey**: Users can now represent experimental labels in a
    ‘gridkey’ table. This can be accessed with the `gridkey` parameter
    in the `.plot()` method. See the gridkey section in the [Plot
    Aesthetics tutorial](../nbs/tutorials/09-plot_aesthetics.ipynb) for
    more details.

5.  **Other Visualization Improvements**:

    - **Comparing means and effect sizes**: The estimation plots now
      include three types of customizable visual features to enhance
      contextualization and comparison of means and effect sizes:

      - **Bars for the mean of the observed values (`raw_bars`)**:
        Colored rectangles that extend from the zero line to the mean of
        each group’s raw data. These bars visually highlight the central
        tendency of the raw data.

      - **Bars for effect size/s (`contrast_bars`)**: Similar to raw
        bars, these highlight the effect-size difference between two
        groups (typically test and control) in the contrast axis. They
        provide a visual representation of the differences between
        groups.

      - **Summary bands (`reference_band`)**: An optional band or ribbon
        that can be added to emphasize a specific effect size’s
        confidence interval that is used as a reference range across the
        entire contrast axis. Unlike raw and contrast bars, these span
        horizontally (or vertically if `horizontal=True`) and are not
        displayed by default.

        Raw and contrast bars are shown by default. Users can customize
        these bars and add summary bands as needed. For detailed
        customization instructions, please refer to the [Plot Aesthetics
        tutorial](../nbs/tutorials/09-plot_aesthetics.ipynb).

    - **Tighter spacing in delta-delta and mini-meta plots**: We have
      adjusted the spacing of delta-delta and mini-meta plots to reduce
      whitespace. The new format brings the overall effect size closer
      to the two-groups effect sizes. In addition, delta-delta plots now
      have a gap in the zero line to separate the delta-delta from the ∆
      effect sizes.

    - **Delta-delta effect sizes for proportion plots**: In addition to
      continuous data, delta-delta plots now support binary data
      (proportions). This means that 2-way designs for binary outcomes
      can be analyzed with DABEST.

    - **Proportion plots sample sizes**: The sample size of each binary
      option for each group can now be displayed. These can be toggled
      on/off via the `prop_sample_counts` parameter.

    - **Effect size lines for paired plots**: Along with lines
      connecting paired observed values, the paired plots now also
      display lines linking the effect sizes within a group in the
      contrast axes. These lines can be toggled on/off via the
      `contrast_paired_lines` parameter.

    - **Baseline error curves**: To represent the baseline/control group
      in the contrast axes, it is now possible to plot the baseline dot
      and the baseline error curve. The dot is shown by default, while
      the curve can be toggled on/off via the `show_baseline_ec`
      parameter. This dot helps make it clear where the baseline comes
      from i.e. the control minus itself. The baseline error curve can
      be used to show that the baseline itself is an estimate inferred
      from the observed values of the control data.

    - **Delta text**: Effect-size deltas (e.g. mean differences) are now
      displayed as numerals next to their respective effect size. This
      can be toggled on/off via the `delta_text` parameter.

    - **Empty circle color palette**: A new swarmplot color palette
      modification is available for unpaired plots via the
      `empty_circle` parameter in the `.plot()` method. This option
      modifies the two-group swarmplots to have empty circles for the
      control group and filled circles for the experimental group.

6.  **Miscellaneous Improvements & Adjustments**

    - **Numba for speed improvements**: We have added
      [Numba](https://numba.pydata.org/) to speed up the various
      calculations in DABEST. Precalculations will be performed during
      import, which will help speed up the subsequent loading and
      plotting of data.

    - **Terminology/naming updates**: During the refactoring of the
      code, we have made several updates to the documentation and
      terminology to improve clarity and consistency. For example:

      - Plot arguments have been adjusted to bring more clarity and
        consistency in naming. Arguments relating to the rawdata plot
        axis will now be typically referred to with `raw` while
        arguments relating to the contrast axis will be referred to with
        `contrast`. For example, `raw_label` replaces `swarm_label` and
        `bar_label`. The various kwargs relating to each different type
        of plot (e.g., `swarmplot_kwargs`) remain unchanged.

      - The method to utilise the Delta *g* effect size is now via the
        .hedges_g.plot() method rather than creating a whole new Delta_g
        object as before. The functionality remains the same, it plots
        hedges_g effect sizes and then the Delta *g* effect size
        alongside these (if a delta-delta experiment was loaded
        correctly).

    - **Updated tutorial pages**: We have updated the tutorial pages to
      reflect the new features and changes. The tutorial pages are now
      more comprehensive and (hopefully!) more intuitive!

    - **Results dataframe for delta-delta and mini-meta plots**: A
      results dataframe can now be extracted for both the delta-delta
      and mini-meta effect size data (similar to the results dataframe
      for the regular effect sizes). These can be found via the
      `.results` attribute of the `.delta_delta` or `.mini_meta` object.

## Contents

<!-- TOC depthFrom:1 depthTo:2 withLinks:1 updateOnSave:1 orderedList:0 -->

- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [How to cite](#how-to-cite)
- [Bugs](#bugs)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [Testing](#testing)
- [DABEST in other languages](#dabest-in-other-languages)

<!-- /TOC -->

## About

DABEST is a package for **D**ata **A**nalysis using
**B**ootstrap-Coupled **EST**imation.

[Estimation
statistics](https://en.wikipedia.org/wiki/Estimation_statistics) are a
[simple framework](https://thenewstatistics.com/itns/) that avoids the
[pitfalls](https://www.nature.com/articles/nmeth.3288) of significance
testing. It employs familiar statistical concepts such as means, mean
differences, and error bars. More importantly, it focuses on the effect
size of one’s experiment or intervention, rather than succumbing to a
false dichotomy engendered by *P* values.

An estimation plot comprises two key features.

1.  It presents all data points as a swarm plot, ordering each point to
    display the underlying distribution.

2.  It illustrates the effect size as a **bootstrap 95% confidence
    interval** on a **separate but aligned axis**.

![The five kinds of estimation
plots](showpiece.png "The five kinds of estimation plots.")

DABEST powers [estimationstats.com](https://www.estimationstats.com/),
allowing everyone access to high-quality estimation plots.

## Installation

This package is tested on Python 3.11 and onwards. It is highly
recommended to download the [Anaconda
distribution](https://www.continuum.io/downloads) of Python in order to
obtain the dependencies easily.

You can install this package via `pip`.

To install, at the command line run

``` shell
pip install dabest
```

You can also
[clone](https://help.github.com/articles/cloning-a-repository) this repo
locally.

Then, navigate to the cloned repo in the command line and run

``` shell
pip install .
```

## Usage

``` python3
import pandas as pd
import dabest

# Load the iris dataset. This step requires internet access.
iris = pd.read_csv("https://github.com/mwaskom/seaborn-data/raw/master/iris.csv")

# Load the above data into `dabest`.
iris_dabest = dabest.load(data=iris, x="species", y="petal_width",
                          idx=("setosa", "versicolor", "virginica"))

# Produce a Cumming estimation plot.
iris_dabest.mean_diff.plot();
```

![A Cumming estimation plot of petal width from the iris
dataset](iris.png)

Please refer to the official
[tutorial](https://acclab.github.io/DABEST-python/) for more useful code
snippets.

## How to cite

**Moving beyond P values: Everyday data analysis with estimation plots**

*Joses Ho, Tayfun Tumkaya, Sameer Aryal, Hyungwon Choi, Adam
Claridge-Chang*

Nature Methods 2019, 1548-7105.
[10.1038/s41592-019-0470-3](http://dx.doi.org/10.1038/s41592-019-0470-3)

[Paywalled publisher
site](https://www.nature.com/articles/s41592-019-0470-3); [Free-to-view
PDF](https://rdcu.be/bHhJ4)

## Bugs

Please report any bugs on the [issue
page](https://github.com/ACCLAB/DABEST-python/issues/new).

## Contributing

All contributions are welcome; please read the [Guidelines for
contributing](../CONTRIBUTING.md) first.

We also have a [Code of Conduct](../CODE_OF_CONDUCT.md) to foster an
inclusive and productive space.

### A wish list for new features

If you have any specific comments and ideas for new features that you
would like to share with us, please read the [Guidelines for
contributing](../CONTRIBUTING.md), create a new issue using Feature
request template or create a new post in [our Google
Group](https://groups.google.com/g/estimationstats).

## Acknowledgements

We would like to thank alpha testers from the [Claridge-Chang
lab](https://www.claridgechang.net/): [Sangyu
Xu](https://github.com/sangyu), [Xianyuan
Zhang](https://github.com/XYZfar), [Farhan
Mohammad](https://github.com/farhan8igib), Jurga Mituzaitė, and
Stanislav Ott.

## Testing

To test DABEST, you need to install
[pytest](https://docs.pytest.org/en/latest) and
[nbdev](https://nbdev.fast.ai/).

- Run `pytest` in the root directory of the source distribution. This
  runs the test suite in the folder `dabest/tests/mpl_image_tests`.
- Run `nbdev_test` in the root directory of the source distribution.
  This runs the value assertion tests in the folder `dabest/tests`

The test suite ensures that the bootstrapping functions and the plotting
functions perform as expected.

For detailed information, please refer to the [test
folder](../nbs/tests/README.md)

## DABEST in other languages

DABEST is also available in R
([dabestr](https://github.com/ACCLAB/dabestr)) and Matlab
([DABEST-Matlab](https://github.com/ACCLAB/DABEST-Matlab)).
