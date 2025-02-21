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

We are proud to announce **DABEST Version Dadar (v2025.03.14)** This new
version of the DABEST Python library includes several new features and
performance improvements. It’s a big one!

1.  **Python 3.13 Support**: DABEST now supports Python 3.10-3.13.

2.  **Horizontal Plots**: This new feature allows users to create
    horizontal plots instead of the regular vertical plots, providing a
    more compact visualization of data. This can be utilized by setting
    `horizontal=True` in the `plot()` method. See the [Horizontal
    Plots](../tutorials/08-horizontal_plot.html) tutorial for more
    details.

3.  **Forest Plots**: This new feature allows users to create forest
    plots! Forest plots provide a simple and intuitive way to visualize
    many delta-delta (or Deltas’ g) or mini-meta effect sizes at once
    from multiple different dabest objects without presenting the raw
    data. See the [Forest Plots](../tutorials/07-forest_plot.html)
    tutorial for more details.

4.  **Gridkey**: This new feature allows users to create a gridkey to
    represent the labels of the groups in the plot. This can be utilized
    with the `gridkey_rows` argument in the `plot()` method. See the
    gridkey section in the [Plot
    Aesthetics](../tutorials/09-plot_aesthetics.html) tutorial for more
    details.

5.  **Aesthetic Updates**: We have made several aesthetic improvements
    to the plots, including:

    - **Swarm, Contrast, and Summary bars**: We have added bars to
      better highlight the various groups and their differences. These
      bars can be customized to suit the user’s needs. The swarm and
      contrast bars are provided by default, while the summary bars can
      be added by the user. See the relevant sections in the [Plot
      Aesthetics](../tutorials/09-plot_aesthetics.html) tutorial for
      more details.

    - **Delta-Delta Plots**: We have modified the delta-delta plot
      format to be more compact and easier to read. The new format
      brings the delta-delta (or Deltas’ g) effect size closer to the
      regular effect sizes. In addition, a gap has been added to the
      zeroline to separate the delta-delta and regular effect sizes.

    - **Delta-delta Effect Sizes for Proportion Plots**: Delta-delta
      effect sizes for proportion plots are now available.

    - **Mini-Meta Plots**: We have modified the mini-meta plot format to
      be more compact and easier to read. The new format brings the
      mini-meta effect size closer to the regular effect sizes.

    - **Proportion Plots Sample Sizes**: We have updated the proportion
      plots to show sample sizes for each group. These can be toggled on
      or off via the `prop_sample_counts` parameter.

    - **Effect Size Lines for Paired Plots**: Effect size lines for
      paired plots are now available. These can be toggled on or off via
      the `es_paired_lines` parameter.

    - **Baseline Error Curves**: Plots now include a baseline error dot
      and curve to show the error of the baseline/control group. By
      default, the dot is shown, while the curve can be added by the
      user (via the `show_baseline_ec` parameter).

    - **Delta Text**: There is now an option to display delta text on
      the contrast axes. It displays the effect size of the contrast
      group relative to the reference group. This can be toggled on or
      off via the `delta_text` parameter. It is on by default.

    - **Empty Circle Color Palette**: A new swarmplot color palette
      modification is available for unpaired plots via the
      `empty_circle` parameter in the `plot()` method. This option
      modifies the two-group swarmplots to have empty circles for the
      control group and filled circles for the experimental group.

6.  **Miscellaneous Improvements & Adjustments**

    - **Numba for Speed Improvements**: We have included Numba to speed
      up the various calculations in DABEST. This should make the
      calculations faster and more efficient. Importing DABEST may take
      a little longer than before, and a progress bar will appear during
      the import process to show the calculations being performed. Once
      imported, loading and plotting data should now be faster.

    - **Terminology Updates**: We have made several updates to the
      documentation and terminology to improve clarity and consistency.
      For example:

      - The method to utilise the Deltas’ g effect size is now via the
        `.hedges_g.plot()` method now rather than creating a whole new
        `Delta_g` object as before. The functionality remains the same,
        it plots hedges_g effect sizes and then the Deltas’ g effect
        size alongside these (if a delta-delta experiment was loaded
        correctly).

    - **Updated Tutorial Pages**: We have updated the tutorial pages to
      reflect the new features and changes. The tutorial pages are now
      more comprehensive and hopefully more intuitive!

    - **Results Dataframe for Delta-Delta and Mini-Meta Plots**: A
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

This package is tested on Python 3.10 and onwards. It is highly
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
