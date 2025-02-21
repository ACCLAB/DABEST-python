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

We are proud to announce **DABEST Version Ondeh (v2024.03.29)**. This
new version of the DABEST Python library provides several new features
and includes performance improvements.

1.  **New Paired Proportion Plot**: This feature builds upon the
    existing proportional analysis capabilities by introducing advanced
    aesthetics and clearer visualization of changes in proportions
    between different groups, inspired by the informative nature of
    Sankey Diagrams. It’s particularly useful for studies that require
    detailed examination of how proportions shift in paired
    observations.

2.  **Customizable Swarm Plot**: Enhancements allow for tailored swarm
    plot aesthetics, notably the adjustment of swarm sides to produce
    asymmetric swarm plots. This customization enhances data
    representation, making visual distinctions more pronounced and
    interpretations clearer.

3.  **Standardized Delta-delta Effect Size**: We added a new metric akin
    to a Hedges’ g for delta-delta effect size, which allows comparisons
    between delta-delta effects generated from metrics with different
    units.

4.  **Miscellaneous Improvements**: This version also encompasses a
    broad range of miscellaneous enhancements, including bug fixes,
    Bootstrapping speed improvements, new templates for raising issues,
    and updated unit tests. These improvements are designed to
    streamline the user experience, increase the software’s stability,
    and expand its versatility. By addressing user feedback and
    identified issues, DABEST continues to refine its functionality and
    reliability.

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

This package is tested on Python 3.8 and onwards. It is highly
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
