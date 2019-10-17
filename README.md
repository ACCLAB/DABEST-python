# DABEST-Python
[![Travis CI build status](https://travis-ci.org/ACCLAB/DABEST-python.svg?branch=master)](https://travis-ci.org/ACCLAB/DABEST-python)
[![minimal Python version](https://img.shields.io/badge/Python%3E%3D-3.5-6666ff.svg)](https://www.anaconda.com/distribution/)
[![PyPI version](https://badge.fury.io/py/dabest.svg)](https://badge.fury.io/py/dabest)
[![Downloads](https://pepy.tech/badge/dabest/month)](https://pepy.tech/project/dabest/month)
[![Free-to-view citation](https://zenodo.org/badge/DOI/10.1038/s41592-019-0470-3.svg)](https://rdcu.be/bHhJ4)
[![License](https://img.shields.io/badge/License-BSD%203--Clause--Clear-orange.svg)](https://spdx.org/licenses/BSD-3-Clause-Clear.html)

## About

DABEST is a package for **D**ata **A**nalysis using **B**ootstrap-Coupled **EST**imation.

[Estimation statistics](https://en.wikipedia.org/wiki/Estimation_statistics) is a [simple framework](https://thenewstatistics.com/itns/) that avoids the [pitfalls](https://www.nature.com/articles/nmeth.3288) of significance testing. It uses familiar statistical concepts: means, mean differences, and error bars. More importantly, it focuses on the effect size of one's experiment/intervention, as opposed to a false dichotomy engendered by *P* values.

An estimation plot has two key features.

1. It presents all datapoints as a swarmplot, which orders each point to display the underlying distribution.

2. It presents the effect size as a **bootstrap 95% confidence interval** on a **separate but aligned axes**.

![The five kinds of estimation plots](docs/source/_images/showpiece.png?raw=true "The five kinds of estimation plots.")

DABEST powers [estimationstats.com](https://www.estimationstats.com/), allowing everyone access to high-quality estimation plots.


## Installation

This package is tested on Python 3.5, 3.6, and 3.7.
It is highly recommended to download the [Anaconda distribution](https://www.continuum.io/downloads) of Python in order to obtain the dependencies easily.

You can install this package via `pip`.

To install, at the command line run
<!-- ```shell
conda config --add channels conda-forge
conda install dabest
```
or -->
```shell
pip install --upgrade dabest
```
You can also [clone](https://help.github.com/articles/cloning-a-repository) this repo locally.

Then, navigate to the cloned repo in the command line and run

```shell
pip install .
```


## Usage

```python3
import pandas as pd
import dabest

# Load the iris dataset. Requires internet access.
iris = pd.read_csv("https://github.com/mwaskom/seaborn-data/raw/master/iris.csv")

# Load the above data into `dabest`.
iris_dabest = dabest.load(data=iris, x="species", y="petal_width",
                          idx=("setosa", "versicolor", "virginica"))

# Produce a Cumming estimation plot.
iris_dabest.mean_diff.plot();
```
![A Cumming estimation plot of petal width from the iris dataset](https://github.com/ACCLAB/DABEST-python/blob/master/iris.png)

Please refer to the official [tutorial](https://acclab.github.io/DABEST-python-docs/tutorial.html) for more useful code snippets.


## How to cite

**Moving beyond P values: Everyday data analysis with estimation plots**

*Joses Ho, Tayfun Tumkaya, Sameer Aryal, Hyungwon Choi, Adam Claridge-Chang*

Nature Methods 2019, 1548-7105. [10.1038/s41592-019-0470-3](http://dx.doi.org/10.1038/s41592-019-0470-3)

[Paywalled publisher site](https://www.nature.com/articles/s41592-019-0470-3); [Free-to-view PDF](https://rdcu.be/bHhJ4)


## Bugs

Please report any bugs on the [Github issue tracker](https://github.com/ACCLAB/DABEST-python/issues/new).


## Contributing

All contributions are welcome; please read the [Guidelines for contributing](https://github.com/ACCLAB/DABEST-python/blob/master/CONTRIBUTING.md) first.

We also have a [Code of Conduct](https://github.com/ACCLAB/DABEST-python/blob/master/CODE_OF_CONDUCT.md) to foster an inclusive and productive space.


## Acknowledgements

We would like to thank alpha testers from the [Claridge-Chang lab](https://www.claridgechang.net/): [Sangyu Xu](https://github.com/sangyu), [Xianyuan Zhang](https://github.com/XYZfar), [Farhan Mohammad](https://github.com/farhan8igib), Jurga Mituzaitė, and Stanislav Ott.


## Testing

To test DABEST, you will need to install [pytest](https://docs.pytest.org/en/latest).

Run `pytest` in the root directory of the source distribution. This runs the test suite in the folder `dabest/tests`. The test suite will ensure that the bootstrapping functions and the plotting functions perform as expected.


## DABEST in other languages

DABEST is also available in R ([dabestr](https://github.com/ACCLAB/dabestr)) and Matlab ([DABEST-Matlab](https://github.com/ACCLAB/DABEST-Matlab)).
