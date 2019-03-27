# DABEST (Python)
[![Travis CI](https://travis-ci.org/ACCLAB/DABEST-python.svg?branch=master)](https://travis-ci.org/ACCLAB/DABEST-python)
[![PyPI version](https://badge.fury.io/py/dabest.svg)](https://badge.fury.io/py/dabest)

## About

DABEST is a package for **D**ata **A**nalysis using **B**ootstrap-Coupled **EST**imation.

[Estimation statistics](https://en.wikipedia.org/wiki/Estimation_statistics) is a [simple framework](https://thenewstatistics.com/itns/) that avoids the [pitfalls](https://www.nature.com/articles/nmeth.3288) of significance testing. It uses familiar statistical concepts: means, mean differences, and error bars. More importantly, it focuses on the effect size of one's experiment/intervention, as opposed to a false dichotomy engendered by *P* values.

An estimation plot has two key features.

1. It presents all datapoints as a swarmplot, which orders each point to display the underlying distribution.

2. It presents the effect size as a **bootstrap 95% confidence interval** on a **separate but aligned axes**.

![The five kinds of estimation plots](docs/source/_images/showpiece.png?raw=true "The five kinds of estimation plots.")

DABEST powers [estimationstats.com](https://www.estimationstats.com/), allowing everyone access to high-quality estimation plots.

## Requirements

DABEST has been tested on Python 3.5, 3.6, and 3.7.

In addition, the following packages are also required:
- [numpy](https://www.numpy.org) (1.15)
- [scipy](https://www.scipy.org) (1.2)
- [matplotlib](https://www.matplotlib.org) (3.0)
- [seaborn](https://seaborn.pydata.org) (0.9)
- [pandas](https://pandas.pydata.org) (0.24).

To obtain these package dependencies easily, it is highly recommended to download the [Anaconda distribution](https://www.continuum.io/downloads) of Python.

## Installation

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
You can also clone this repo locally (see intstructions [here](https://help.github.com/articles/cloning-a-repository)).

Then, navigate to the cloned repo in the command line and run

```shell
pip install .
```

## Usage

Please refer to the [documentation](https://acclab.github.io/DABEST-python-docs).

## How to cite

**Moving beyond P values: Everyday data analysis with estimation plots**

*Joses Ho, Tayfun Tumkaya, Sameer Aryal, Hyungwon Choi, Adam Claridge-Chang*

[https://doi.org/10.1101/377978](https://doi.org/10.1101/377978)


## Matlab version

There is also a [Matlab version](https://github.com/ACCLAB/DABEST-Matlab) of DABEST.


## R version

R users can install [dabestr](https://github.com/ACCLAB/dabestr).



## Testing

To test DABEST, you will need to install [pytest](https://docs.pytest.org/en/latest).

Run `pytest` in the root directory of the source distribution. This runs the test suite in the folder `dabest/tests`. The test suite will ensure that the bootstrapping functions and the plotting functions perform as expected.


## Bugs

Please report any bugs on the [Github issue tracker](https://github.com/ACCLAB/DABEST-python/issues/new).


## Contributing

All contributions are welcome. Please fork the [Github repo](https://github.com/ACCLAB/DABEST-python) and open a pull request.
