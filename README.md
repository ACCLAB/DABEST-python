# DABEST (Python)
[![Travis CI](https://travis-ci.org/ACCLAB/DABEST-python.svg)](https://travis-ci.org/ACCLAB/DABEST-python)
[![PyPI](https://img.shields.io/pypi/v/dabest.svg)](https://pypi.python.org/pypi/dabest/0.1.4)

## About

DABEST is a package for **D**ata **A**nalysis using **B**ootstrapped **EST**imation.

![Two-group contrast plot](https://www.github.com/ACCLAB/DABEST-Python/img/readme-fig.png)

## Requirements

Python 3.6 is strongly recommended, although this has been tested with Python 2.7 and Python 3.5.

In addition, the following packages are also required:
- [numpy](https://www.numpy.org/) (1.13.x)
- [scipy](https://www.scipy.org/) (1.0.x)
- [matplotlib](https://www.matplotlib.org/) (2.0.x)
- [seaborn](https://seaborn.pydata.org/) (0.8.x)
- [pandas](https://pandas.pydata.org/) (0.23.x).

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
You can also clone this repo locally (see intstructions [here](https://help.github.com/articles/cloning-a-repository/)).

Then, navigate to the cloned repo in the command line and run

```shell
pip install .
```


## Usage

Please refer to the [documentation](https://acclab.github.io/DABEST-python-docs/index.html).


## Matlab version

There is also a [Matlab version](https://github.com/ACCLAB/DABEST-Matlab) of DABEST.


## R version

There isn't an implementation of DABEST in R, and there are no current plans to create one.

However, it is possible to use the R package `reticulate` to run Python code. Please take a look at this [tutorial](https://acclab.github.io/DABEST-python-docs/dabest-r.html) on how to use `reticulate` to analyse data in R.
