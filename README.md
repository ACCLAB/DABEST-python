# DABEST (Python)
[![Travis CI](https://travis-ci.org/ACCLAB/DABEST-python.svg)](https://travis-ci.org/ACCLAB/DABEST-python)
[![PyPI](https://badge.fury.io/py/dabest.svg)](https://pypi.python.org/pypi/dabest)

## About

DABEST is a package for **D**ata **A**nalysis using **B**ootstrap-Coupled **EST**imation.
![Two-group contrast plot](docs/source/_images/showpiece.png?raw=true "The five kinds of estimation plots.")

## Requirements

DABEST has been tested on Python 2.7, 3.5, 3.6, and 3.7.

In addition, the following packages are also required:
- [numpy](https://www.numpy.org) (1.15)
- [scipy](https://www.scipy.org) (1.1)
- [matplotlib](https://www.matplotlib.org) (2.2)
- [seaborn](https://seaborn.pydata.org) (0.9)
- [pandas](https://pandas.pydata.org) (0.23).

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

Please refer to the [documentation](https://acclab.github.io/DABEST-python-docs/index.html).


## How to cite

Moving beyond P values: Everyday data analysis with estimation plots
Joses Ho, Tayfun Tumkaya, Sameer Aryal, Hyungwon Choi, Adam Claridge-Chang
[https://doi.org/10.1101/377978](https://doi.org/10.1101/377978)


## Matlab version

There is also a [Matlab version](https://github.com/ACCLAB/DABEST-Matlab) of DABEST.


## R version

It is possible to use the R package `reticulate` to run Python code. Please take a look at this [tutorial](https://acclab.github.io/DABEST-python-docs/dabest-r.html) on how to use `reticulate` to analyse data in R.
