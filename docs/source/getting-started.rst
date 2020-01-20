.. _Getting Started:

===============
Getting Started
===============


Requirements
------------

Python 3.7 is strongly recommended. DABEST has also been tested with Python 3.5 and 3.6.

In addition, the following packages are also required (listed with their minimal versions):

* `numpy 1.17 <https://www.numpy.org>`_
* `scipy 1.2 <https://www.scipy.org>`_
* `matplotlib 3.0 <https://www.matplotlib.org>`_
* `pandas 0.25.3 <https://pandas.pydata.org>`_
* `seaborn 0.9 <https://seaborn.pydata.org>`_
* `lqrt 0.3.2 <https://github.com/alyakin314/lqrt>`_

To obtain these package dependencies easily, it is highly recommended to download the `Anaconda <https://www.continuum.io/downloads>`_ distribution of Python.


Installation
------------

1. Using ``pip``

At the command line, run

.. code-block:: console

  $ pip install dabest

2. Using Github

Clone the `DABEST-python repo <https://github.com/ACCLAB/DABEST-python>`_ locally (see instructions `here <https://help.github.com/articles/cloning-a-repository/>`_).

Then, navigate to the cloned repo in the command line and run

.. code-block:: console

  $ pip install .



Testing
-------

To test DABEST, you will need to install `pytest <https://docs.pytest.org/en/latest/>`_.

Run ``pytest`` in the root directory of the source distribution. This runs the test suite in ``dabest/tests`` folder. The test suite will ensure that the bootstrapping functions and the plotting functions perform as expected.


Bugs
----
Please report any bugs on the `Github issue tracker <https://github.com/ACCLAB/DABEST-python/issues/new>`_ for DABEST-python.


Contributing
------------
All contributions are welcome. Please fork the `Github repo <https://github.com/ACCLAB/DABEST-python/>`_ and open a pull request.
