.. _getting-started:

===============
Getting Started
===============

------------
Requirements
------------

Python 3.6 is strongly recommended. ``dabest`` has also been tested with Python 2.7 and Python 3.5.

In addition, the following packages are also required:

* `numpy <https://www.numpy.org>`_ (1.13.x)
* `scipy <https://www.scipy.org>`_ (1.0.x)
* `matplotlib <https://www.matplotlib.org>`_ (2.0.x)
* `pandas <https://pandas.pydata.org>`_ (0.23.x).
* `seaborn <https://seaborn.pydata.org>`_ (0.8.x)

To obtain these package dependencies easily, it is highly recommended to download the `Anaconda <https://www.continuum.io/downloads>`_ distribution of Python.

------------
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


-------
Testing
-------

To test ``dabest``, you will need to install `pytest <https://docs.pytest.org/en/latest/>`_.

Run ``pytest`` in the root directory of the source distribution. This runs the test suite in ``dabest/tests`` folder. The test suite will ensure that the bootstrapping functions and the plotting functions perform as expected.

----
Bugs
----
Please report any bugs on the `Github issue tracker <https://github.com/ACCLAB/DABEST-python/issues/new>`_ for DABEST-python.

------------
Contributing
------------
All contributions are welcome. Please fork the `Github repo    <https://github.com/ACCLAB/DABEST-python/>`_ and open a pull request.
