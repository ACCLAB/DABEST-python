.. dabest documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive

======
DABEST
======

Update
------
**DABEST version 2023.02.14** has been released: 
:doc:`release-notes`

Briefly, this release introduces several new functions:

  - Additional plotting functions for repeated measures experimental designs (:doc:`repeatedmeasures`)
  - Calculating of Cohen's *h* and proportional plots for binary data (:doc:`proportion-plot`)
  - Calculating and plotting delta-deltas for 2x2 experimental designs (:doc:`deltadelta`)
  - Calculating and plotting of weighted deltas for mini-meta-analysis of experimental replicates (:doc:`minimetadelta`)

Refer to :doc:`release-notes` for full details.


-----------------------------------------------
Data Analysis with Bootstrap-coupled ESTimation
-----------------------------------------------

Analyze your data with estimation statistics!
---------------------------------------------

.. image:: _images/showpiece.png


News
----
March 2023:
  - **v2023.02.14** released. The following features have been added:
     - Additional plotting functions for repeated measures experimental designs
     - Calculating of Cohen's *h* and proportional plots for binary data
     - Calculating and plotting delta-deltas for 2x2 experimental designs
     - Calculating and plotting of weighted deltas for mini-meta-analysis of experimental replicates
  - See :doc:`release-notes` for more details.

October 2020:
  - v0.3.1 released. The minimal versions of dependencies have been upgraded. Also, the minimal version of Python required is now 3.6.

January 2020:
 - v0.3.0 released. Approximate permutation tests have been added, and are now the default p-values reported in the textual output. The LqRT tests were also refactored to a user-callable property. For more information, see the :doc:`release-notes`.

December 2019:
  - v0.2.8 released. This release adds the `Lq-Likelihood-Ratio-Type Test <https://github.com/alyakin314/lqrt>`_ in the statistical output, and also a bugfix for  slopegraph and reference line keyword parsing.

October 2019:
  - v0.2.7 released. A minor bugfix in the handling of wide datasets with unequal Ns in each group.
  - v0.2.6 released. This release has one new feature (plotting of estimation plot inside any :py:mod:`matplotlib` :py:class:`Axes`; see the section on :ref:`inset plot` in the :doc:`plotaesthetics`). There are also two bug patches for the handling of bootstrap plotting, and of dependency installation.

September 2019:
  - v0.2.5 released. This release addresses two feature requests, and also patches two bugs: one affecting the paired difference CIs, and one involving NaNs in unused/irrelevant columns.

May 2019:
  - v0.2.4 released. This is a patch for a set of bugs that mis-aligned Gardner-Altman plots, and also adds the capability to tweak the x-position of the Tufte gapped lines.

  - v0.2.3 released. This is a fix for a bug that did not properly handle x-columns which were pandas Categorical objects.

April 2019:
  - v0.2.2 released. This is a minor bugfix that addressed an issue for an edge case where the mean or median difference was exactly zero.

March 2019:
  - v0.2.1 released. This is a minor bugfix that addressed an issue in gapped line plotting.
  - v0.2.0 released. This is a major update that makes several breaking changes to the API.

Contents
--------

.. toctree::
  :maxdepth: 1

  robust-beautiful
  bootstraps
  getting-started
  tutorial
  repeatedmeasures
  proportion-plot
  minimetadelta
  deltadelta
  plotaesthetics
  release-notes
  api
  about
  citation
