.. dabest documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive

======
DABEST
======

-----------------------------------------------
Data Analysis with Bootstrap-coupled ESTimation
-----------------------------------------------

Analyze your data with estimation statistics!
---------------------------------------------

.. image:: _images/showpiece.png


News
----
December 2019:
  - v0.2.8 released. This release adds the `Lq-Likelihood-Ratio-Type Test <https://github.com/alyakin314/lqrt>`_ in the statistical output, and also a bugfix for  slopegraph and reference line keyword parsing. For more information, see the :doc:`release-notes`.

October 2019:
  - v0.2.7 released. A minor bugfix in the handling of wide datasets with unequal Ns in each group. 
  - v0.2.6 released. This release has one new feature (plotting of estimation plot inside any :py:mod:`matplotlib` :py:class:`Axes`; see the section on :ref:`inset plot` in the :doc:`tutorial`). There are also two bug patches for the handling of bootstrap plotting, and of dependency installation. 

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
  :maxdepth: 2

  robust-beautiful
  bootstraps
  getting-started
  tutorial
  release-notes
  api
  about
