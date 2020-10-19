.. _Release Notes:

=============
Release Notes
=============

v0.3.1
------

This release updates the package requirements to:
  - :py:mod:`numpy`: 0.19
  - :py:mod:`matplotlib`: 3.3
  - :py:mod:`scipy`: 1.5
  - :py:mod:`pandas`: 1.1
  - :py:mod:`seaborn`: 0.11
  - :py:mod:`lqrt`: 0.3


v0.3.0
------
This is a new major release that refactors the statistical test output. All users are strongly encouraged to update to this release.

Main Changes:
  - the Lq-Likelihood Ratio-Type test results are now computed only when the user calls the `lqrt` property of the `dabest_object.effect_size` :py:class:`EffectSizeDataFrame` object. This refactor was done to mitigate the lengthy computation time the test took. See Issues `#94 <https://github.com/ACCLAB/DABEST-python/issues/94>`_ and `#91 <https://github.com/ACCLAB/DABEST-python/issues/91>`_.
  - The default hypothesis test reported by calling the `dabest_object.effect_size` :py:class:`EffectSizeDataFrame` object is now the `non-parametric two-sided permutation t-test <https://en.wikipedia.org/wiki/Resampling_(statistics)#Permutation_tests>`_. Conceptually, this hypothesis test mirrors the usage of the bootstrap (to obtain the 95% confidence intervals). Read more at :doc:`tutorial`. See Issue `#92 <https://github.com/ACCLAB/DABEST-python/issues/92>`_ and PR `#93 <https://github.com/ACCLAB/DABEST-python/issues/93>`_.
  - The minimum version of :py:mod:`numpy` is now v0.17, which has an updated method of generating random samples. The resampling code used in :py:mod:`dabest` has thus been updated as well.


v0.2.8
------

This release fixes minor bugs, and implements a new statistical test.

Feature Additions:
  -  Implement `Lq-Likelihood-Ratio-Type Test <https://github.com/alyakin314/lqrt>`_ in statistical output with `PR #85 <https://github.com/ACCLAB/DABEST-python/pull/85>`_; thanks to Adam Li (`@adam2392 <https://github.com/adam2392>`_).

Bug-fixes:
  - Fix bugs in slopegraph and reference line keyword parsing with `PR #86 <https://github.com/ACCLAB/DABEST-python/pull/86>`_; thanks to DizietAsahi (`DizietAsahi <https://github.com/DizietAsahi>`_).



v0.2.7
------

Bug-fixes:
  - Bug affecting display of Tufte gapped lines in Cumming plots if the supplied :py:mod:`pandas` :py:class:`DataFrame` was in 'wide' format, but did not have equal number of Ns in the groups. (`Issue #79 <https://github.com/ACCLAB/DABEST-python/issues/79>`_)


v0.2.6
------

Feature additions:
  - It is now possible to specify a pre-determined :py:mod:`matplotlib` :py:class:`Axes` to create the estimation plot in. See :ref:`inset plot` in the :doc:`tutorial` (`Pull request #73 <https://github.com/ACCLAB/DABEST-python/pull/73>`_; thanks to Adam Nekimken (`@anekimken <https://github.com/anekimken>`_).
  -


Bug-fixes:
  - Ensure all dependencies are installed along with DABEST. (`Pull request #71 <https://github.com/ACCLAB/DABEST-python/pull/71>`_; thanks to Matthew Edwards (`@mje-nz <https://github.com/mje-nz>`_).
  - Handle infinities in bootstraps during plotting. (`Issue #72 <https://github.com/ACCLAB/DABEST-python/issues/72>`_, `Pull request #74 <https://github.com/ACCLAB/DABEST-python/pull/71>`_)

v0.2.5
------

Feature additions:
  - Adding Ns of each group to the results DataFrame. (`Issue #45 <https://github.com/ACCLAB/DABEST-python/issues/45>`_)
  - Auto-labelling the swarmplot rawdata axes y-label. (`Issue #51 <https://github.com/ACCLAB/DABEST-python/issues/51>`_)

Bug-fixes:
  - Bug affecting calculation of paired difference confidence intervals. (`Issue #48 in ACCLAB/dabestr <https://github.com/ACCLAB/dabestr/issues/48>`_)
  - NaNs in unused/unrelated columns would result in null results (`Issue #44 <https://github.com/ACCLAB/DABEST-python/issues/44>`_)


v0.2.4
------

This release fixes the following issues:
  - Misalignment of Gardner-Altman plots when the dataset loaded is wide, but has NaNs in a column. (`Issue #40 <https://github.com/ACCLAB/DABEST-python/issues/40>`_)
  - Misalignment of Hedges' g Gardner Altman plots (Also Issue #40).
  - Add ``groups_summaries_offset`` argument for better control over gapped Tufte line plotting. The default offset is now set at 0.1 as well. (`Issue #35 <https://github.com/ACCLAB/DABEST-python/issues/35>`_

v0.2.3
------

This release fixes a bug that did not handle when the supplied ``x`` was a :py:mod:`pandas` :py:class:`Categorical` object, but the ``idx`` did not include all the original categories.


v0.2.2
------

This release fixes a `bug <https://github.com/ACCLAB/DABEST-python/pull/30>`_ that has a mean difference or median difference of exactly 0. (`Pull request #73 <https://github.com/ACCLAB/DABEST-python/pull/73>`_; thanks to Mason Malone (`@MasonM <https://github.com/MasonM>`_).


v0.2.1
------

This release fixes a bug that misplotted the gapped summary lines in Cumming plots when the *x*-variable was a :py:mod:`pandas` :py:class:`Categorical` object.


v0.2.0
------

We have redesigned the interface from the ground up. This allows speed and flexibility to compute different effect sizes (including Cohen's *d*, Hedges' *g*, and Cliff's delta). Statistical arguments are now parsed differently from graphical arguments.

In short, any code relying on v0.1.x will **not work with v0.2.0, and must be upgraded.**

Now, every analysis session begins with ``dabest.load()``.

.. code-block:: python
    :linenos:

    my_data = dabest.load(my_dataframe, idx=("Control", "Test"))

This creates a :py:class:`Dabest` object with effect sizes as instances.

.. code-block:: python
    :linenos:

    my_data.mean_diff

which prints out:

.. parsed-literal::

    DABEST v0.2.0
    =============

    Good afternoon!
    The current time is Mon Mar  4 17:03:29 2019.

    The unpaired mean difference between Control 1 and Test 1 is 0.48 [95%CI 0.205, 0.774].

    5000 bootstrap samples were taken; the confidence interval is bias-corrected and accelerated.
    The p-value(s) reported are the likelihood(s) of observing the effect size(s),
    if the null hypothesis of zero difference is true.

The following are valid effect sizes:

.. code-block:: python
    :linenos:

    my_data.mean_diff
    my_data.median_diff
    my_data.cohens_d
    my_data.hedges_g
    my_data.cliffs_delta

To produce an estimation plot, each effect size instance has a ``plot()`` method.

.. code-block:: python
    :linenos:

    my_data.mean_diff.plot()

See the :doc:`tutorial`  and :doc:`api` for more details, including keyword options for the ``load()`` and ``plot()`` methods.


v0.1.7
------

The keyword ``cumming_vertical_spacing`` has been added to tweak the vertical spacing between the rawdata swarm axes and the contrast axes in Cumming estimation plots.

v0.1.6
------

Several keywords have been added to allow more fine-grained control over a selection of plot elements.

* ``swarm_dotsize``
* ``difference_dotsize``
* ``ci_linewidth``
* ``summary_linewidth``

The new keyword ``context`` allows you to set the plotting context as defined by seaborn's `plotting_context() <https://seaborn.pydata.org/generated/seaborn.plotting_context.html>`_ .

Now, if ``paired=True``, you will need to supply an ``id_col``, which is a column in the DataFrame which specifies which sample the datapoint belongs to. See the :doc:`tutorial` for more details.


v0.1.5
------
Fix bug that wasn't updating the seaborn version upon setup and install.


v0.1.4
------
Update dependencies to

* numpy 1.15
* scipy 1.1
* matplotlib 2.2
* seaborn 0.9

Aesthetic changes

* add ``tick_length`` and ``tick_pad`` arguments to allow tweaking of the axes tick lengths, and padding of the tick labels, respectively.


v0.1.3
------
Update dependencies to

* pandas v0.23

Bugfixes

* fix bug that did not label ``swarm_label`` if raw data was in tidy form
* fix bug that did not dropnans for unpaired diff


v0.1.2
------
Update dependencies to

* numpy v1.13
* scipy v1.0
* pandas v0.22
* seaborn v0.8


v0.1.1
------
`Update LICENSE to BSD-3 Clear. <https://github.com/ACCLAB/DABEST-python/commit/615c4cbb9145cf7b9451bf1840a20475ebcb2e99>`_
