.. _Release Notes:

=============
Release Notes
=============


v0.1.6
======
Several keywords have been added to allow more fine-grained control over a selection of plot elements.

* `swarm_dotsize`
* `difference_dotsize`
* `ci_linewidth`
* `summary_linewidth`

The new keyword `context` allows you to set the plotting context as defined by seaborn's `plotting_context() <https://seaborn.pydata.org/generated/seaborn.plotting_context.html>`_ .

Now, if `paired=True`, you will need to supply an `id_col`, which is a column in the DataFrame which specifies which sample the datapoint belongs to. See the :doc:`tutorial` for more details.


v0.1.5
======
Fix bug that wasn't updating the seaborn version upon setup and install.


v0.1.4
======
Update dependencies to

* numpy 1.15
* scipy 1.1
* matplotlib 2.2
* seaborn 0.9

Aesthetic changes

* add `tick_length` and `tick_pad` arguments to allow tweaking of the axes tick lengths, and padding of the tick labels, respectively.

v0.1.3
=====
Update dependencies to

* pandas v0.23

Bugfixes

* fix bug that did not label `swarm_label` if raw data was in tidy form
* fix bug that did not dropnans for unpaired diff


v0.1.2
======
Update dependencies to

* numpy v1.13
* scipy v1.0
* pandas v0.22
* seaborn v0.8


v0.1.1
=======
`Update LICENSE to BSD-3 Clear. <https://github.com/ACCLAB/DABEST-python/commit/615c4cbb9145cf7b9451bf1840a20475ebcb2e99>`_
