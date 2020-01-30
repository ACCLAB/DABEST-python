.. _api:

.. currentmodule:: dabest

API
===

Loading Data
-------------

.. autofunction:: load


Computing Effect Sizes
----------------------

.. autoclass:: dabest._classes.Dabest
  :members: mean_diff, median_diff, cohens_d, hedges_g, cliffs_delta
  :member-order: bysource

.. .. autoclass:: dabest._classes.TwoGroupsEffectSize


Plotting Data
-------------

.. autoclass:: dabest._classes.EffectSizeDataFrame
  :members: plot, lqrt
  :member-order: bysource



Permutation Tests
-----------------

.. autoclass:: dabest._classes.PermutationTest