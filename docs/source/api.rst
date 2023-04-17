.. _api:

.. currentmodule:: dabest

===
API
===

Loading Data
-------------

.. autofunction:: load


Computing Effect Sizes
----------------------

.. autoclass:: dabest._classes.Dabest
  :members: mean_diff, median_diff, cohens_d, hedges_g, cliffs_delta, cohens_h
  :member-order: bysource

.. .. autoclass:: dabest._classes.TwoGroupsEffectSize

.. autoclass:: dabest._classes.MiniMetaDelta
  :members: difference, bca_low, bca_high, bootstraps, to_dict

.. autoclass:: dabest._classes.DeltaDelta
  :members: difference, bca_low, bca_high, bootstraps, bootstraps_delta_delta, to_dict

Plotting Data
-------------

.. autoclass:: dabest._classes.EffectSizeDataFrame
  :members: plot, lqrt
  :member-order: bysource



Permutation Tests
-----------------

.. autoclass:: dabest._classes.PermutationTest