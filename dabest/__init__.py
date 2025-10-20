from ._api import load, prop_dataset
from ._stats_tools import effsize as effsize
from ._stats_tools import confint_2group_diff as ci_2g
from ._effsize_objects import TwoGroupsEffectSize, PermutationTest
from ._dabest_object import Dabest
from .forest_plot import forest_plot


import os
if os.environ.get('SKIP_NUMBA_COMPILE') != '1':
    from ._stats_tools.precompile import precompile_all, _NUMBA_COMPILED
    if not _NUMBA_COMPILED:
        precompile_all()

__version__ = "2025.10.20"