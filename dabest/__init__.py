from ._api_old import plot
# from .stats_tools.confint_2group_diff import difference_ci
from .stats_tools.confint_1group import summary_ci_1group
from .stats_tools import effsize as effsize

# DEV
from .api import load
from .classes import TwoGroupsEffectSize, EffectSizeDataFrame
from .stats_tools import confint_2group_diff as ci2g
from .plot_tools import halfviolin

__version__ = "0.2.0"
