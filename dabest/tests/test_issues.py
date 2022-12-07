import pandas as pd
from .._api import load
from .utils import create_demo_dataset
import matplotlib.pyplot as plt
import numpy as np

data_103 = pd.read_csv("https://raw.githubusercontent.com/ZHANGROU-99/yieldCurve/main/F_Py.csv")
df_103 = load(data=data_103,x="Identifiers",y="Values",idx=(("S1","S2"),("I1","I2"),("D1","D2")),paired=True,id_col = "Pairs")

data_129 = pd.read_csv("https://raw.githubusercontent.com/ZHANGROU-99/yieldCurve/main/Data.csv")
df_129 = load(data=data_129, x="X",y="Y",idx=('Ref',"B"))


# def test_issue_103_color_col_formatting():
#     fig = df_103.mean_diff.plot(color_col="Type");
#     plt.show()
#
# def test_issue_129_median_CI():
#     fig = df_129.median_diff.plot(float_contrast=False)
#     plt.show()
#
# def test_issue_107_plot_only_diff():
#     from .utils import create_demo_dataset
#     from .. import plot_tools
#     import numpy as np
#     import seaborn as sns
#     df = create_demo_dataset()
#     two_groups_unpaired = load(df, idx=("Control 1", "Test 1"))
#     ax = plt.gca()
#     for row in two_groups_unpaired.mean_diff.results.itertuples():
#         v = ax.violinplot(row.bootstraps[~np.isinf(row.bootstraps)], positions=[row.Index])
#         plot_tools.halfviolin(v, fill_color=sns.color_palette()[row.Index])
#
#         # Effect size
#         ax.plot([row.Index], row.difference, marker="o", color=plt.rcParams["ytick.color"])
#         # Confidence interval
#         ax.plot([row.Index, row.Index], [row.bca_low, row.bca_high], linestyle="-", color=plt.rcParams["ytick.color"])
#
#     ax.set_xticks(two_groups_unpaired.mean_diff.results.index)
#     ax.set_xticklabels(two_groups_unpaired.mean_diff.results.test);
#     plt.show()

feeds = pd.read_csv("https://raw.githubusercontent.com/ZHANGROU-99/yieldCurve/main/feeds.csv")
df_new = load(data = feeds,  y = 'EverFeed', x = 'CombinedGroupBy', idx = tuple(np.unique(feeds.CombinedGroupBy)))

def test_01_cohens_h():
    df_new.cohens_h.plot()
    plt.show()

def test_01_mean_diff():
    df_new.mean_diff.plot()
    plt.show()

def test_02_not_binary():
    v = load(data=feeds, y='Volume_nl', x='CombinedGroupBy', idx=tuple(np.unique(feeds.CombinedGroupBy)))
    v.mean_diff.plot()
    plt.show()

def test_03_not_binary_cohens_h():
    v = load(data=feeds, y='Volume_nl', x='CombinedGroupBy', idx=tuple(np.unique(feeds.CombinedGroupBy)))
    v.cohens_h.plot()