#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com

"""
**dabest** is a Python package for Data Analysis and Visualization using Bootstrap-Coupled Estimation.

Estimation statistics is a simple framework <https://thenewstatistics.com/itns/> that—while avoiding the pitfalls of significance testing—uses familiar statistical concepts: means, mean differences, and error bars. More importantly, it focuses on the effect size of one's experiment/intervention, as opposed to significance testing.

An estimation plot has two key features. Firstly, it presents all datapoints as a swarmplot, which orders each point to display the underlying distribution. Secondly, an estimation plot presents the effect size as a bootstrap 95% confidence interval on a separate but aligned axes.

**dabest** creates estimation plots for mean differences, median differences, standardized effect sizes (Cohen's _d_ and Hedges' _g_), and ordinal effect sizes (Cliff's delta).

Please cite this work as:
Moving beyond P values: Everyday data analysis with estimation plots
Joses Ho, Tayfun Tumkaya, Sameer Aryal, Hyungwon Choi, Adam Claridge-Chang
https://doi.org/10.1101/377978
"""


from ._api import load
from ._stats_tools import effsize as effsize
from ._classes import TwoGroupsEffectSize 

__version__ = "0.2.4"
