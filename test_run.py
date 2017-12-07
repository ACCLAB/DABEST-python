#! /usr/bin/env python

# Load Libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
# Use SVG renderer.
# mpl.use('SVG')
mpl.use('TkAgg')
# Ensure that text is rendered as text and not as paths.
plt.rcParams['svg.fonttype'] = 'none'

import seaborn as sns
sns.set(style='ticks',context='talk')
import dabest

import pandas as pd
import numpy as np
import scipy as sp

# Dummy dataset
Ns = 50
dataset = list()
for seed in [10,11,12,13,14,15]:
    np.random.seed(seed) # fix the seed so we get the same numbers each time.
    dataset.append(np.random.randn(Ns))
df = pd.DataFrame(dataset).T
cols = ['Control','Group1','Group2','Group3','Group4','Group5']
df.columns = cols
# Create some upwards/downwards shifts.
df['Group2'] = df['Group2']-0.1
df['Group3'] = df['Group3']+0.2
df['Group4'] = (df['Group4']*1.1)+4
df['Group5'] = (df['Group5']*1.1)-1
# Add gender column for color.
df['Gender'] = np.concatenate([np.repeat('Male', Ns/2),
                              np.repeat('Female', Ns/2)])

f,c = dabest.plot(data=df,
                  idx=(('Group1','Group3','Group2'),
                       ('Control','Group4')),
                  color_col='Gender',
                  custom_palette={'Male':'blue',
                                  'Female':'red'},
                  float_contrast=True,
                  swarm_label='my swarm',
                  contrast_label='contrast',
                  fig_size=(10,8))

# Save the figure.
f.savefig('testfig.svg',
          transparent=True,
          # ensure no white background on plot
          frameon=False,
          bbox_inches='tight',
          format='svg')
