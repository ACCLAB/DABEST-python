#! /usr/bin/env python

# #! /usr/bin/env python
# import numpy as np
#
# import tests
#
# # Load Libraries
# import matplotlib as mpl
# import matplotlib.pyplot as plt
#
# from matplotlib.testing.decorators import image_comparison
#
# # Use SVG renderer.
# # mpl.use('SVG')
# mpl.use('TkAgg')
# # Ensure that text is rendered as text and not as paths.
# plt.rcParams['svg.fonttype'] = 'none'
#
# import seaborn as sns
# sns.set(style='ticks',context='talk')
#
# savefig_kwargs = {'transparent': True,
#                  'frameon': False,
#                  'bbox_inches': 'tight',
#                  'format': 'svg'}
#
#
# # f.savefig('testfig.svg', **savefig_kwargs)

def create_dummy_dataset(n=50):
    import pandas as pd
    import numpy as np

    # Dummy dataset
    Ns = n
    dataset = list()
    for seed in [10,11,12,13,14,15]:
        # fix the seed so we get the same numbers each time.
        np.random.seed(seed)
        dataset.append(np.random.randn(Ns))
    df = pd.DataFrame(dataset).T
    cols = ['Control','Group1','Group2','Group3','Group4','Group5']
    df.columns = cols
    # Create some upwards/downwards shifts.
    df['Group2'] = df['Group2'] - 0.1
    df['Group3'] = df['Group3'] + 0.2
    df['Group4'] = (df['Group4']*1.1) + 4
    df['Group5'] = (df['Group5']*1.1) - 1
    # Add gender column for color.
    df['Gender'] = np.concatenate([np.repeat('Male', Ns/2),
                                  np.repeat('Female', Ns/2)])

    return df

def Gardner_Altman_unpaired(df):
    from .. import api

    return api.plot(data=df,
                       idx=('Control','Group1'))

Gardner_Altman_unpaired(create_dummy_dataset())

def Gardner_Altman_paired(df):
    from .. import api

    return api.plot(data=df,
                       idx=('Control','Group1'),
                       paired=True)

def Cumming_two_group_unpaired(df):
    from .. import api

    return api.plot(data=df,
                      idx=('Control','Group1'),
                      float_contrast=True)

def Cumming_two_group_paired(df):
    from .. import api

    return api.plot(data=df,
                       idx=('Control','Group1'),
                       paired=True
                       float_contrast=True)

def custom_swarm_label(df):
    from .. import api

    return api.plot(data=df,
                       idx=('Control','Group1'),
                       swarm_label='my swarm',
                       contrast_label='contrast')

def custom_contrast_label(df):
    from .. import api

    return api.plot(data=df,
                       idx=('Control','Group1'),
                       contrast_label='contrast')

def with_color_col(df):
    from .. import api

    return api.plot(data=df,
                       idx=('Control','Group1'),
                       color_col='Gender')

def with_custom_palette(df):
    import dabest

    return dabest.plot(data=df,
                       idx=('Control','Group1'),
                       color_col='Gender',
                       custom_palette={'Male':'blue',
                                       'Female':'red'}
                        )
