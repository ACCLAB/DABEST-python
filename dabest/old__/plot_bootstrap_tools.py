from collections import OrderedDict
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import seaborn.apionly as sns
import pandas as pd
import numpy as np
import warnings

def plotbootstrap(coll, bslist, ax, violinWidth, 
                  violinOffset, marker='o', color='k', 
                  markerAlpha=0.75,
                  markersize=None,
                  CiAlpha=0.75,
                  offset=True,
                  linewidth=2, 
                  rightspace=0.2,
                 **kwargs):
    '''subfunction to plot the bootstrapped distribution along with BCa intervals.'''
    if markersize is None:
         mSize=12.
    else:
        mSize=markersize

    autoxmin=ax.get_xlim()[0]
    x, _=np.array(coll.get_offsets()).T
    xmax=x.max()

    if offset:
        violinbasex=xmax + violinOffset
    else:
        violinbasex=1
        
    # array=list(bslist.items())[7][1]
    array=bslist['diffarray']
    
    v=ax.violinplot(array, [violinbasex], 
                      widths=violinWidth * 2, 
                      showextrema=False, showmeans=False)
    
    for b in v['bodies']:
        m=np.nanmean(b.get_paths()[0].vertices[:, 0])
        b.get_paths()[0].vertices[:, 0]=np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
        b.set_color('k')
    
    # Plot the summary measure.
    ax.plot(violinbasex, bslist['summary'],
             marker=marker,
             markerfacecolor=color, 
             markersize=mSize,
             alpha=markerAlpha
            )

    # Plot the CI.
    ax.plot([violinbasex, violinbasex],
             [bslist['bca_ci_low'], bslist['bca_ci_high']],
             color=color, 
             alpha=CiAlpha,
             linestyle='solid'
            )
    
    ax.set_xlim(autoxmin, (violinbasex + violinWidth + rightspace))
    
    if array.min() < 0 < array.min():
        ax.set_ylim(array.min(), array.max())
    elif 0 <= array.min(): 
        ax.set_ylim(0, array.max() * 1.1)
    elif 0 >= array.max():
        ax.set_ylim(array.min() * 1.1, 0)
        
def plotbootstrap_hubspoke(bslist, ax, violinWidth, violinOffset, 
                           marker='o', color='k', 
                           markerAlpha=0.75,
                           markersize=None,
                           CiAlpha=0.75,
                           linewidth=2,
                          **kwargs):
    
    '''subfunction to plot the bootstrapped distribution along with BCa intervals for hub-spoke plots.'''

    if markersize is None:
        mSize=12.
    else:
        mSize=markersize

    ylims=list()
    
    for i in range(0, len(bslist)):
        bsi=bslist[i]
        # array=list(bsi.items())[7][1] # Pull out the bootstrapped array.
        array=bsi['diffarray']
        ylims.append(array)
        
        # Then plot as violinplot.
        v=ax.violinplot(array, [i+1], 
                          widths=violinWidth * 2, 
                          showextrema=False, showmeans=False)
        
        for b in v['bodies']:
            m=np.mean(b.get_paths()[0].vertices[:, 0])
            b.get_paths()[0].vertices[:, 0]=np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
            b.set_color('k')
            # Plot the summary measure.
            ax.plot(i+1, bsi['summary'],
                     marker=marker,
                     markerfacecolor=color, 
                     markersize=mSize,
                     alpha=markerAlpha
                    )

            # Plot the CI.
            ax.plot([i+1, i+1],
                     [bsi['bca_ci_low'], bsi['bca_ci_high']],
                     color=color, 
                     alpha=CiAlpha,
                     linestyle='solid'
                    )
            
    ylims=np.array(ylims).flatten()
    if ylims.min() < 0 and ylims.max() < 0: # All effect sizes are less than 0.
        ax.set_ylim(1.1 * ylims.min(), 0)
    elif ylims.min() > 0:                   # All effect sizes are more than 0.
        ax.set_ylim(-0.25, 1.1 * ylims.max())
    elif ylims.min() < 0 < ylims.max():     # One or more effect sizes straddle 0.
        ax.set_ylim(1.1 * ylims.min(), 1.1 * ylims.max())

def swarmsummary(data, x, y, idx=None, statfunction=None, 
                 violinOffset=0.1, violinWidth=0.2, 
                 figsize=(7,7), legend=True,
                 smoothboot=False,
                 rawMarkerSize=10,
                 summaryMarkerSize=12,
                 rawMarkerType='o',
                 summaryMarkerType='o',
                 **kwargs):
    df=data # so we don't re-order the rawdata!
    # initialise statfunction
    if statfunction == None:
        statfunction=np.mean
        
    # calculate bootstrap list.
    bslist=OrderedDict()

    if idx is None:
        levs=df[x].unique()   # DO NOT USE the numpy.unique() method.
                                # It will not preserve the order of appearance of the levels.
    else:
        levs=idx

    for i in range (0, len(levs)):
        temp_df=df.loc[df[x] == levs[i]]
        bslist[levs[i]]=bootstrap(temp_df[y], statfunction=statfunction, smoothboot=smoothboot)
    
    bsplotlist=list(bslist.items())
    
    # Initialise figure
    #sns.set_style('ticks')
    fig, ax=plt.subplots(figsize=figsize)
    sw=sns.swarmplot(data=df, x=x, y=y, order=levs, 
      size=rawMarkerSize, marker=rawMarkerType, **kwargs)
    y_lims=list()
    
    for i in range(0, len(bslist)):
        plotbootstrap(sw.collections[i], 
                      bslist=bsplotlist[i][1], 
                      ax=ax, 
                      violinWidth=violinWidth, 
                      violinOffset=violinOffset,
                      marker=summaryMarkerType,
                      markersize=summaryMarkerSize,
                      color='k', 
                      linewidth=2)
        
        # Get the y-offsets, save into a list.
        _, y=np.array(sw.collections[i].get_offsets()).T 
        y_lims.append(y)
    
    # Concatenate the list of y-offsets
    y_lims=np.concatenate(y_lims)
    ax.set_ylim(0.9 * y_lims.min(), 1.1 * y_lims.max())
    
    if legend is True:
        ax.legend(loc='center left', bbox_to_anchor=(1.1, 1))
    elif legend is False:
        ax.legend().set_visible(False)
        
    sns.despine(ax=ax, trim=True)
    
    return fig, pd.DataFrame.from_dict(bslist)