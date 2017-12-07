'''The bootstrapContrast module.'''
from __future__ import division

from scipy.stats import ttest_ind, ttest_1samp, ttest_rel, mannwhitneyu, norm
from collections import OrderedDict
from numpy.random import randint
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, MaxNLocator, LinearLocator, FixedLocator
from decimal import Decimal
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams, rcdefaults
import sys
import seaborn.apionly as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# This imports the custom functions used.
# These have been placed in separate .py files for reduced code clutter.
from .mpl_tools import rotateTicks, normalizeSwarmY, normalizeContrastY, offsetSwarmX, resetSwarmX, getSwarmSpan
from .mpl_tools import align_yaxis, halfviolin, drawback_y, drawback_x
from .bootstrap_tools import bootstrap, jackknife_indexes, bca
from .plot_bootstrap_tools import plotbootstrap, plotbootstrap_hubspoke, swarmsummary
## This is for sandboxing. Features and functions under testing go here.
# from .sandbox import contrastplot_test
from .prototype import cp_proto
from .plot_tools_ import halfviolin,align_yaxis,rotate_ticks

# Taken without modification from scikits.bootstrap package
# Keep python 2/3 compatibility, without using six. At some point,
# we may need to add six as a requirement, but right now we can avoid it.
try:
    xrange
except NameError:
    xrange = range

class InstabilityWarning(UserWarning):
    """Issued when results may be unstable."""
    pass

    
def contrastplot(
    data, x=None, y=None, idx=None, idcol=None,

    alpha=0.75, 
    axis_title_size=None,

    ci=95,
    contrastShareY=True,
    contrastEffectSizeLineStyle='solid',
    contrastEffectSizeLineColor='black',

    contrastYlim=None,
    contrastZeroLineStyle='solid', 
    contrastZeroLineColor='black', 
    connectPairs=True,

    effectSizeYLabel="Effect Size", 

    figsize=None, 
    floatContrast=True,
    floatSwarmSpacer=0.2,

    heightRatio=(1, 1),

    lineWidth=2,
    legend=True,
    legendFontSize=14,
    legendFontProps={},

    paired=False,
    pairedDeltaLineAlpha=0.3,
    pairedDeltaLineWidth=1.2,
    pal=None, 

    rawMarkerSize=8,
    rawMarkerType='o',
    reps=3000,
    
    showGroupCount=True,
    showCI=False, 
    showAllYAxes=False,
    showRawData=True,
    smoothboot=False, 
    statfunction=None, 

    summaryBar=False, 
    summaryBarColor='grey',
    summaryBarAlpha=0.25,

    summaryColour='black', 
    summaryLine=True, 
    summaryLineStyle='solid', 
    summaryLineWidth=0.25, 

    summaryMarkerSize=10, 
    summaryMarkerType='o',

    swarmShareY=True, 
    swarmYlim=None, 

    tickAngle=45,
    tickAlignment='right',

    violinOffset=0.375,
    violinWidth=0.2, 
    violinColor='k',

    xticksize=None,
    yticksize=None,

    **kwargs):

    '''Takes a pandas DataFrame and produces a contrast plot:
    either a Cummings hub-and-spoke plot or a Gardner-Altman contrast plot.
    Paired and unpaired options available.

    Keyword arguments:
        data: pandas DataFrame
            
        x: string
            column name containing categories to be plotted on the x-axis.

        y: string
            column name containing values to be plotted on the y-axis.

        idx: tuple
            flxible declaration of groupwise comparisons.

        idcol: string
            for paired plots.

        alpha: float
            alpha (transparency) of raw swarmed data points.
            
        axis_title_size=None
        ci=95
        contrastShareY=True
        contrastEffectSizeLineStyle='solid'
        contrastEffectSizeLineColor='black'
        contrastYlim=None
        contrastZeroLineStyle='solid'
        contrastZeroLineColor='black'
        effectSizeYLabel="Effect Size"
        figsize=None
        floatContrast=True
        floatSwarmSpacer=0.2
        heightRatio=(1,1)
        lineWidth=2
        legend=True
        legendFontSize=14
        legendFontProps={}
        paired=False
        pairedDeltaLineAlpha=0.3
        pairedDeltaLineWidth=1.2
        pal=None
        rawMarkerSize=8
        rawMarkerType='o'
        reps=3000
        showGroupCount=True
        showCI=False
        showAllYAxes=False
        showRawData=True
        smoothboot=False
        statfunction=None
        summaryBar=False
        summaryBarColor='grey'
        summaryBarAlpha=0.25
        summaryColour='black'
        summaryLine=True
        summaryLineStyle='solid'
        summaryLineWidth=0.25
        summaryMarkerSize=10
        summaryMarkerType='o'
        swarmShareY=True
        swarmYlim=None
        tickAngle=45
        tickAlignment='right'
        violinOffset=0.375
        violinWidth=0.2
        violinColor='k'
        xticksize=None
        yticksize=None

    Returns:
        An matplotlib Figure.
        Organization of figure Axes.
    '''

    # Check that `data` is a pandas dataframe
    if 'DataFrame' not in str(type(data)):
        raise TypeError("The object passed to the command is not not a pandas DataFrame.\
         Please convert it to a pandas DataFrame.")

    # make sure that at least x, y, and idx are specified.
    if x is None and y is None and idx is None:
        raise ValueError('You need to specify `x` and `y`, or `idx`. Neither has been specifed.')

    if x is None:
        # if x is not specified, assume this is a 'wide' dataset, with each idx being the name of a column.
        datatype='wide'
        # Check that the idx are legit columns.
        all_idx=np.unique([element for tupl in idx for element in tupl])
        # # melt the data.
        # data=pd.melt(data,value_vars=all_idx)
        # x='variable'
        # y='value'
    else:
        # if x is specified, assume this is a 'long' dataset with each row corresponding to one datapoint.
        datatype='long'
        # make sure y is not none.
        if y is None:
            raise ValueError("`paired` is false, but no y-column given.")
        # Calculate Ns.
        counts=data.groupby(x)[y].count()

    # Get and set levels of data[x]
    if paired is True:
        violinWidth=0.1
        # # Calculate Ns--which should be simply the number of rows in data.
        # counts=len(data)
        # is idcol supplied?
        if idcol is None and datatype=='long':
            raise ValueError('`idcol` has not been supplied but a paired plot is desired; please specify the `idcol`.')
        if idx is not None:
            # check if multi-plot or not
            if all(isinstance(element, str) for element in idx):
                # check that every idx is a column name.
                idx_not_in_cols=[n
                for n in idx
                if n not in data[x].unique()]
                if len(idx_not_in_cols)!=0:
                    raise ValueError(str(idx_not_in_cols)+" cannot be found in the columns of `data`.")
                # data_wide_cols=[n for n in idx if n in data.columns]
                # if idx is supplied but not a multiplot (ie single list or tuple)
                if len(idx) != 2:
                    raise ValueError(idx+" does not have length 2.")
                else:
                    tuple_in=(tuple(idx, ),)
                widthratio=[1]
            elif all(isinstance(element, tuple) for element in idx):
                # if idx is supplied, and it is a list/tuple of tuples or lists, we have a multiplot!
                idx_not_in_cols=[n
                for tup in idx
                for n in tup
                if n not in data[x].unique()]
                if len(idx_not_in_cols)!=0:
                    raise ValueError(str(idx_not_in_cols)+" cannot be found in the column "+x)
                # data_wide_cols=[n for tup in idx for n in tup if n in data.columns]
                if ( any(len(element) != 2 for element in idx) ):
                    # If any of the tuples does not contain exactly 2 elements.
                    raise ValueError(element+" does not have length 2.")
                # Make sure the widthratio of the seperate multiplot corresponds to how 
                # many groups there are in each one.
                tuple_in=idx
                widthratio=[]
                for i in tuple_in:
                    widthratio.append(len(i))
        elif idx is None:
            raise ValueError('Please specify idx.')
        showRawData=False # Just show lines, do not show data.
        showCI=False # wait till I figure out how to plot this for sns.barplot.
        if datatype=='long':
            if idx is None:
                ## If `idx` is not specified, just take the FIRST TWO levels alphabetically.
                tuple_in=tuple(np.sort(np.unique(data[x]))[0:2],)
            # pivot the dataframe if it is long!
            data_pivot=data.pivot_table(index = idcol, columns = x, values = y)

    elif paired is False:
        if idx is None:
            widthratio=[1]
            tuple_in=( tuple(data[x].unique()) ,)
            if len(tuple_in[0])>2:
                floatContrast=False
        else:
            if all(isinstance(element, str) for element in idx):
                # if idx is supplied but not a multiplot (ie single list or tuple)
                # check all every idx specified can be found in data[x]
                idx_not_in_x=[n for n in idx 
                if n not in data[x].unique()]
                if len(idx_not_in_x)!=0:
                    raise ValueError(str(idx_not_in_x)+" cannot be found in the column "+x)
                tuple_in=(idx, )
                widthratio=[1]
                if len(idx)>2:
                    floatContrast=False
            elif all(isinstance(element, tuple) for element in idx):
                # if idx is supplied, and it is a list/tuple of tuples or lists, we have a multiplot!
                idx_not_in_x=[n
                for tup in idx
                for n in tup
                if n not in data[x].unique()]
                if len(idx_not_in_x)!=0:
                    raise ValueError(str(idx_not_in_x)+" cannot be found in the column "+x)
                tuple_in=idx

                if ( any(len(element)>2 for element in tuple_in) ):
                    # if any of the tuples in idx has more than 2 groups, we turn set floatContrast as False.
                    floatContrast=False
                # Make sure the widthratio of the seperate multiplot corresponds to how 
                # many groups there are in each one.
                widthratio=[]
                for i in tuple_in:
                    widthratio.append(len(i))
            else:
                raise TypeError("The object passed to `idx` consists of a mixture of single strings and tuples. \
                    Please make sure that `idx` is either a tuple of column names, or a tuple of tuples, for plotting.")

    # Ensure summaryLine and summaryBar are not displayed together.
    if summaryLine is True and summaryBar is True:
        summaryBar=True
        summaryLine=False
    # Turn off summary line if floatContrast is true
    if floatContrast:
        summaryLine=False
    # initialise statfunction
    if statfunction == None:
        statfunction=np.mean
    # Create list to collect all the contrast DataFrames generated.
    contrastList=list()
    contrastListNames=list()

    # Setting color palette for plotting.
    if pal is None:
        if 'hue' in kwargs:
            colorCol=kwargs['hue']
            if colorCol not in data.columns:
                raise ValueError(colorCol+' is not a column name.')
            colGrps=data[colorCol].unique()#.tolist()
            plotPal=dict( zip( colGrps, sns.color_palette(n_colors=len(colGrps)) ) )
        else:
            if datatype=='long':
                colGrps=data[x].unique()#.tolist()
                plotPal=dict( zip( colGrps, sns.color_palette(n_colors=len(colGrps)) ) )
            if datatype=='wide':
                plotPal=np.repeat('k',len(data))
    else:
        if datatype=='long':
            plotPal=pal
        if datatype=='wide':
            plotPal=list(map(lambda x:pal[x], data[hue]))

    if swarmYlim is None:
        # get range of _selected groups_.
        # u = list()
        # for t in tuple_in:
        #     for i in np.unique(t):
        #         u.append(i)
        # u = np.unique(u)
        u=np.unique([element for tupl in tuple_in for element in tupl])
        if datatype=='long':
            tempdat=data[data[x].isin(u)]
            swarm_ylim=np.array([np.min(tempdat[y]), np.max(tempdat[y])])
        if datatype=='wide':
            allMin=list()
            allMax=list()
            for col in u:
                allMin.append(np.min(data[col]))
                allMax.append(np.max(data[col]))
            swarm_ylim=np.array( [np.min(allMin),np.max(allMax)] )
        swarm_ylim=np.round(swarm_ylim)
    else:
        swarm_ylim=np.array([swarmYlim[0],swarmYlim[1]])

    if summaryBar is True:
        lims=swarm_ylim
        # check that 0 lies within the desired limits.
        # if not, extend (upper or lower) limit to zero.
        if 0 not in range( int(round(lims[0])),int(round(lims[1])) ): # turn swarm_ylim to integer range.
            # check if all negative:.
            if lims[0]<0. and lims[1]<0.:
                swarm_ylim=np.array([np.min(lims),0.])
            # check if all positive.
            elif lims[0]>0. and lims[1]>0.:
                swarm_ylim=np.array([0.,np.max(lims)])

    if contrastYlim is not None:
        contrastYlim=np.array([contrastYlim[0],contrastYlim[1]])

    # plot params
    if axis_title_size is None:
        axis_title_size=27
    if yticksize is None:
        yticksize=22
    if xticksize is None:
        xticksize=22

    # Set clean style
    sns.set(style='ticks')

    axisTitleParams={'labelsize' : axis_title_size}
    xtickParams={'labelsize' : xticksize}
    ytickParams={'labelsize' : yticksize}
    svgParams={'fonttype' : 'none'}

    rc('axes', **axisTitleParams)
    rc('xtick', **xtickParams)
    rc('ytick', **ytickParams)
    rc('svg', **svgParams) 

    if figsize is None:
        if len(tuple_in)>2:
            figsize=(12,(12/np.sqrt(2)))
        else:
            figsize=(8,(8/np.sqrt(2)))
    
    # calculate CI.
    if ci<0 or ci>100:
        raise ValueError('ci should be between 0 and 100.')
    alpha_level=(100.-ci)/100.

    # Initialise figure, taking into account desired figsize.
    fig=plt.figure(figsize=figsize)

    # Initialise GridSpec based on `tuple_in` shape.
    gsMain=gridspec.GridSpec( 
        1, np.shape(tuple_in)[0], 
         # 1 row; columns based on number of tuples in tuple.
         width_ratios=widthratio,
         wspace=0 )

    for gsIdx, current_tuple in enumerate(tuple_in):
        #### FOR EACH TUPLE IN IDX
        if datatype=='long':
            plotdat=data[data[x].isin(current_tuple)]
            plotdat[x]=plotdat[x].astype("category")
            plotdat[x].cat.set_categories(
                current_tuple,
                ordered=True,
                inplace=True)
            plotdat.sort_values(by=[x])
            # # Drop all nans. 
            # plotdat.dropna(inplace=True)
            summaries=plotdat.groupby(x)[y].apply(statfunction)
        if datatype=='wide':
            plotdat=data[list(current_tuple)]
            summaries=statfunction(plotdat)
            plotdat=pd.melt(plotdat) ##### NOW I HAVE MELTED THE WIDE DATA.
            
        if floatContrast is True:
            # Use fig.add_subplot instead of plt.Subplot.
            ax_raw=fig.add_subplot(gsMain[gsIdx],
                frame_on=False)
            ax_contrast=ax_raw.twinx()
        else:
        # Create subGridSpec with 2 rows and 1 column.
            subGridSpec=gridspec.GridSpecFromSubplotSpec(2, 1,
                subplot_spec=gsMain[gsIdx],
                wspace=0)
            # Use plt.Subplot instead of fig.add_subplot
            ax_raw=plt.Subplot(fig,
                subGridSpec[0, 0],
                frame_on=False)
            ax_contrast=plt.Subplot(fig,
                subGridSpec[1, 0],
                sharex=ax_raw,
                frame_on=False)
        # Calculate the boostrapped contrast
        bscontrast=list()
        if paired is False:
            tempplotdat=plotdat[[x,y]] # only select the columns used for x and y plotting.
            for i in range (1, len(current_tuple)):
                # Note that you start from one. No need to do auto-contrast!
                # if datatype=='long':aas
                    tempbs=bootstrap_contrast(
                        data=tempplotdat.dropna(), 
                        x=x,
                        y=y,
                        idx=[current_tuple[0], current_tuple[i]],
                        statfunction=statfunction,
                        smoothboot=smoothboot,
                        alpha_level=alpha_level,
                        reps=reps)
                    bscontrast.append(tempbs)
                    contrastList.append(tempbs)
                    contrastListNames.append(current_tuple[i]+' vs. '+current_tuple[0])

        #### PLOT RAW DATA.
        ax_raw.set_ylim(swarm_ylim)
        # ax_raw.yaxis.set_major_locator(MaxNLocator(n_bins='auto'))
        # ax_raw.yaxis.set_major_locator(LinearLocator())
        if paired is False and showRawData is True:
            # Seaborn swarmplot doc says to set custom ylims first.
            sw=sns.swarmplot(
                data=plotdat, 
                x=x, y=y, 
                order=current_tuple, 
                ax=ax_raw, 
                alpha=alpha, 
                palette=plotPal,
                size=rawMarkerSize,
                marker=rawMarkerType,
                **kwargs)

            if floatContrast:
                # Get horizontal offset values.
                maxXBefore=max(sw.collections[0].get_offsets().T[0])
                minXAfter=min(sw.collections[1].get_offsets().T[0])
                xposAfter=maxXBefore+floatSwarmSpacer
                xAfterShift=minXAfter-xposAfter
                # shift the (second) swarmplot
                offsetSwarmX(sw.collections[1], -xAfterShift)
                # shift the tick.
                ax_raw.set_xticks([0.,1-xAfterShift])

        elif paired is True:
            if showRawData is True:
                sw=sns.swarmplot(data=plotdat, 
                    x=x, y=y, 
                    order=current_tuple, 
                    ax=ax_raw, 
                    alpha=alpha, 
                    palette=plotPal,
                    size=rawMarkerSize,
                    marker=rawMarkerType,
                **kwargs)
            if connectPairs is True:
                # Produce paired plot with lines.
                before=plotdat[plotdat[x]==current_tuple[0]][y].tolist()
                after=plotdat[plotdat[x]==current_tuple[1]][y].tolist()
                linedf=pd.DataFrame(
                    {'before':before,
                    'after':after}
                    )
                # to get color, need to loop thru each line and plot individually.
                for ii in range(0,len(linedf)):
                    ax_raw.plot( [0,0.25], [ linedf.loc[ii,'before'],
                                            linedf.loc[ii,'after'] ],
                                linestyle='solid',
                                linewidth=pairedDeltaLineWidth,
                                color=plotPal[current_tuple[0]],
                                alpha=pairedDeltaLineAlpha,
                               )
                ax_raw.set_xlim(-0.25,0.5)
                ax_raw.set_xticks([0,0.25])
                ax_raw.set_xticklabels([current_tuple[0],current_tuple[1]])

        # if swarmYlim is None:
        #     # if swarmYlim was not specified, tweak the y-axis 
        #     # to show all the data without losing ticks and range.
        #     ## Get all yticks.
        #     axxYTicks=ax_raw.yaxis.get_majorticklocs()
        #     ## Get ytick interval.
        #     YTickInterval=axxYTicks[1]-axxYTicks[0]
        #     ## Get current ylim
        #     currentYlim=ax_raw.get_ylim()
        #     ## Extend ylim by adding a fifth of the tick interval as spacing at both ends.
        #     ax_raw.set_ylim(
        #         currentYlim[0]-(YTickInterval/5),
        #         currentYlim[1]+(YTickInterval/5)
        #         )
        #     ax_raw.yaxis.set_major_locator(MaxNLocator(nbins='auto'))
        # ax_raw.yaxis.set_major_locator(MaxNLocator(nbins='auto'))
        # ax_raw.yaxis.set_major_locator(LinearLocator())

        if summaryBar is True:
            if paired is False:
                bar_raw=sns.barplot(
                    x=summaries.index.tolist(),
                    y=summaries.values,
                    facecolor=summaryBarColor,
                    ax=ax_raw,
                    alpha=summaryBarAlpha)
                if floatContrast is True:
                    maxSwarmSpan=2/10.
                    xlocs=list()
                    for i, bar in enumerate(bar_raw.patches):
                        x_width=bar.get_x()
                        width=bar.get_width()
                        centre=x_width + (width/2.)
                        if i == 0:
                            bar.set_x(centre-maxSwarmSpan/2.)
                            xlocs.append(centre)
                        else:
                            bar.set_x(centre-xAfterShift-maxSwarmSpan/2.)
                            xlocs.append(centre-xAfterShift)
                        bar.set_width(maxSwarmSpan)
                    ax_raw.set_xticks(xlocs) # make sure xticklocs match the barplot.
                elif floatContrast is False:
                    maxSwarmSpan=4/10.
                    xpos=ax_raw.xaxis.get_majorticklocs()
                    for i, bar in enumerate(bar_raw.patches):
                        bar.set_x(xpos[i]-maxSwarmSpan/2.)
                        bar.set_width(maxSwarmSpan)
            else:
                # if paired is true
                ax_raw.bar([0,0.25], 
                    [ statfunction(plotdat[current_tuple[0]]),
                    statfunction(plotdat[current_tuple[1]]) ],
                    color=summaryBarColor,
                    alpha=0.5,
                    width=0.05)
                ## Draw zero reference line.
                ax_raw.add_artist(Line2D(
                    (ax_raw.xaxis.get_view_interval()[0],
                     ax_raw.xaxis.get_view_interval()[1]),
                    (0,0),
                    color='k', linewidth=1.25)
                                 )

        if summaryLine is True:
            if paired is True:
                xdelta=0
            else:
                xdelta=summaryLineWidth
            for i, m in enumerate(summaries):
                ax_raw.plot(
                    (i-xdelta, 
                    i+xdelta), # x-coordinates
                    (m, m),
                    color=summaryColour, 
                    linestyle=summaryLineStyle)

        if showCI is True:
                sns.barplot(
                    data=plotdat, 
                    x=x, y=y, 
                    ax=ax_raw, 
                    alpha=0, ci=95)

        ax_raw.set_xlabel("")
        if floatContrast is False:
            fig.add_subplot(ax_raw)

        #### PLOT CONTRAST DATA.
        if len(current_tuple)==2:
            if paired is False:
                # Plot the CIs on the contrast axes.
                plotbootstrap(sw.collections[1],
                              bslist=tempbs,
                              ax=ax_contrast, 
                              violinWidth=violinWidth,
                              violinOffset=violinOffset,
                              markersize=summaryMarkerSize,
                              marker=summaryMarkerType,
                              offset=floatContrast,
                              color=violinColor,
                              linewidth=1)
            else:
                bootsDelta = bootstrap(
                    plotdat[current_tuple[1]]-plotdat[current_tuple[0]],
                    statfunction=statfunction,
                    smoothboot=smoothboot,
                    alpha_level=alpha_level,
                    reps=reps)
                contrastList.append(bootsDelta)
                contrastListNames.append(current_tuple[1]+' vs. '+current_tuple[0])
                summDelta = bootsDelta['summary']
                lowDelta = bootsDelta['bca_ci_low']
                highDelta = bootsDelta['bca_ci_high']

                if floatContrast:
                    xpos=0.375
                else:
                    xpos=0.25

                # Plot the summary measure.
                ax_contrast.plot(xpos, bootsDelta['summary'],
                         marker=summaryMarkerType,
                         markerfacecolor='k',
                         markersize=summaryMarkerSize,
                         alpha=0.75
                        )
                # Plot the CI.
                ax_contrast.plot([xpos, xpos],
                         [lowDelta, highDelta],
                         color='k',
                         alpha=0.75,
                         # linewidth=1,
                         linestyle='solid'
                        )
                
                # Plot the violin-plot.
                v = ax_contrast.violinplot(bootsDelta['stat_array'], [xpos], 
                                           widths = violinWidth, 
                                           showextrema = False, 
                                           showmeans = False)
                halfviolin(v, half = 'right', color = 'k')

            if floatContrast:
                # Set reference lines
                if paired is False:
                    ## First get leftmost limit of left reference group
                    xtemp, _=np.array(sw.collections[0].get_offsets()).T
                    leftxlim=xtemp.min()
                    ## Then get leftmost limit of right test group
                    xtemp, _=np.array(sw.collections[1].get_offsets()).T
                    rightxlim=xtemp.min()
                    ref=tempbs['summary']
                else:
                    leftxlim=0
                    rightxlim=0.25
                    ref=bootsDelta['summary']
                    ax_contrast.set_xlim(-0.25, 0.5) # does this work?

                ## zero line
                ax_contrast.hlines(0,                   # y-coordinates
                                leftxlim, 3.5,       # x-coordinates, start and end.
                                linestyle=contrastZeroLineStyle,
                                linewidth=1,
                                color=contrastZeroLineColor)

                ## effect size line
                ax_contrast.hlines(ref, 
                                rightxlim, 3.5,        # x-coordinates, start and end.
                                linestyle=contrastEffectSizeLineStyle,
                                linewidth=1,
                                color=contrastEffectSizeLineColor)


                if paired is False:
                    es=float(tempbs['summary'])
                    refSum=tempbs['statistic_ref']
                else:
                    es=float(bootsDelta['summary'])
                    refSum=statfunction(plotdat[current_tuple[0]])
                ## If the effect size is positive, shift the right axis up.
                if es>0:
                    rightmin=ax_raw.get_ylim()[0]-es
                    rightmax=ax_raw.get_ylim()[1]-es
                ## If the effect size is negative, shift the right axis down.
                elif es<0:
                    rightmin=ax_raw.get_ylim()[0]+es
                    rightmax=ax_raw.get_ylim()[1]+es
                ax_contrast.set_ylim(rightmin, rightmax)

                if gsIdx>0:
                    ax_contrast.set_ylabel('')
                align_yaxis(ax_raw, refSum, ax_contrast, 0.)

            else:
                # Set bottom axes ybounds
                if contrastYlim is not None:
                    ax_contrast.set_ylim(contrastYlim)

                if paired is False:
                    # Set xlims so everything is properly visible!
                    swarm_xbounds=ax_raw.get_xbound()
                    ax_contrast.set_xbound(swarm_xbounds[0] -(summaryLineWidth * 1.1), 
                        swarm_xbounds[1] + (summaryLineWidth * 1.1))
                else:
                    ax_contrast.set_xlim(-0.05,0.25+violinWidth)

        else:
            # Plot the CIs on the bottom axes.
            plotbootstrap_hubspoke(
                bslist=bscontrast,
                ax=ax_contrast,
                violinWidth=violinWidth,
                violinOffset=violinOffset,
                markersize=summaryMarkerSize,
                marker=summaryMarkerType,
                linewidth=lineWidth)

        if floatContrast is False:
            fig.add_subplot(ax_contrast)

        if gsIdx>0:
            ax_raw.set_ylabel('')
            ax_contrast.set_ylabel('')

    # Turn contrastList into a pandas DataFrame,
    contrastList=pd.DataFrame(contrastList).T
    contrastList.columns=contrastListNames

    # Get number of axes in figure for aesthetic tweaks.
    axesCount=len(fig.get_axes())
    for i in range(0, axesCount, 2):
        # Set new tick labels.
        # The tick labels belong to the SWARM axes
        # for both floating and non-floating plots.
        # This is because `sharex` was invoked.
        axx=fig.axes[i]
        newticklabs=list()
        for xticklab in axx.xaxis.get_ticklabels():
            t=xticklab.get_text()
            if paired:
                N=str(counts)
            else:
                N=str(counts.ix[t])

            if showGroupCount:
                newticklabs.append(t+' n='+N)
            else:
                newticklabs.append(t)
            axx.set_xticklabels(
                newticklabs,
                rotation=tickAngle,
                horizontalalignment=tickAlignment)

    ## Loop thru SWARM axes for aesthetic touchups.
    for i in range(0, axesCount, 2):
        axx=fig.axes[i]

        if floatContrast is False:
            axx.xaxis.set_visible(False)
            sns.despine(ax=axx, trim=True, bottom=False, left=False)
        else:
            sns.despine(ax=axx, trim=True, bottom=True, left=True)

        if i==0:
            drawback_y(axx)

        if i!=axesCount-2 and 'hue' in kwargs:
            # If this is not the final swarmplot, remove the hue legend.
            axx.legend().set_visible(False)

        if showAllYAxes is False:
            if i in range(2, axesCount):
                axx.yaxis.set_visible(False)
            else:
                # Draw back the lines for the relevant y-axes.
                # Not entirely sure why I have to do this.
                drawback_y(axx)
        else:
            drawback_y(axx)

        # Add zero reference line for swarmplots with bars.
        if summaryBar is True:
            axx.add_artist(Line2D(
                (axx.xaxis.get_view_interval()[0], 
                    axx.xaxis.get_view_interval()[1]), 
                (0,0),
                color='black', linewidth=0.75
                )
            )
        
        if legend is False:
            axx.legend().set_visible(False)
        else:
            if i==axesCount-2: # the last (rightmost) swarm axes.
                axx.legend(loc='top right',
                    bbox_to_anchor=(1.1,1.0),
                    fontsize=legendFontSize,
                    **legendFontProps)

    ## Loop thru the CONTRAST axes and perform aesthetic touch-ups.
    ## Get the y-limits:
    for j,i in enumerate(range(1, axesCount, 2)):
        axx=fig.get_axes()[i]

        if floatContrast is False:
            xleft, xright=axx.xaxis.get_view_interval()
            # Draw zero reference line.
            axx.hlines(y=0,
                xmin=xleft-1, 
                xmax=xright+1,
                linestyle=contrastZeroLineStyle,
                linewidth=0.75,
                color=contrastZeroLineColor)
            # reset view interval.
            axx.set_xlim(xleft, xright)

            if showAllYAxes is False:
                if i in range(2, axesCount):
                    axx.yaxis.set_visible(False)
                else:
                    # Draw back the lines for the relevant y-axes, only is axesCount is 2.
                    # Not entirely sure why I have to do this.
                    if axesCount==2:
                        drawback_y(axx)

            sns.despine(ax=axx, 
                top=True, right=True, 
                left=False, bottom=False, 
                trim=True)
            if j==0 and axesCount==2:
                # Draw back x-axis lines connecting ticks.
                drawback_x(axx)
            # Rotate tick labels.
            rotateTicks(axx,tickAngle,tickAlignment)

        elif floatContrast is True:
            if paired is True:
                # Get the bootstrapped contrast range.
                lower=np.min(contrastList.ix['stat_array',j])
                upper=np.max(contrastList.ix['stat_array',j])
            else:
                lower=np.min(contrastList.ix['diffarray',j])
                upper=np.max(contrastList.ix['diffarray',j])
            meandiff=contrastList.ix['summary', j]

            ## Make sure we have zero in the limits.
            if lower>0:
                lower=0.
            if upper<0:
                upper=0.

            ## Get the tick interval from the left y-axis.
            leftticks=fig.get_axes()[i-1].get_yticks()
            tickstep=leftticks[1] -leftticks[0]

            ## First re-draw of axis with new tick interval
            axx.yaxis.set_major_locator(MultipleLocator(base=tickstep))
            newticks1=axx.get_yticks()

            ## Obtain major ticks that comfortably encompass lower and upper.
            newticks2=list()
            for a,b in enumerate(newticks1):
                if (b >= lower and b <= upper):
                    # if the tick lies within upper and lower, take it.
                    newticks2.append(b)
            # if the meandiff falls outside of the newticks2 set, add a tick in the right direction.
            if np.max(newticks2)<meandiff:
                ind=np.where(newticks1 == np.max(newticks2))[0][0] # find out the max tick index in newticks1.
                newticks2.append( newticks1[ind+1] )
            elif meandiff<np.min(newticks2):
                ind=np.where(newticks1 == np.min(newticks2))[0][0] # find out the min tick index in newticks1.
                newticks2.append( newticks1[ind-1] )
            newticks2=np.array(newticks2)
            newticks2.sort()

            ## Second re-draw of axis to shrink it to desired limits.
            axx.yaxis.set_major_locator(FixedLocator(locs=newticks2))
            
            ## Despine the axes.
            sns.despine(ax=axx, trim=True, 
                bottom=False, right=False,
                left=True, top=True)

    # Normalize bottom/right Contrast axes to each other for Cummings hub-and-spoke plots.
    if (axesCount>2 and 
        contrastShareY is True and 
        floatContrast is False):

        # Set contrast ylim as max ticks of leftmost swarm axes.
        if contrastYlim is None:
            lower=list()
            upper=list()
            for c in range(0,len(contrastList.columns)):
                lower.append( np.min(contrastList.ix['bca_ci_low',c]) )
                upper.append( np.max(contrastList.ix['bca_ci_high',c]) )
            lower=np.min(lower)
            upper=np.max(upper)
        else:
            lower=contrastYlim[0]
            upper=contrastYlim[1]

        normalizeContrastY(fig, 
            contrast_ylim = contrastYlim, 
            show_all_yaxes = showAllYAxes)

    # Zero gaps between plots on the same row, if floatContrast is False
    if (floatContrast is False and showAllYAxes is False):
        gsMain.update(wspace=0.)

    else:    
        # Tight Layout!
        gsMain.tight_layout(fig)
    
    # And we're all done.
    rcdefaults() # restore matplotlib defaults.
    sns.set() # restore seaborn defaults.
    return fig, contrastList