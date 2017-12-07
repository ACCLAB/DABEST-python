from scipy.stats import ttest_ind, ttest_1samp, ttest_rel, mannwhitneyu, norm
from collections import OrderedDict
from numpy.random import randint
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, MaxNLocator, FixedLocator, AutoLocator, FormatStrFormatter
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
from .bootstrap_tools import ci, bootstrap, bootstrap_contrast, bootstrap_indexes, jackknife_indexes, getstatarray, bca
from .plot_bootstrap_tools import plotbootstrap, plotbootstrap_hubspoke, swarmsummary

def contrastplot_test(
    data, x, y, idx=None, 
    
    alpha=0.75, 
    axis_title_size=None,

    barWidth=5,

    contrastShareY=True,
    contrastEffectSizeLineStyle='solid',
    contrastEffectSizeLineColor='black',
    contrastYlim=None,
    contrastZeroLineStyle='solid', 
    contrastZeroLineColor='black', 

    effectSizeYLabel="Effect Size", 

    figsize=None, 
    floatContrast=True,
    floatSwarmSpacer=0.2,

    heightRatio=(1, 1),

    idcol=None,

    lineWidth=2,
    legend=True,
    legendFontSize=14,
    legendFontProps={},

    paired=False,
    pal=None, 

    rawMarkerSize=8,
    rawMarkerType='o',
    reps=3000,
    
    showGroupCount=True,
    show95CI=False, 
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

    '''Takes a pandas dataframe and produces a contrast plot:
    either a Cummings hub-and-spoke plot or a Gardner-Altman contrast plot.
    -----------------------------------------------------------------------
    Description of flags upcoming.'''

    # Check that `data` is a pandas dataframe
    if 'DataFrame' not in str(type(data)):
        raise TypeError("The object passed to the command is not not a pandas DataFrame.\
         Please convert it to a pandas DataFrame.")

    # Get and set levels of data[x]    
    if idx is None:
        widthratio=[1]
        allgrps=np.sort(data[x].unique())
        if paired:
            # If `idx` is not specified, just take the FIRST TWO levels alphabetically.
            tuple_in=tuple(allgrps[0:2],)
        else:
            # No idx is given, so all groups are compared to the first one in the DataFrame column.
            tuple_in=(tuple(allgrps), )
            if len(allgrps)>2:
                floatContrast=False

    else:
        if all(isinstance(element, str) for element in idx):
            # if idx is supplied but not a multiplot (ie single list or tuple) 
            tuple_in=(idx, )
            widthratio=[1]
            if len(idx)>2:
                floatContrast=False
        elif all(isinstance(element, tuple) for element in idx):
            # if idx is supplied, and it is a list/tuple of tuples or lists, we have a multiplot!
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
                Please make sure that `idx` is either a tuple of column names, or a tuple of tuples for plotting.")

    # initialise statfunction
    if statfunction == None:
        statfunction=np.mean

    # Create list to collect all the contrast DataFrames generated.
    contrastList=list()
    contrastListNames=list()
    # # Calculate the bootstraps according to idx.
    # for ix, current_tuple in enumerate(tuple_in):
    #     bscontrast=list()
    #     for i in range (1, len(current_tuple)):
    #     # Note that you start from one. No need to do auto-contrast!
    #         tempbs=bootstrap_contrast(
    #             data=data,
    #             x=x,
    #             y=y,
    #             idx=[current_tuple[0], current_tuple[i]],
    #             statfunction=statfunction,
    #             smoothboot=smoothboot,
    #             reps=reps)
    #         bscontrast.append(tempbs)
    #         contrastList.append(tempbs)
    #         contrastListNames.append(current_tuple[i]+' vs. '+current_tuple[0])

    # Setting color palette for plotting.
    if pal is None:
        if 'hue' in kwargs:
            colorCol=kwargs['hue']
            colGrps=data[colorCol].unique()
            nColors=len(colGrps)
        else:
            colorCol=x
            colGrps=data[x].unique()
            nColors=len([element for tupl in tuple_in for element in tupl])
        plotPal=dict( zip( colGrps, sns.color_palette(n_colors=nColors) ) )
    else:
        plotPal=pal

    # Ensure summaryLine and summaryBar are not displayed together.
    if summaryLine is True and summaryBar is True:
        summaryBar=True
        summaryLine=False
    # Turn off summary line if floatContrast is true
    if floatContrast:
        summaryLine=False

    if swarmYlim is None:
        # get range of _selected groups_.
        u = list()
        for t in idx:
            for i in np.unique(t):
                u.append(i)
        u = np.unique(u)
        tempdat=data[data[x].isin(u)]
        swarm_ylim=np.array([np.min(tempdat[y]), np.max(tempdat[y])])
    else:
        swarm_ylim=np.array([swarmYlim[0],swarmYlim[1]])

    if contrastYlim is not None:
        contrastYlim=np.array([contrastYlim[0],contrastYlim[1]])

    barWidth=barWidth/1000 # Not sure why have to reduce the barwidth by this much! 
    if showRawData is True:
        maxSwarmSpan=0.25
    else:
        maxSwarmSpan=barWidth

    # Expand the ylim in both directions.
    ## Find half of the range of swarm_ylim.
    swarmrange=swarm_ylim[1] -swarm_ylim[0]
    pad=0.1*swarmrange
    x2=np.array([swarm_ylim[0]-pad, swarm_ylim[1]+pad])
    swarm_ylim=x2

    # plot params
    if axis_title_size is None:
        axis_title_size=25
    if yticksize is None:
        yticksize=18
    if xticksize is None:
        xticksize=18

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
        plotdat=data[data[x].isin(current_tuple)]
        plotdat[x]=plotdat[x].astype("category")
        plotdat[x].cat.set_categories(
            current_tuple,
            ordered=True,
            inplace=True)
        plotdat.sort_values(by=[x])
        # Drop all nans. 
        plotdat=plotdat.dropna()

        # Calculate summaries.
        summaries=plotdat.groupby([x],sort=True)[y].apply(statfunction)

        if floatContrast is True:
            # Use fig.add_subplot instead of plt.Subplot
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
        for i in range (1, len(current_tuple)):
        # Note that you start from one. No need to do auto-contrast!
            tempbs=bootstrap_contrast(
                data=data,
                x=x,
                y=y,
                idx=[current_tuple[0], current_tuple[i]],
                statfunction=statfunction,
                smoothboot=smoothboot,
                reps=reps)
            bscontrast.append(tempbs)
            contrastList.append(tempbs)
            contrastListNames.append(current_tuple[i]+' vs. '+current_tuple[0])
        
        #### PLOT RAW DATA.
        if showRawData is True:
            # Seaborn swarmplot doc says to set custom ylims first.
            ax_raw.set_ylim(swarm_ylim)
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

        if summaryBar is True:
            bar_raw=sns.barplot(
                x=summaries.index.tolist(),
                y=summaries.values,
                facecolor=summaryBarColor,
                ax=ax_raw,
                alpha=summaryBarAlpha)
        
        if floatContrast:
            # Get horizontal offset values.
            maxXBefore=max(sw.collections[0].get_offsets().T[0])
            minXAfter=min(sw.collections[1].get_offsets().T[0])
            xposAfter=maxXBefore+floatSwarmSpacer
            xAfterShift=minXAfter-xposAfter
            # shift the swarmplots
            offsetSwarmX(sw.collections[1], -xAfterShift)

            ## get swarm with largest span, set as max width of each barplot.
            for i, bar in enumerate(bar_raw.patches):
                x_width=bar.get_x()
                width=bar.get_width()
                centre=x_width + (width/2.)
                if i == 0:
                    bar.set_x(centre-maxSwarmSpan/2.)
                else:
                    bar.set_x(centre-xAfterShift-maxSwarmSpan/2.)
                bar.set_width(maxSwarmSpan)

            ## Set the ticks locations for ax_raw.
            ax_raw.xaxis.set_ticks((0, xposAfter))
            firstTick=ax_raw.xaxis.get_ticklabels()[0].get_text()
            secondTick=ax_raw.xaxis.get_ticklabels()[1].get_text()
            ax_raw.set_xticklabels([firstTick,#+' n='+count[firstTick],
                                     secondTick],#+' n='+count[secondTick]],
                                   rotation=tickAngle,
                                   horizontalalignment=tickAlignment)

        if summaryLine is True:
            for i, m in enumerate(summaries):
                ax_raw.plot(
                    (i -summaryLineWidth, 
                    i + summaryLineWidth), # x-coordinates
                    (m, m),
                    color=summaryColour, 
                    linestyle=summaryLineStyle)

        if show95CI is True:
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
            if floatContrast:
                # Set reference lines
                ## First get leftmost limit of left reference group
                xtemp, _=np.array(sw.collections[0].get_offsets()).T
                leftxlim=xtemp.min()
                ## Then get leftmost limit of right test group
                xtemp, _=np.array(sw.collections[1].get_offsets()).T
                rightxlim=xtemp.min()

                ## zero line
                ax_contrast.hlines(0,                   # y-coordinates
                                leftxlim, 3.5,       # x-coordinates, start and end.
                                linestyle=contrastZeroLineStyle,
                                linewidth=0.75,
                                color=contrastZeroLineColor)

                ## effect size line
                ax_contrast.hlines(tempbs['summary'], 
                                rightxlim, 3.5,        # x-coordinates, start and end.
                                linestyle=contrastEffectSizeLineStyle,
                                linewidth=0.75,
                                color=contrastEffectSizeLineColor)

                
                ## If the effect size is positive, shift the right axis up.
                if float(tempbs['summary'])>0:
                    rightmin=ax_raw.get_ylim()[0] -float(tempbs['summary'])
                    rightmax=ax_raw.get_ylim()[1] -float(tempbs['summary'])
                ## If the effect size is negative, shift the right axis down.
                elif float(tempbs['summary'])<0:
                    rightmin=ax_raw.get_ylim()[0] + float(tempbs['summary'])
                    rightmax=ax_raw.get_ylim()[1] + float(tempbs['summary'])

                ax_contrast.set_ylim(rightmin, rightmax)

                    
                if gsIdx>0:
                    ax_contrast.set_ylabel('')

                align_yaxis(ax_raw, tempbs['statistic_ref'], ax_contrast, 0.)

            else:
                # Set bottom axes ybounds
                if contrastYlim is not None:
                    ax_contrast.set_ylim(contrastYlim)
                
                # Set xlims so everything is properly visible!
                swarm_xbounds=ax_raw.get_xbound()
                ax_contrast.set_xbound(swarm_xbounds[0] -(summaryLineWidth * 1.1), 
                    swarm_xbounds[1] + (summaryLineWidth * 1.1))

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
    
    ########
    axesCount=len(fig.get_axes())

    ## Loop thru SWARM axes for aesthetic touchups.
    for i in range(0, axesCount, 2):
        axx=fig.axes[i]

        if i!=axesCount-2 and 'hue' in kwargs:
            # If this is not the final swarmplot, remove the hue legend.
            axx.legend().set_visible(False)

        if floatContrast is False:
            axx.xaxis.set_visible(False)
            sns.despine(ax=axx, trim=True, bottom=False, left=False)
        else:
            sns.despine(ax=axx, trim=True, bottom=True, left=True)

        if showAllYAxes is False:
            if i in range(2, axesCount):
                axx.yaxis.set_visible(showAllYAxes)
            else:
                # Draw back the lines for the relevant y-axes.
                # Not entirely sure why I have to do this.
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

        # I don't know why the swarm axes controls the contrast axes ticks....
        if showGroupCount:
            count=data.groupby(x).count()[y]
            newticks=list()
            for ix, t in enumerate(axx.xaxis.get_ticklabels()):
                t_text=t.get_text()
                nt=t_text+' n='+str(count[t_text])
                newticks.append(nt)
            axx.xaxis.set_ticklabels(newticks)

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
            # # Draw back x-axis lines connecting ticks.
            # drawback_x(axx)

            if showAllYAxes is False:
                if i in range(2, axesCount):
                    axx.yaxis.set_visible(False)
                else:
                    # Draw back the lines for the relevant y-axes.
                    # Not entirely sure why I have to do this.
                    drawback_y(axx)

            sns.despine(ax=axx, 
                top=True, right=True, 
                left=False, bottom=False, 
                trim=True)

            # Rotate tick labels.
            rotateTicks(axx,tickAngle,tickAlignment)

        else:
            # Re-draw the floating axis to the correct limits.
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

    # if (axesCount==2 and 
    #     floatContrast is False):
    #     drawback_x(fig.get_axes()[1])
    #     drawback_y(fig.get_axes()[1])

    # if swarmShareY is False:
    #     for i in range(0, axesCount, 2):
    #         drawback_y(fig.get_axes()[i])
                       
    # if contrastShareY is False:
    #     for i in range(1, axesCount, 2):
    #         if floatContrast is True:
    #             sns.despine(ax=fig.get_axes()[i], 
    #                        top=True, right=False, left=True, bottom=True, 
    #                        trim=True)
    #         else:
    #             sns.despine(ax=fig.get_axes()[i], trim=True)

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
