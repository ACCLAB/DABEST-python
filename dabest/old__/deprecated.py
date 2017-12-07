def contrastplot_old(data, x, y, idx = None, 
    
    alpha = 0.75, 
    axis_title_size = None,

    barWidth = 5,

    contrastShareY = True,
    contrastEffectSizeLineStyle = 'solid',
    contrastEffectSizeLineColor = 'black',
    contrastYlim = None,
    contrastZeroLineStyle = 'solid', 
    contrastZeroLineColor = 'black', 

    effectSizeYLabel = "Effect Size", 

    figsize = None, 
    floatContrast = True,
    floatSwarmSpacer = 0.2,

    heightRatio = (1, 1),

    lineWidth = 2,
    legend = True,

    pal = None, 

    rawMarkerSize = 8,
    rawMarkerType = 'o',
    reps = 3000,
    
    showGroupCount=True,
    show95CI = False, 
    showAllYAxes = False,
    showRawData = True,
    smoothboot = False, 
    statfunction = None, 
    summaryBar = False, 
    summaryBarColor = 'grey',
    summaryColour = 'black', 
    summaryLine = True, 
    summaryLineStyle = 'solid', 
    summaryLineWidth = 0.25, 
    summaryMarkerSize = 10, 
    summaryMarkerType = 'o',
    swarmShareY = True, 
    swarmYlim = None, 

    tickAngle=45,
    tickAlignment='right',

    violinOffset = 0.375,
    violinWidth = 0.2, 

    xticksize = None,
    yticksize = None,

    **kwargs):
    '''Takes a pandas dataframe and produces a contrast plot:
    either a Cummings hub-and-spoke plot or a Gardner-Altman contrast plot.
    -----------------------------------------------------------------------
    Description of flags upcoming.'''

    sns.set_context(font_scale=1.5)

    # Check that `data` is a pandas dataframe
    if 'DataFrame' not in str(type(data)):
        raise TypeError("The object passed to the command is not not a pandas DataFrame.\
         Please convert it to a pandas DataFrame.")

    # Select only the columns for plotting and grouping. 
    # Also set palette based on total number of categories in data['x'] or data['hue_column']
    if 'hue' in kwargs:
        data = data[ [x,y,kwargs['hue']] ]
        u = kwargs['hue']
    else:
        data = data[[x,y]]
        u = x
    
    # Drop all nans. 
    data = data.dropna()

    # Set clean style
    sns.set(style = 'ticks')

    # plot params
    if axis_title_size is None:
        axis_title_size = 20
    if yticksize is None:
        yticksize = 16
    if xticksize is None:
        xticksize = 16

    axisTitleParams = {'labelsize' : axis_title_size}
    xtickParams = {'labelsize' : xticksize}
    ytickParams = {'labelsize' : yticksize}
    svgParams = {'fonttype' : 'none'}

    rc('axes', **axisTitleParams)
    rc('xtick', **xtickParams)
    rc('ytick', **ytickParams)
    rc('svg', **svgParams)
    rc('legend',**{'fontsize':16})
    rc('legend',**{'markerscale':1.75})

    # initialise statfunction
    if statfunction == None:
        statfunction = np.mean

    # Ensure summaryLine and summaryBar are not displayed together.
    if summaryLine is True and summaryBar is True:
        summaryBar = True
        summaryLine = False
    
    # Here we define the palette on all the levels of the 'x' column.
    # Thus, if the same pandas dataframe is re-used across different plots,
    # the color identity of each group will be maintained.
    if pal is None:
        plotPal = dict( zip( data[u].unique(), sns.color_palette(n_colors = len(data[u].unique())) ) 
                      )
    else:
        plotPal = pal
        
    # Get and set levels of data[x]    
    if idx is None:
        # No idx is given, so all groups are compared to the first one in the DataFrame column.
        levs_tuple = (tuple(data[x].unique()), )
        widthratio = [1]
        if len(data[x].unique()) > 2:
            floatContrast = False
    else:
        # check if multi-plot or not
        if all(isinstance(element, str) for element in idx):
            # if idx is supplied but not a multiplot (ie single list or tuple) 
            levs_tuple = (idx, )
            widthratio = [1]
            if len(idx) > 2:
                floatContrast = False
        elif all(isinstance(element, tuple) for element in idx):
            # if idx is supplied, and it is a list/tuple of tuples or lists, we have a multiplot!
            levs_tuple = idx
            if (any(len(element) > 2 for element in levs_tuple) and floatContrast == True):
                # if any of the tuples in idx has more than 2 groups, we turn set floatContrast as False.
                floatContrast = False
            # Make sure the widthratio of the seperate multiplot corresponds to how 
            # many groups there are in each one.
            widthratio = []
            for i in levs_tuple:
                widthratio.append(len(i))
    u = list()
    for t in levs_tuple:
        for i in np.unique(t):
            u.append(i)
    u = np.unique(u)

    tempdat = data.copy()
    # Make sure the 'x' column is a 'category' type.
    tempdat[x] = tempdat[x].astype("category")
    tempdat = tempdat[tempdat[x].isin(u)]
    # Filters out values that were not specified in idx.
    tempdat[x].cat.set_categories(u, ordered = True, inplace = True)
    if swarmYlim is None:
        swarm_ylim = np.array([np.min(tempdat[y]), np.max(tempdat[y])])
    else:
        swarm_ylim = np.array([swarmYlim[0],swarmYlim[1]])

    if contrastYlim is not None:
        contrastYlim = np.array([contrastYlim[0],contrastYlim[1]])

    # Expand the ylim in both directions.
    ## Find half of the range of swarm_ylim.
    swarmrange = swarm_ylim[1] - swarm_ylim[0]
    pad = 0.1 * swarmrange
    x2 = np.array([swarm_ylim[0]-pad, swarm_ylim[1]+pad])
    swarm_ylim = x2
    
    # Create list to collect all the contrast DataFrames generated.
    contrastList = list()
    contrastListNames = list()
    
    if figsize is None:
        if len(levs_tuple) > 2:
            figsize = (12,(12/np.sqrt(2)))
        else:
            figsize = (8,(8/np.sqrt(2)))

    barWidth=barWidth/1000 # Not sure why have to reduce the barwidth by this much! 

    if showRawData is True:
        maxSwarmSpan = 0.25
    else:
        maxSwarmSpan = barWidth         
        
    # Initialise figure, taking into account desired figsize.
    fig = plt.figure(figsize = figsize)
    
    # Initialise GridSpec based on `levs_tuple` shape.
    gsMain = gridspec.GridSpec( 1, np.shape(levs_tuple)[0], # 1 row; columns based on number of tuples in tuple.
                               width_ratios = widthratio,
                               wspace=0 ) 
    
    for gsIdx, levs in enumerate(levs_tuple):
        # Create temp copy of the data for plotting!
        plotdat = data.copy()
        
        # Make sure the 'x' column is a 'category' type.
        plotdat[x] = plotdat[x].astype("category")
        plotdat = plotdat[plotdat[x].isin(levs)]
        plotdat[x].cat.set_categories(levs, ordered = True, inplace = True)
        
        # then order according to `levs`!
        plotdat.sort_values(by = [x])

        # Calculate summaries.
        # means=plotdat.groupby([x], sort=True).mean()[y]
        summaries=plotdat.groupby([x], sort=True)[y].apply(statfunction)

        if len(levs) == 2:            
            # Calculate bootstrap contrast. 
            tempbs = bootstrap_contrast(data = data, 
                                        x = x, 
                                        y = y,
                                        idx = levs, 
                                        statfunction = statfunction, 
                                        smoothboot = smoothboot,
                                        reps = reps)
            
            contrastListNames.append( str(levs[1]) + " v.s " + str(levs[0]) )
            contrastList.append(tempbs)

            if floatContrast is True:
                ax_left = fig.add_subplot(gsMain[gsIdx], frame_on = False) 
                # Use fig.add_subplot instead of plt.Subplot
                
                if showRawData is True:
                    # Plot the raw data as a swarmplot.
                    sw = sns.swarmplot(data = plotdat, x = x, y = y, 
                                      order = levs, ax = ax_left, 
                                      alpha = alpha, palette = plotPal,
                                      size = rawMarkerSize,
                                      marker = rawMarkerType,
                                      **kwargs)
                    sw.set_ylim(swarm_ylim)
                
                maxXBefore = max(sw.collections[0].get_offsets().T[0])
                minXAfter = min(sw.collections[1].get_offsets().T[0])
                xposAfter = maxXBefore + floatSwarmSpacer
                xAfterShift = minXAfter - xposAfter
                offsetSwarmX(sw.collections[1], -xAfterShift)

                if summaryBar is True:
                    bar_raw = sns.barplot(x = summaries.index.tolist(),
                     y = summaries.values, 
                     facecolor = summaryBarColor,
                     ax = ax_left,
                     alpha = 0.25)
                    ## get swarm with largest span, set as max width of each barplot.
                    for i, bar in enumerate(bar_raw.patches):
                        x_width = bar.get_x()
                        width = bar.get_width()
                        centre = x_width + width/2.
                        if i == 0:
                            bar.set_x(centre - maxSwarmSpan/2.)
                        else:
                            bar.set_x(centre - xAfterShift - maxSwarmSpan/2.)
                        bar.set_width(maxSwarmSpan)
                
                ## Set the ticks locations for ax_left.
                # axLeftLab = ax_left.get_xaxis().get_ticklabels
                ax_left.get_xaxis().set_ticks((0, xposAfter))
                firstTick=ax_left.get_xaxis().get_ticklabels()[0].get_text()
                secondTick=ax_left.get_xaxis().get_ticklabels()[1].get_text()
                ## Set the tick labels!
                ax_left.set_xticklabels([firstTick,#+' n='+count[firstTick],
                                         secondTick],#+' n='+count[secondTick]],
                                       rotation = tickAngle,
                                       horizontalalignment = tickAlignment)
                ## Remove left axes x-axis title.
                ax_left.set_xlabel("")

                # Set up floating axis on right.
                ax_right = ax_left.twinx()

                # Then plot the bootstrap
                # We should only be looking at sw.collections[1],
                # as contrast plots will only be floating in this condition.
                plotbootstrap(sw.collections[1],
                              bslist = tempbs, 
                              ax = ax_right,
                              violinWidth = violinWidth, 
                              violinOffset = violinOffset,
                              markersize = summaryMarkerSize,
                              marker = summaryMarkerType,
                              color = 'k', 
                              linewidth = 2)

                # Set reference lines
                ## First get leftmost limit of left reference group
                xtemp, _ = np.array(sw.collections[0].get_offsets()).T
                leftxlim = xtemp.min()
                ## Then get leftmost limit of right test group
                xtemp, _ = np.array(sw.collections[1].get_offsets()).T
                rightxlim = xtemp.min()

                ## zero line
                ax_right.hlines(0,                   # y-coordinates
                                leftxlim, 3.5,       # x-coordinates, start and end.
                                linestyle = contrastZeroLineStyle,
                                linewidth = 0.75,
                                color = contrastZeroLineColor)

                ## effect size line
                ax_right.hlines(tempbs['summary'], 
                                rightxlim, 3.5,        # x-coordinates, start and end.
                                linestyle = contrastEffectSizeLineStyle,
                                linewidth = 0.75,
                                color = contrastEffectSizeLineColor)

                
                ## If the effect size is positive, shift the right axis up.
                if float(tempbs['summary']) > 0:
                    rightmin = ax_left.get_ylim()[0] - float(tempbs['summary'])
                    rightmax = ax_left.get_ylim()[1] - float(tempbs['summary'])
                ## If the effect size is negative, shift the right axis down.
                elif float(tempbs['summary']) < 0:
                    rightmin = ax_left.get_ylim()[0] + float(tempbs['summary'])
                    rightmax = ax_left.get_ylim()[1] + float(tempbs['summary'])

                ax_right.set_ylim(rightmin, rightmax)

                if legend is True:
                    ax_left.legend(loc='center left', bbox_to_anchor=(1.1, 1))
                elif legend is False:
                    ax_left.legend().set_visible(False)
                    
                if gsIdx > 0:
                    ax_right.set_ylabel('')

                align_yaxis(ax_left, tempbs['statistic_ref'], ax_right, 0.)

            elif floatContrast is False:
                # Create subGridSpec with 2 rows and 1 column.
                gsSubGridSpec = gridspec.GridSpecFromSubplotSpec(2, 1, 
                                                                 subplot_spec = gsMain[gsIdx],
                                                                 wspace=0)
                ax_top = plt.Subplot(fig, gsSubGridSpec[0, 0], frame_on = False)

                if show95CI is True:
                    sns.barplot(data = plotdat, x = x, y = y, ax = ax_top, alpha = 0, ci = 95)

                # Plot the swarmplot on the top axes.
                sw = sns.swarmplot(data = plotdat, x = x, y = y, 
                                  order = levs, ax = ax_top, 
                                  alpha = alpha, palette = plotPal,
                                  size = rawMarkerSize,
                                  marker = rawMarkerType,
                                  **kwargs)
                sw.set_ylim(swarm_ylim)

                # Then plot the summary lines.
                if summaryLine is True:
                    for i, m in enumerate(summaries):
                        ax_top.plot((i - summaryLineWidth, i + summaryLineWidth),           # x-coordinates
                                    (m, m),                                                 # y-coordinates
                                    color = summaryColour, linestyle = summaryLineStyle)
                elif summaryBar is True:
                    sns.barplot(x = summaries.index.tolist(), 
                        y = summaries.values, 
                        facecolor = summaryBarColor, 
                        ci=0,
                        ax = ax_top, 
                        alpha = 0.25)
                    
                if legend is True:
                    ax_top.legend(loc='center left', bbox_to_anchor=(1.1, 1))
                elif legend is False:
                    ax_top.legend().set_visible(False)
                    
                fig.add_subplot(ax_top)
                ax_top.set_xlabel('')
                
                # Initialise bottom axes
                ax_bottom = plt.Subplot(fig, gsSubGridSpec[1, 0], sharex = ax_top, frame_on = False)

                # Plot the CIs on the bottom axes.
                plotbootstrap(sw.collections[1],
                              bslist = tempbs,
                              ax = ax_bottom, 
                              violinWidth = violinWidth,
                              markersize = summaryMarkerSize,
                              marker = summaryMarkerType,
                              offset = False,
                              violinOffset = 0,
                              linewidth = 2)

                # Set bottom axes ybounds
                if contrastYlim is not None:
                #     ax_bottom.set_ylim( tempbs['diffarray'].min(), tempbs['diffarray'].max() )
                # else:
                    ax_bottom.set_ylim(contrastYlim)
                
                # Set xlims so everything is properly visible!
                swarm_xbounds = ax_top.get_xbound()
                ax_bottom.set_xbound(swarm_xbounds[0] - (summaryLineWidth * 1.1), 
                                     swarm_xbounds[1] + (summaryLineWidth * 1.1))
                
                fig.add_subplot(ax_bottom)

                # Hide the labels for non leftmost plots.
                if gsIdx > 0:
                    ax_top.set_ylabel('')
                    ax_bottom.set_ylabel('')
                    
        elif len(levs) > 2:
            bscontrast = list()
            # Create subGridSpec with 2 rows and 1 column.
            gsSubGridSpec = gridspec.GridSpecFromSubplotSpec(2, 1, 
                                                     subplot_spec = gsMain[gsIdx],
                                                     wspace=0)
                        
            # Calculate the hub-and-spoke bootstrap contrast.
            for i in range (1, len(levs)): # Note that you start from one. No need to do auto-contrast!
                tempbs = bootstrap_contrast(data = data,
                                            x = x, 
                                            y = y, 
                                            idx = [levs[0], levs[i]],
                                            statfunction = statfunction, 
                                            smoothboot = smoothboot,
                                            reps = reps)
                bscontrast.append(tempbs)
                contrastList.append(tempbs)
                contrastListNames.append(levs[i] + ' vs. ' + levs[0])

            # Initialize the top swarmplot axes.
            ax_top = plt.Subplot(fig, gsSubGridSpec[0, 0], frame_on = False)
            
            if show95CI is True:
                sns.barplot(data = plotdat, x = x, y = y, ax = ax_top, alpha = 0, ci = 95)

            sw = sns.swarmplot(data = plotdat, x = x, y = y, 
                              order = levs, ax = ax_top, 
                              alpha = alpha, palette = plotPal,
                              size = rawMarkerSize,
                              marker = rawMarkerType,
                              **kwargs)
            sw.set_ylim(swarm_ylim)

            # Then plot the summary lines.
            if summaryLine is True:
                for i, m in enumerate(summaries):
                    ax_top.plot((i - summaryLineWidth, i + summaryLineWidth),           # x-coordinates
                                (m, m),                                                 # y-coordinates
                                color = summaryColour, linestyle = summaryLineStyle)
            elif summaryBar is True:
                sns.barplot(x = summaries.index.tolist(), 
                    y = summaries.values, 
                    facecolor = summaryBarColor, 
                    ax = ax_top, 
                    alpha = 0.25)

            if legend is True:
                ax_top.legend(loc='center left', bbox_to_anchor=(1.1, 1))
            elif legend is False:
                ax_top.legend().set_visible(False)
            
            fig.add_subplot(ax_top)
            ax_top.set_xlabel('')

            # Initialise the bottom swarmplot axes.
            ax_bottom = plt.Subplot(fig, gsSubGridSpec[1, 0], sharex = ax_top, frame_on = False)

            # Plot the CIs on the bottom axes.
            plotbootstrap_hubspoke(bslist = bscontrast,
                                   ax = ax_bottom, 
                                   violinWidth = violinWidth,
                                   violinOffset = violinOffset,
                                   markersize = summaryMarkerSize,
                                   marker = summaryMarkerType,
                                   linewidth = lineWidth)
            # Set bottom axes ybounds
            if contrastYlim is not None:
                ax_bottom.set_ybound(contrastYlim)
            
            # Set xlims so everything is properly visible!
            swarm_xbounds = ax_top.get_xbound()
            ax_bottom.set_xbound(swarm_xbounds[0] - (summaryLineWidth * 1.1), 
                                 swarm_xbounds[1] + (summaryLineWidth * 1.1))
            
            # Label the bottom y-axis
            fig.add_subplot(ax_bottom)
            ax_bottom.set_ylabel(effectSizeYLabel)
            
            if gsIdx > 0:
                ax_top.set_ylabel('')
                ax_bottom.set_ylabel('')
            
    # Turn contrastList into a pandas DataFrame,
    contrastList = pd.DataFrame(contrastList).T
    contrastList.columns = contrastListNames

    axesCount=len(fig.get_axes())

    # Loop thru the CONTRAST axes and perform aesthetic touch-ups.
    for j,i in enumerate(range(1, axesCount, 2)):
        axx=fig.get_axes()[i]

        if floatContrast is False:
            xleft, xright=axx.xaxis.get_view_interval()
            # Draw zero reference line.
            axx.hlines(y = 0,
                xmin = xleft-1, 
                xmax = xright+1,
                linestyle = contrastZeroLineStyle,
                linewidth = 0.75,
                color = contrastZeroLineColor)
            # reset view interval.
            axx.set_xlim(xleft, xright)

            sns.despine(ax = axx, 
                top = True, right = True, 
                left = False, bottom = False, 
                trim = True)

            # Rotate tick labels.
            rotateTicks(axx,tickAngle,tickAlignment)

        else:
            # Re-draw the floating axis to the correct limits.
            ## Get the 'correct limits':
            lower = np.min( contrastList.ix['diffarray',j] )
            upper = np.max( contrastList.ix['diffarray',j] )
            meandiff = contrastList.ix['summary', j]

            ## Make sure we have zero in the limits.
            if lower > 0:
                lower = 0.
            if upper < 0:
                upper = 0.

            ## Get the tick interval from the left y-axis.
            leftticks = fig.get_axes()[i-1].get_yticks()
            tickstep = leftticks[1] - leftticks[0]

            ## First re-draw of axis with new tick interval
            axx.yaxis.set_major_locator(MultipleLocator(base = tickstep))
            newticks1 = axx.get_yticks()

            ## Obtain major ticks that comfortably encompass lower and upper.
            newticks2 = list()
            for a,b in enumerate(newticks1):
                if (b >= lower and b <= upper):
                    # if the tick lies within upper and lower, take it.
                    newticks2.append(b)
            # if the meandiff falls outside of the newticks2 set, add a tick in the right direction.
            if np.max(newticks2) < meandiff:
                ind = np.where(newticks1 == np.max(newticks2))[0][0] # find out the max tick index in newticks1.
                newticks2.append( newticks1[ind+1] )
            elif meandiff < np.min(newticks2):
                ind = np.where(newticks1 == np.min(newticks2))[0][0] # find out the min tick index in newticks1.
                newticks2.append( newticks1[ind-1] )
            newticks2 = np.array(newticks2)
            newticks2.sort()

            ## Second re-draw of axis to shrink it to desired limits.
            axx.yaxis.set_major_locator(FixedLocator(locs = newticks2))
            
            # ## Obtain minor ticks that fall within the major ticks.
            # majorticks = fig.get_axes()[i].yaxis.get_majorticklocs()
            # oldminorticks = fig.get_axes()[i].yaxis.get_minorticklocs()

            ## Despine, trim, and redraw the lines.
            sns.despine(ax = axx, trim = True, 
                bottom = False, right = False,
                left = True, top = True)

    # Now loop thru SWARM axes for aesthetic touchups.
    for i in range(0, axesCount, 2):
        axx=fig.get_axes()[i]

        if i != axesCount - 2 and 'hue' in kwargs:
            # If this is not the final swarmplot, remove the hue legend.
            axx.legend().set_visible(False)

        if floatContrast is True:
            sns.despine(ax = axx, trim = True, right = True)
        else:
            sns.despine(ax = axx, trim = True, bottom = True, right = True)
            axx.get_xaxis().set_visible(False)

        if (showAllYAxes is False and i in range( 2, len(fig.get_axes())) ):
            axx.get_yaxis().set_visible(showAllYAxes)
        else:
            # Draw back the lines for the relevant y-axes.
            drawback_y(axx)

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

    ########
    # Normalize bottom/right Contrast axes to each other for Cummings hub-and-spoke plots.
    if (axesCount > 2 and 
        contrastShareY is True and 
        floatContrast is False):

        # Set contrast ylim as max ticks of leftmost swarm axes.
        if contrastYlim is None:
          contrastYmin = fig.axes[1].yaxis.get_ticklocs()[0]
          contrastYmax = fig.axes[1].yaxis.get_ticklocs()[-1]

        normalizeContrastY(fig, 
            contrast_ylim = contrastYlim, 
            show_all_yaxes = showAllYAxes)

    if (axesCount==2 and 
        floatContrast is False):
        drawback_x(fig.get_axes()[1])
        drawback_y(fig.get_axes()[1])

    if swarmShareY is False:
        for i in range(0, axesCount, 2):
            drawback_y(fig.get_axes()[i])
                       
    if contrastShareY is False:
        for i in range(1, axesCount, 2):
            if floatContrast is True:
                sns.despine(ax = fig.get_axes()[i], 
                           top = True, right = False, left = True, bottom = True, 
                           trim = True)
            else:
                sns.despine(ax = fig.get_axes()[i], trim = True)

    # Zero gaps between plots on the same row, if floatContrast is False
    if (floatContrast is False and showAllYAxes is False):
        gsMain.update(wspace = 0.)

    else:    
        # Tight Layout!
        gsMain.tight_layout(fig)
    
    # And we're all done.
    # rcdefaults() # restore matplotlib defaults.
    # sns.set() # restore seaborn defaults.
    return fig, contrastList