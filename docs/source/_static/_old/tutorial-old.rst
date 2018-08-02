.. _Tutorial:
.. highlight:: python
  :linenothreshold: 2
  :dedent: 4

========
Tutorial
========
.. `Download <https://raw.githubusercontent.com/ACCLAB/DABEST-python-docs/master/ipynb/dabest_tutorial.ipynb>`_ this tutorial as a Jupyter notebook.

First, we load the libraries we will be using::

  import pandas as pd
  import numpy as np
  import dabest

Next, we create a dummy dataset to illustrate how ``dabest`` functions. In this dataset, each column corresponds to a group of observations, and each row is simply an index number referring to an observation.
This is known as a 'wide' dataset. See this `writeup <https://sejdemyr.github.io/r-tutorials/basics/wide-and-long/>`_ for more details. ::

  from scipy.stats import norm # Used in generation of populations.

  np.random.seed(9999) # Fix the seed so the results are replicable.
  pop_size = 10000 # Size of each population.
  Ns = 20 # The number of samples taken from each population

  # Create populations
  control_pop = norm.rvs(loc=3, scale=0.4, size=pop_size)
  pop1 = norm.rvs(loc=3.5, scale=0.5, size=pop_size)
  pop2 = norm.rvs(loc=2.5, scale=0.6, size=pop_size)
  pop3 = norm.rvs(loc=3, scale=0.75, size=pop_size)
  pop4 = norm.rvs(loc=3.5, scale=0.75, size=pop_size)
  pop5 = norm.rvs(loc=3.25, scale=0.4, size=pop_size)

  # Sample from the populations
  choice_kwargs = dict(size=Ns, replace=False)

  control = np.random.choice(control_pop, **choice_kwargs)
  g1 = np.random.choice(pop1, **choice_kwargs)
  g2 = np.random.choice(pop2, **choice_kwargs)
  g3 = np.random.choice(pop3, **choice_kwargs)
  g4 = np.random.choice(pop4, **choice_kwargs)
  g5 = np.random.choice(pop5, **choice_kwargs)

  # Add a `gender` column for coloring the data.
  females = np.repeat('Female', Ns/2).tolist()
  males = np.repeat('Male', Ns/2).tolist()
  gender = females + males

  # Combine samples and gender into a DataFrame.
  df = pd.DataFrame({'Control': control,
                     'Group 1': g1,
                     'Group 2': g2,
                     'Group 3': g3,
                     'Group 4': g4,
                     'Group 5': g5,
                     'Gender' : gender,
                    })

Gardner-Altman Independant Two-Group Estimation Plot
----------------------------------------------------

We can easily use ``dabest.plot()`` to create a **Gardner-Altman estimation plot** to plot the mean difference (and its 95% confidence interval) alongside the raw data::

  f, b = dabest.plot(df, idx=('Control','Group 1'))
.. image:: _images/f1.png


A few things to note:

- The ``dabest.plot()`` function will return **two objects**: a matplotlib ``Figure`` and a pandas ``DataFrame``. (In the Jupyter Notebook, with the ``magic`` command ``%matplotlib inline``, the figure should automatically appear.) In the above example, the Figure is assigned to the variable ``f1``, while the DataFrame is assigned to ``b1``.

- ``dabest.plot()`` will automatically drop any NaNs (aka empty cells) in the data.

- The *Ns* (appended to the group names in the xtick labels) indicate the number of datapoints being plotted, and used to calculate the contrasts.

- The pandas ``DataFrame`` returned by plot contains the pairwise comparisons made in the course of generating the plot, with confidence intervals (95% by default) and relevant *P* values. The DataFrame produced is shown below, and can be accessed by normal pandas commands.

.. image:: _images/dataframe_out.png
    :width: 900px
    :align: center

The default color palette will assign a color to each group in running sequence of the `tab10 <<https://matplotlib.org/mpl_examples/color/colormaps_reference_03.png>`_ palette.

Gardner-Altman Paired Two-Group Estimation Plot
-----------------------------------------------

To create a **paired Gardner-Altman estimation plot**, between two measurements of the same sample, we use the ``paired`` keyword within the ``dabest.plot()`` function::

  f2, b2 = dabest.plot(df, idx=('Control','Group 1'),
                       paired=True)
.. image:: _images/f2.png

Gardner-Altman Mutiple Two-Group Estimation Plot
------------------------------------------------

The ``dabest.plot()`` function automatically tiles two or more two-group Gardner-Altman plots. This is designed to meet data visualization and presentation paradigms that are predominant in academic biomedical research.

This is done mainly through the ``idx`` option. You can indicate two or more tuples to create a seperate subplot for that contrast.

The effect sizes and confidence intervals for each two-group plot will be computed::

  f3, b3 = dabest.plot(df,
                     idx=(('Control','Group1'),
                          ('Group2','Group3'),
                          ('Group4','Group5')),
                     color_col='Gender')
.. image:: _images/f3.png

Cumming hub-and-spoke plot
--------------------------

A common experimental design seen in contemporary biomedical research is a shared-control, or 'hub-and-spoke' design. Two or more experimental groups are compared to a common control group.

A hub-and-spoke plot implements estimation statistics and aesthetics on such an experimental design.

If more than 2 columns/groups are indicated in a tuple passed to ``idx``, then ``dabest.plot()`` will produce a hub-and-spoke plot, where the first group in the tuple is considered the control group. The mean difference and confidence intervals of each subsequent group will be computed against the first control group::

  f4, b4 = dabest.plot(df,
                   idx=('Control', 'Group2', 'Group4'),
                   fig_size=(6,5),
                   color_col='Gender')
.. image:: _images/f4.png

In a Cumming plot, the bootstrapped effect size is shown on the lower panel for all comparisons. By default, a summary line is plotted for each group. The mean is indicated by the gap, and the standard deviation is plotted as lines flanking the gap.

One can display the median with the 25th and 75th percentiles (a Tufte-style boxplot) using the ``group_summaries`` keyword in ``dabest.plot()``::

  f5, b5 = dabest.plot(df,
                   idx=('Control', 'Group2', 'Group4'),
                   fig_size=(6,5),
                   color_col='Gender')
.. image:: _images/f5.png

Controlling aesthetics
----------------------

Below we run through ways of customizing various aesthetic features.

Changing the contrast y-limits::

  f6, b6 = dabest.plot(df,
                       idx=('Control','Group1','Group2'),
                       color_col='Gender',
                       contrast_ylim=(-2,2))
.. image:: _images/f6.png

Changing the swarmplot y-limits::

  f7, b7 = dabest.plot(df,
                      idx=('Control','Group1','Group2'),
                      color_col='Gender',
                      swarm_ylim=(-10,10))
.. image:: _images/f7.png

Changing the size of the dots in the swarmplot. This is done through the ``swarmplot_kwargs`` keyword in ``dabest.plot()``, which accepts a dictionary. You can pass any keywords that ``sns.swarmplot`` can accept::

  f8, b8 = dabest.plot(df,
                         idx=('Control','Group1','Group2'),
                         color_col='Gender',
                         swarmplot_kwargs={'size':10}
                        )
.. image:: _images/f8.png

Custom y-axis labels::

  f9, b9 = dabest.plot(df,
                     idx=('Control','Group1','Group2'),
                     color_col='Gender',
                     swarm_label='My Custom\nSwarm Label',
                     contrast_label='This is the\nEstimation plot'
                    )
.. image:: _images/f9.png

Applying a custom palette. This can be done in two ways.

First, we could pass a list (of `colors accepted <https://matplotlib.org/examples/color/named_colors.html>`_ by ``matplotlib``) to the ``custom_palette`` keyword::

  f10, b10 = dabest.plot(df,
                     idx=('Control','Group1','Group4'),
                     color_col='Gender',
                     custom_palette=['green', 'tomato']
                    )
.. image:: _images/f10.png

The second way is to pass a dictionary::

  f, b = dabest.plot(df,
                     idx=('Control','Group1','Group4'),
                     color_col='Gender',
                     custom_palette=dict(Male='green', Female='tomato')
                    )
.. image:: _images/f11.png

Custom y-axis labels for both swarm and contrast axes::

  f, b = dabest.plot(df,
                     idx=('Control','Group1','Group4'),
                     color_col='Gender',
                     swarm_label='my swarm',
                     contrast_label='The\nContrasts' # add line break.
                    )
.. image:: _images/f12.png

Working with 'melted' data
---------------------------

``dabest.plot()`` can also work with 'melted' or 'longform' data. This term is so used because each row will now correspond to a single datapoint, with one column carrying the value (value) and other columns carrying 'metadata' describing that datapoint.

For more details on wide vs long or 'melted' data, see  https://en.wikipedia.org/wiki/Wide_and_narrow_data.

To read more about melting a dataframe, see https://pandas.pydata.org/pandas-docs/stable/generated/pandas.melt.html.

To demonstrate this, we will use ``pandas`` to melt the dataframe we have been using this far::

  x = 'group'
  y = 'my_metric'
  color_col = 'Gender'
  value_cols = df.columns[:-1]

  df_melt=pd.melt(df.reset_index(),
                  id_vars=['index',color_col],
                  value_vars=value_cols,
                  value_name=y,
                  var_name=x)

If you are using a melted DataFrame, you will need to specify the x (containing the categorical group names) and y (containing the numerical values for plotting) columns::

  f, b = dabest.plot(df_melt,
                   x='group',
                   y='my_metric',
                   idx=('Control','Group 1'),
                   color_col='Gender',
                  )
.. image:: _images/f13.png
