.. _Tutorial:
.. highlight:: python
  :linenothreshold: 2

========
Tutorial
========

--------------
Load Libraries
--------------

.. code:: ipython3

    import numpy as np
    import pandas as pd

    import dabest
    %matplotlib inline

    print("We're using DABEST v{}".format(dabest.__version__))

.. parsed-literal::

    We're using DABEST v0.1.6

--------------------
Create dummy dataset
--------------------

Here, we create a dummy dataset to illustrate how ``dabest`` functions.
In this dataset, each column corresponds to a group of observations, and
each row is simply an index number referring to an observation.

This is known as a 'wide' dataset. See this
`writeup <https://sejdemyr.github.io/r-tutorials/basics/wide-and-long/>`__
for more details.

.. code:: ipython3

    from scipy.stats import norm # Used in generation of populations.

    np.random.seed(9999) # Fix the seed so the results are replicable.
    pop_size = 10000 # Size of each population.
    Ns = 20 # The number of samples taken from each population

    # Create populations
    pop1 = norm.rvs(loc=3, scale=0.4, size=pop_size)
    pop2 = norm.rvs(loc=3.5, scale=0.5, size=pop_size)
    pop3 = norm.rvs(loc=2.5, scale=0.6, size=pop_size)
    pop4 = norm.rvs(loc=3, scale=0.75, size=pop_size)
    pop5 = norm.rvs(loc=3.5, scale=0.75, size=pop_size)
    pop6 = norm.rvs(loc=3.25, scale=0.4, size=pop_size)


    # Sample from the populations
    sampling_kwargs = dict(size=Ns, replace=False)

    g1 = np.random.choice(pop1, **sampling_kwargs)
    g2 = np.random.choice(pop2, **sampling_kwargs)
    g3 = np.random.choice(pop3, **sampling_kwargs)
    g4 = np.random.choice(pop4, **sampling_kwargs)
    g5 = np.random.choice(pop5, **sampling_kwargs)
    g6 = np.random.choice(pop6, **sampling_kwargs)

    # Add a `gender` column for coloring the data.
    females = np.repeat('Female', Ns/2).tolist()
    males = np.repeat('Male', Ns/2).tolist()
    gender = females + males

    # Add an `id` column for paired data plotting.
    # More info below!
    id_col = pd.Series(range(1, Ns+1))

    # Combine samples and gender into a DataFrame.
    df = pd.DataFrame({'Control' : g1,
                       'Group 1' : g2,
                       'Group 2' : g3,
                       'Group 3' : g4,
                       'Group 4' : g5,
                       'Group 5' : g6,
                       'Gender'  : gender,
                       'ID'      : id_col
                      })

Note that we have 6 groups of observations, with an additional
non-numerical column indicating gender, and a column indicating the
identity of each observation.

.. code:: ipython3

    df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Control</th>
          <th>Group 1</th>
          <th>Group 2</th>
          <th>Group 3</th>
          <th>Group 4</th>
          <th>Group 5</th>
          <th>Gender</th>
          <th>ID</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>2.742313</td>
          <td>3.737751</td>
          <td>2.703766</td>
          <td>3.933794</td>
          <td>2.138557</td>
          <td>2.997997</td>
          <td>Female</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.681590</td>
          <td>2.954575</td>
          <td>3.158262</td>
          <td>2.983600</td>
          <td>3.377651</td>
          <td>3.699350</td>
          <td>Female</td>
          <td>2</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3.180724</td>
          <td>2.531722</td>
          <td>2.474184</td>
          <td>2.286611</td>
          <td>3.450214</td>
          <td>2.507875</td>
          <td>Female</td>
          <td>3</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.961873</td>
          <td>2.629912</td>
          <td>2.431826</td>
          <td>1.985591</td>
          <td>3.565215</td>
          <td>3.251389</td>
          <td>Female</td>
          <td>4</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2.867556</td>
          <td>3.335618</td>
          <td>2.454033</td>
          <td>3.887869</td>
          <td>2.841621</td>
          <td>3.513511</td>
          <td>Female</td>
          <td>5</td>
        </tr>
      </tbody>
    </table>
    </div>


---------------
Producing Plots
---------------

Independant two-group estimation plot
-------------------------------------

The simplest estimation plot can be generated with ``dabest.plot()``.
Supply the DataFrame; supply the two groups you want to compare in the
``idx`` argument as a tuple or list.

.. code:: ipython3

    fig1, results1 = dabest.plot(df, idx=('Control','Group 1'))



.. image:: _images/tutorial_8_0.png


The ``dabest.plot()`` function will return 2 objects: a matplotlib
``Figure`` and a pandas ``DataFrame``.

In the Jupyter Notebook, with ``%matplotlib inline``, the figure should
automatically appear.

``dabest.plot()`` will automatically drop any NaNs in the data. Note how
the Ns (appended to the group names in the xtick labels) indicate the
number of datapoints being plotted, and used to calculate the contrasts.

The pandas ``DataFrame`` returned by ``dabest.plot()`` contains the
statistics computed in the course of generating the plot, with
confidence intervals (95% by default) and relevant p-values.

.. code:: ipython3

    # prints out the DataFrame returned by `dabest.plot()`.
    results1




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>reference_group</th>
          <th>experimental_group</th>
          <th>stat_summary</th>
          <th>bca_ci_low</th>
          <th>bca_ci_high</th>
          <th>ci</th>
          <th>is_difference</th>
          <th>is_paired</th>
          <th>pvalue_2samp_ind_ttest</th>
          <th>pvalue_mann_whitney</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Control</td>
          <td>Group 1</td>
          <td>0.564092</td>
          <td>0.251295</td>
          <td>0.900291</td>
          <td>95.0</td>
          <td>True</td>
          <td>False</td>
          <td>0.001811</td>
          <td>0.004703</td>
        </tr>
      </tbody>
    </table>
    </div>



You can color the dots with any column in the DataFrame, using the
``color_col`` keyword.

.. code:: ipython3

    f2, results2 = dabest.plot(df, idx=('Control','Group 1'),
                               color_col='Gender')



.. image:: _images/tutorial_12_0.png

Paired two-group estimation plot
--------------------------------

DABEST can also produce estimation plots for paired observations
(repeated measures). This is done by setting the ``paired`` option to
``True``. You will also have to tell ``dabest.plot()`` which column
contains the identity of the each datum with the ``id_col`` keyword.

The estimation plot uses lines to indicate the pairs of observations.
This is known as a `slopegraph <https://www.edwardtufte.com/bboard/q-and-a-fetch-msg?msg_id=0003nk>`_. The combined visual effect of the slopes
of these lines serves to give the viewer an intuitive sense of the effect
size between the two groups of repeated observations.

.. code:: ipython3

    f3, results3 = dabest.plot(df, idx=('Control','Group 1'),
                               color_col='Gender',
                               paired=True, id_col="ID")



.. image:: _images/tutorial_15_0.png


Note that the statistical output records that the difference is a paired
one, in the ``is_paired`` column.

.. code:: ipython3

    results3




.. raw:: html

    <div>
    <style scoped>
      .dataframe {
        display: block;
        overflow-x: scroll;
        border-collapse: collapse;
      }

      .dataframe thead th {
        text-align: centre;
        background-color: #586e75;
        color: #eee8d5;
      }

      .dataframe td {
        padding:10px 25px 10px 1px;
        border-left: 1px solid #000;
        border-right: 1px solid #000;
      }

      .dataframe th td {
        border-bottom: 1px solid #ddd;
      }

      .dataframe tbody tr:nth-child(even) {
        background-color: #f2f2f2;
      }

      .dataframe tbody tr td {
        vertical-align: centre;
        text-align: right;
      }

      .dataframe tbody tr:hover {
        background-color: #eee8d5;
      }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>reference_group</th>
          <th>experimental_group</th>
          <th>stat_summary</th>
          <th>bca_ci_low</th>
          <th>bca_ci_high</th>
          <th>ci</th>
          <th>is_difference</th>
          <th>is_paired</th>
          <th>pvalue_2samp_paired_ttest</th>
          <th>pvalue_wilcoxon</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Control</td>
          <td>Group 1</td>
          <td>0.564092</td>
          <td>0.262493</td>
          <td>0.842903</td>
          <td>95.0</td>
          <td>True</td>
          <td>True</td>
          <td>0.001285</td>
          <td>0.003185</td>
        </tr>
      </tbody>
    </table>
    </div>



If you want to plot the raw swarmplot instead of the paired lines, use
the ``show_pairs`` flag to set this. The contrasts computed will still
be paired, as indicated by the DataFrame produced.

.. code:: ipython3

    f4, results4 = dabest.plot(df, idx=('Control','Group 1'),
                               color_col='Gender',
                               paired=True, id_col="ID",
                               show_pairs=False)



.. image:: _images/tutorial_19_0.png


Multi two-group estimation plot
-------------------------------

In a multi-group design, you can horizontally tile two or more two-group
floating-contrasts. This is designed to meet data visualization and
presentation paradigms that are predominant in academic biomedical
research, where several two-group experiments are presented together.

This is done mainly through the ``idx`` option. You can nest two or more
tuples or lists within a single tuple/list to create a seperate subplot
for that contrast.

The effect sizes and confidence intervals for each two-group plot will
be computed.

.. code:: ipython3

    f5, results5 = dabest.plot(df, idx=(('Control','Group 1'),
                                        ('Group 2','Group 3'),
                                        ('Group 4','Group 5'))
                              )

    results5




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>reference_group</th>
          <th>experimental_group</th>
          <th>stat_summary</th>
          <th>bca_ci_low</th>
          <th>bca_ci_high</th>
          <th>ci</th>
          <th>is_difference</th>
          <th>is_paired</th>
          <th>pvalue_2samp_ind_ttest</th>
          <th>pvalue_mann_whitney</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Control</td>
          <td>Group 1</td>
          <td>0.564092</td>
          <td>0.256753</td>
          <td>0.892652</td>
          <td>95.0</td>
          <td>True</td>
          <td>False</td>
          <td>0.001811</td>
          <td>0.004703</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Group 2</td>
          <td>Group 3</td>
          <td>0.253319</td>
          <td>-0.112335</td>
          <td>0.616018</td>
          <td>95.0</td>
          <td>True</td>
          <td>False</td>
          <td>0.190823</td>
          <td>0.155570</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Group 4</td>
          <td>Group 5</td>
          <td>-0.278511</td>
          <td>-0.551978</td>
          <td>0.019770</td>
          <td>95.0</td>
          <td>True</td>
          <td>False</td>
          <td>0.070806</td>
          <td>0.041124</td>
        </tr>
      </tbody>
    </table>
    </div>




.. image:: _images/tutorial_21_1.png


Each two-group experiment has its own floating contrast axes. Another
way to present this is to place all the effect sizes (and their
bootstrap distributions) on a common axes, beneath the raw data. This is
controlled with the ``float_contrast`` option.

.. code:: ipython3

    f6, results6 = dabest.plot(df, idx=(('Control','Group 1'),
                                        ('Group 2','Group 3'),
                                        ('Group 4','Group 5')),
                               float_contrast=False
                              )



.. image:: _images/tutorial_23_0.png


.. code:: ipython3

    f6_paired, results6_paired = dabest.plot(df, idx=(('Control','Group 1'),
                                                      ('Group 2','Group 3'),
                                                      ('Group 4','Group 5')),
                                             float_contrast=False,
                                             color_col='Gender',
                                             paired=True, id_col='ID'
                                            )



.. image:: _images/tutorial_24_0.png


Shared-control estimation plot
------------------------------

A common experimental setup seen in contemporary biomedical research is
a shared-control design (also known as a 'hub-and-spoke' design), where
two or more experimental groups are compared to a common control group.

A hub-and-spoke plot implements estimation statistics and aesthetics on
such an experimental design.

If more than 2 columns/groups are directly passed to ``idx``, then
``dabest.plot()`` will produce a shared-control estimation plot, where
the first group in the tuple/list is considered the control group. The
mean difference and confidence intervals of each subsequent group will
be computed against the first control group.

.. code:: ipython3

    f7, results7 = dabest.plot(df, idx=('Control', 'Group 2', 'Group 4'),
                       color_col='Gender')

    results7




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>reference_group</th>
          <th>experimental_group</th>
          <th>stat_summary</th>
          <th>bca_ci_low</th>
          <th>bca_ci_high</th>
          <th>ci</th>
          <th>is_difference</th>
          <th>is_paired</th>
          <th>pvalue_2samp_ind_ttest</th>
          <th>pvalue_mann_whitney</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Control</td>
          <td>Group 2</td>
          <td>-0.049862</td>
          <td>-0.318671</td>
          <td>0.231657</td>
          <td>95.0</td>
          <td>True</td>
          <td>False</td>
          <td>0.734693</td>
          <td>0.714980</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Control</td>
          <td>Group 4</td>
          <td>0.698509</td>
          <td>0.409364</td>
          <td>0.969665</td>
          <td>95.0</td>
          <td>True</td>
          <td>False</td>
          <td>0.000028</td>
          <td>0.000093</td>
        </tr>
      </tbody>
    </table>
    </div>




.. image:: _images/tutorial_26_1.png


In a shared control plot, the effect sizes and bootstrap 95CIs are shown
on the lower panel for all comparisons. By default, a summary line is
plotted for each group. The mean (indicated by the gap) and the standard
deviation is plotted as a gapped line.

One can display the median with the 25th and 75th percentiles (a
Tufte-style boxplot) using the ``group_summaries`` keyword argument in
the function.

.. code:: ipython3

    f8, results8 = dabest.plot(df, idx=('Control', 'Group 2', 'Group 4'),
                               color_col='Gender',
                               group_summaries='median_quartiles')



.. image:: _images/tutorial_28_0.png

----------------------
Controlling Aesthetics
----------------------

Below we run through ways of customizing various aesthetic features.

.. code:: ipython3

    # Changing the contrast y-limits.

    f9, results9 = dabest.plot(df, idx=('Control','Group 1','Group 2'),
                               color_col='Gender',
                               contrast_ylim=(-2,2))



.. image:: _images/tutorial_30_0.png


.. code:: ipython3

    # Changing the swarmplot y-limits.

    f10, results10 = dabest.plot(df, idx=('Control', 'Group 1', 'Group 2'),
                                 color_col='Gender',
                                 swarm_ylim=(-10, 10))



.. image:: _images/tutorial_31_0.png


.. code:: ipython3

    # Changing the figure size.
    # The default figure size has been tweaked for
    # optimal visual harmony and proportion in most use cases.
    # You can, however, tweak the figure size.

    f11, results11 = dabest.plot(df, idx=('Control','Group 1','Group 2'),
                                 color_col='Gender',
                                 fig_size=(10, 4) # xy dimensions in inches.
                                )



.. image:: _images/tutorial_32_0.png


.. code:: ipython3

    # Changing the size of the dots in the swarmplot.

    f12, results12 = dabest.plot(df, idx=('Control','Group 1','Group 2'),
                                 color_col='Gender', swarm_dotsize=13,
                                )



.. image:: _images/tutorial_33_0.png


.. code:: ipython3

    # Custom y-axis labels.
    f13, results13 = dabest.plot(df, idx=('Control','Group 1','Group 2'),
                                 color_col='Gender',
                                 # Insert line breaks manually with `\n`.
                                 swarm_label='My Custom\nSwarm Label',
                                 contrast_label='This is the\nEstimation Plot'
                                )



.. image:: _images/tutorial_34_0.png


.. code:: ipython3

    # Any of matplotlib's named colors will work.
    # See https://matplotlib.org/examples/color/named_colors.html

    f14, results14 = dabest.plot(df, idx=('Control','Group 1','Group 2'),
                                 color_col='Gender',
                                 # The categories in `color_col` will be
                                 # assigned alphabetically according
                                 # to the order of colors below.
                                 custom_palette=['darkorange', 'slategrey']
                                )



.. image:: _images/tutorial_35_0.png


.. code:: ipython3

    # You can also pass colors in the RGB tuple form (r, g, b),
    # or in hexadecimal form (if you're more familiar with HTML color codes).

    f15, results15 = dabest.plot(df, idx=('Control','Group 1','Group 2'),
                                 color_col='Gender',
                                 # Below I pass darkorange in RGB,
                                 # and slategrey in hexadecimal.
                                 custom_palette=[(1.0, 0.549, 0.0), '#708090']
                                )



.. image:: _images/tutorial_36_0.png


.. code:: ipython3

    # Passing a dictionary as a custom palette.
    f16, results16 = dabest.plot(df, idx=('Control','Group 1','Group 2'),
                                 color_col='Gender',
                                 custom_palette={'Male'   : 'slategrey',
                                                 'Female' : 'darkorange'}
                                )



.. image:: _images/tutorial_37_0.png


.. code:: ipython3

    # Tweaking the tick length and padding between tick and label.

    f17, results17 = dabest.plot(df, idx=('Control','Group 1','Group 2'),
                                 color_col='Gender',
                                 tick_length=5, tick_pad=5
                                )



.. image:: _images/tutorial_38_0.png

----------------------------------------------
Appendix: On working with 'melted' DataFrames.
----------------------------------------------

``dabest.plot`` can also work with 'melted' or 'longform' data. This
term is so used because each row will now correspond to a single
datapoint, with one column carrying the value (``value``) and other
columns carrying 'metadata' describing that datapoint (in this case,
``group`` and ``Gender``).

For more details on wide vs long or 'melted' data, see
https://en.wikipedia.org/wiki/Wide\_and\_narrow\_data

To read more about melting a dataframe,see
https://pandas.pydata.org/pandas-docs/stable/generated/pandas.melt.html

.. code:: ipython3

    x='group'
    y='my_metric'
    color_col='Gender'
    value_cols = df.columns[:-2] # select all but the 'Gender' and 'ID' columns.

    df_melt=pd.melt(df,
                    id_vars=['ID',color_col],
                    value_vars=value_cols,
                    value_name=y,
                    var_name=x)

    df_melt.head() # Gives the first five rows of `df_melt`.




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>ID</th>
          <th>Gender</th>
          <th>group</th>
          <th>my_metric</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>Female</td>
          <td>Control</td>
          <td>2.742313</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>Female</td>
          <td>Control</td>
          <td>2.681590</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>Female</td>
          <td>Control</td>
          <td>3.180724</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4</td>
          <td>Female</td>
          <td>Control</td>
          <td>1.961873</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5</td>
          <td>Female</td>
          <td>Control</td>
          <td>2.867556</td>
        </tr>
      </tbody>
    </table>
    </div>



If you are using a melted DataFrame, you will need to specify the ``x``
(containing the categorical group names) and ``y`` (containing the
numerical values for plotting) columns.

.. code:: ipython3

    f17, results17 = dabest.plot(df_melt,
                                 x='group',
                                 y='my_metric',
                                 idx=('Control','Group 1'),
                                 color_col='Gender'
                                )
    results17




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>reference_group</th>
          <th>experimental_group</th>
          <th>stat_summary</th>
          <th>bca_ci_low</th>
          <th>bca_ci_high</th>
          <th>ci</th>
          <th>is_difference</th>
          <th>is_paired</th>
          <th>pvalue_2samp_ind_ttest</th>
          <th>pvalue_mann_whitney</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Control</td>
          <td>Group 1</td>
          <td>0.564092</td>
          <td>0.244049</td>
          <td>0.881805</td>
          <td>95.0</td>
          <td>True</td>
          <td>False</td>
          <td>0.001811</td>
          <td>0.004703</td>
        </tr>
      </tbody>
    </table>
    </div>




.. image:: _images/tutorial_42_1.png
