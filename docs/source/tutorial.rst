.. _Tutorial:

========
Tutorial
========


Load Libraries
--------------

.. code-block:: python3
  :linenos:


    import numpy as np
    import pandas as pd
    import dabest

    print("We're using DABEST v{}".format(dabest.__version__))


.. parsed-literal::

    We're using DABEST v0.3.9999


Create dataset for demo
-----------------------

Here, we create a dataset to illustrate how ``dabest`` functions. In
this dataset, each column corresponds to a group of observations.

.. code-block:: python3
  :linenos:


    from scipy.stats import norm # Used in generation of populations.

    np.random.seed(9999) # Fix the seed so the results are replicable.
    # pop_size = 10000 # Size of each population.
    Ns = 20 # The number of samples taken from each population

    # Create samples
    c1 = norm.rvs(loc=3, scale=0.4, size=Ns)
    c2 = norm.rvs(loc=3.5, scale=0.75, size=Ns)
    c3 = norm.rvs(loc=3.25, scale=0.4, size=Ns)

    t1 = norm.rvs(loc=3.5, scale=0.5, size=Ns)
    t2 = norm.rvs(loc=2.5, scale=0.6, size=Ns)
    t3 = norm.rvs(loc=3, scale=0.75, size=Ns)
    t4 = norm.rvs(loc=3.5, scale=0.75, size=Ns)
    t5 = norm.rvs(loc=3.25, scale=0.4, size=Ns)
    t6 = norm.rvs(loc=3.25, scale=0.4, size=Ns)


    # Add a `gender` column for coloring the data.
    females = np.repeat('Female', Ns/2).tolist()
    males = np.repeat('Male', Ns/2).tolist()
    gender = females + males

    # Add an `id` column for paired data plotting.
    id_col = pd.Series(range(1, Ns+1))

    # Combine samples and gender into a DataFrame.
    df = pd.DataFrame({'Control 1' : c1,     'Test 1' : t1,
                       'Control 2' : c2,     'Test 2' : t2,
                       'Control 3' : c3,     'Test 3' : t3,
                       'Test 4'    : t4,     'Test 5' : t5, 'Test 6' : t6,
                       'Gender'    : gender, 'ID'  : id_col
                      })

Note that we have 9 groups (3 Control samples and 6 Test samples). Our
dataset also has a non-numerical column indicating gender, and another
column indicating the identity of each observation.

This is known as a ‘wide’ dataset. See this
`writeup <https://sejdemyr.github.io/r-tutorials/basics/wide-and-long/>`__
for more details.

.. code-block:: python3
  :linenos:


    df.head()




.. raw:: html

    <div>
    <style scoped>
      /*  .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        } */
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Control 1</th>
          <th>Test 1</th>
          <th>Control 2</th>
          <th>Test 2</th>
          <th>Control 3</th>
          <th>Test 3</th>
          <th>Test 4</th>
          <th>Test 5</th>
          <th>Test 6</th>
          <th>Gender</th>
          <th>ID</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>2.793984</td>
          <td>3.420875</td>
          <td>3.324661</td>
          <td>1.707467</td>
          <td>3.816940</td>
          <td>1.796581</td>
          <td>4.440050</td>
          <td>2.937284</td>
          <td>3.486127</td>
          <td>Female</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>3.236759</td>
          <td>3.467972</td>
          <td>3.685186</td>
          <td>1.121846</td>
          <td>3.750358</td>
          <td>3.944566</td>
          <td>3.723494</td>
          <td>2.837062</td>
          <td>2.338094</td>
          <td>Female</td>
          <td>2</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3.019149</td>
          <td>4.377179</td>
          <td>5.616891</td>
          <td>3.301381</td>
          <td>2.945397</td>
          <td>2.832188</td>
          <td>3.214014</td>
          <td>3.111950</td>
          <td>3.270897</td>
          <td>Female</td>
          <td>3</td>
        </tr>
        <tr>
          <th>3</th>
          <td>2.804638</td>
          <td>4.564780</td>
          <td>2.773152</td>
          <td>2.534018</td>
          <td>3.575179</td>
          <td>3.048267</td>
          <td>4.968278</td>
          <td>3.743378</td>
          <td>3.151188</td>
          <td>Female</td>
          <td>4</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2.858019</td>
          <td>3.220058</td>
          <td>2.550361</td>
          <td>2.796365</td>
          <td>3.692138</td>
          <td>3.276575</td>
          <td>2.662104</td>
          <td>2.977341</td>
          <td>2.328601</td>
          <td>Female</td>
          <td>5</td>
        </tr>
      </tbody>
    </table>
    </div>



Loading Data
------------

Before we create estimation plots and obtain confidence intervals for
our effect sizes, we need to load the data and the relevant groups.

We simply supply the DataFrame to ``dabest.load()``. We also must supply
the two groups you want to compare in the ``idx`` argument as a tuple or
list.

.. code-block:: python3
  :linenos:


    two_groups_unpaired = dabest.load(df, idx=("Control 1", "Test 1"), resamples=5000)

Calling this ``Dabest`` object gives you a gentle greeting, as well as
the comparisons that can be computed.

.. code-block:: python3
  :linenos:


    two_groups_unpaired




.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good evening!
    The current time is Sun Aug 29 18:00:54 2021.

    Effect size(s) with 95% confidence intervals will be computed for:
    1. Test 1 minus Control 1

    5000 resamples will be used to generate the effect size bootstraps.



Changing statistical parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can change the width of the confidence interval that will be
produced by manipulating the ``ci`` argument.

.. code-block:: python3
  :linenos:


    two_groups_unpaired_ci90 = dabest.load(df, idx=("Control 1", "Test 1"), ci=90)

.. code-block:: python3
  :linenos:


    two_groups_unpaired_ci90




.. parsed-literal::

    DABEST v0.3.1
    =============

    Good afternoon!
    The current time is Mon Oct 19 17:12:44 2020.

    Effect size(s) with 90% confidence intervals will be computed for:
    1. Test 1 minus Control 1

    5000 resamples will be used to generate the effect size bootstraps.



Effect sizes
------------

``dabest`` now features a range of effect sizes:
  - the mean difference (``mean_diff``)
  - the median difference (``median_diff``)
  - `Cohen’s d <https://en.wikipedia.org/wiki/Effect_size#Cohen's_d>`__ (``cohens_d``)
  - `Hedges’ g <https://en.wikipedia.org/wiki/Effect_size#Hedges'_g>`__ (``hedges_g``)
  - `Cliff’s delta <https://en.wikipedia.org/wiki/Effect_size#Effect_size_for_ordinal_data>`__ (``cliffs_delta``)

Each of these are attributes of the ``Dabest`` object.

.. code-block:: python3
  :linenos:


    two_groups_unpaired.mean_diff




.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good evening!
    The current time is Sun Aug 29 18:10:44 2021.

    The unpaired mean difference between Control 1 and Test 1 is 0.48 [95%CI 0.221, 0.768].
    The p-value of the two-sided permutation t-test is 0.001, calculated for legacy purposes only. 

    5000 bootstrap samples were taken; the confidence interval is bias-corrected and accelerated.
    Any p-value reported is the probability of observing theeffect size (or greater),
    assuming the null hypothesis ofzero difference is true.
    For each p-value, 5000 reshuffles of the control and test labels were performed.

    To get the results of all valid statistical tests, use `.mean_diff.statistical_tests`



For each comparison, the type of effect size is reported (here, it’s the
“unpaired mean difference”). The confidence interval is reported as:
[*confidenceIntervalWidth* *LowerBound*, *UpperBound*]

This confidence interval is generated through bootstrap resampling. See
:doc:`bootstraps` for more details.

Since v0.3.0, DABEST will report the p-value of the `non-parametric two-sided approximate permutation t-test <https://en.wikipedia.org/wiki/Resampling_(statistics)#Permutation_tests>`__. This is also known as the Monte Carlo permutation test.

For unpaired comparisons, the p-values and test statistics of `Welch's t test <https://en.wikipedia.org/wiki/Welch%27s_t-test>`__, `Student's t test <https://en.wikipedia.org/wiki/Student%27s_t-test>`__, and `Mann-Whitney U test <https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test>`__ can be found in addition. For paired comparisons, the p-values and test statistics of the `paired Student's t <https://en.wikipedia.org/wiki/Student%27s_t-test#Paired_samples>`__ and `Wilcoxon <https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test>`__ tests are presented.

.. code-block:: python3
  :linenos:


    pd.options.display.max_columns = 50
    two_groups_unpaired.mean_diff.results




.. raw:: html

    <div>
    <style scoped>
        /* .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        } */
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>control</th>
          <th>test</th>
          <th>control_N</th>
          <th>test_N</th>
          <th>effect_size</th>
          <th>is_paired</th>
          <th>difference</th>
          <th>ci</th>
          <th>bca_low</th>
          <th>bca_high</th>
          <th>bca_interval_idx</th>
          <th>pct_low</th>
          <th>pct_high</th>
          <th>pct_interval_idx</th>
          <th>bootstraps</th>
          <th>resamples</th>
          <th>random_seed</th>
          <th>permutations</th>
          <th>pvalue_permutation</th>
          <th>permutation_count</th>
          <th>permutations_var</th>
          <th>pvalue_welch</th>
          <th>statistic_welch</th>
          <th>pvalue_students_t</th>
          <th>statistic_students_t</th>
          <th>pvalue_mann_whitney</th>
          <th>statistic_mann_whitney</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Control 1</td>
          <td>Test 1</td>
          <td>20</td>
          <td>20</td>
          <td>mean difference</td>
          <td>None</td>
          <td>0.48029</td>
          <td>95</td>
          <td>0.220869</td>
          <td>0.767721</td>
          <td>(140, 4889)</td>
          <td>0.215697</td>
          <td>0.761716</td>
          <td>(125, 4875)</td>
          <td>[0.6686169333655454, 0.4382051534234943, 0.665...</td>
          <td>5000</td>
          <td>12345</td>
          <td>[-0.17259843762502491, 0.03802293852634886, -0...</td>
          <td>0.001</td>
          <td>5000</td>
          <td>[0.026356588154404337, 0.027102495439046997, 0...</td>
          <td>0.002094</td>
          <td>-3.308806</td>
          <td>0.002057</td>
          <td>-3.308806</td>
          <td>0.001625</td>
          <td>83.0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code-block:: python3
  :linenos:


    two_groups_unpaired.mean_diff.statistical_tests




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
          <th>control</th>
          <th>test</th>
          <th>control_N</th>
          <th>test_N</th>
          <th>effect_size</th>
          <th>is_paired</th>
          <th>difference</th>
          <th>ci</th>
          <th>bca_low</th>
          <th>bca_high</th>
          <th>pvalue_permutation</th>
          <th>pvalue_welch</th>
          <th>statistic_welch</th>
          <th>pvalue_students_t</th>
          <th>statistic_students_t</th>
          <th>pvalue_mann_whitney</th>
          <th>statistic_mann_whitney</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Control 1</td>
          <td>Test 1</td>
          <td>20</td>
          <td>20</td>
          <td>mean difference</td>
          <td>None</td>
          <td>0.48029</td>
          <td>95</td>
          <td>0.220869</td>
          <td>0.767721</td>
          <td>0.001</td>
          <td>0.002094</td>
          <td>-3.308806</td>
          <td>0.002057</td>
          <td>-3.308806</td>
          <td>0.001625</td>
          <td>83.0</td>
        </tr>
      </tbody>
    </table>
    </div>



Let’s compute the Hedges’ *g* for our comparison.

.. code-block:: python3
  :linenos:


    two_groups_unpaired.hedges_g




.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good evening!
    The current time is Sun Aug 29 18:12:17 2021.

    The unpaired Hedges' g between Control 1 and Test 1 is 1.03 [95%CI 0.349, 1.62].
    The p-value of the two-sided permutation t-test is 0.001, calculated for legacy purposes only. 

    5000 bootstrap samples were taken; the confidence interval is bias-corrected and accelerated.
    Any p-value reported is the probability of observing theeffect size (or greater),
    assuming the null hypothesis ofzero difference is true.
    For each p-value, 5000 reshuffles of the control and test labels were performed.

    To get the results of all valid statistical tests, use `.hedges_g.statistical_tests`



.. code-block:: python3
  :linenos:


    two_groups_unpaired.hedges_g.results




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
          <th>control</th>
          <th>test</th>
          <th>control_N</th>
          <th>test_N</th>
          <th>effect_size</th>
          <th>is_paired</th>
          <th>difference</th>
          <th>ci</th>
          <th>bca_low</th>
          <th>bca_high</th>
          <th>bca_interval_idx</th>
          <th>pct_low</th>
          <th>pct_high</th>
          <th>pct_interval_idx</th>
          <th>bootstraps</th>
          <th>resamples</th>
          <th>random_seed</th>
          <th>permutations</th>
          <th>pvalue_permutation</th>
          <th>permutation_count</th>
          <th>permutations_var</th>
          <th>pvalue_welch</th>
          <th>statistic_welch</th>
          <th>pvalue_students_t</th>
          <th>statistic_students_t</th>
          <th>pvalue_mann_whitney</th>
          <th>statistic_mann_whitney</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Control 1</td>
          <td>Test 1</td>
          <td>20</td>
          <td>20</td>
          <td>Hedges' g</td>
          <td>None</td>
          <td>1.025525</td>
          <td>95</td>
          <td>0.349394</td>
          <td>1.618579</td>
          <td>(42, 4724)</td>
          <td>0.472844</td>
          <td>1.74166</td>
          <td>(125, 4875)</td>
          <td>[1.1337301267831184, 0.8311210968422604, 1.539...</td>
          <td>5000</td>
          <td>12345</td>
          <td>[-0.3295089865590538, 0.07158401210924781, -0....</td>
          <td>0.001</td>
          <td>5000</td>
          <td>[0.026356588154404337, 0.027102495439046997, 0... </td>
          <td>0.002094</td>
          <td>-3.308806</td>
          <td>0.002057</td>
          <td>-3.308806</td>
          <td>0.001625</td>
          <td>83.0</td>
        </tr>
      </tbody>
    </table>
    </div>



Producing estimation plots
--------------------------

To produce a **Gardner-Altman estimation plot**, simply use the
``.plot()`` method. You can read more about its genesis and design
inspiration at :doc:`robust-beautiful`.

Every effect size instance has access to the ``.plot()`` method. This
means you can quickly create plots for different effect sizes easily.

.. code-block:: python3
  :linenos:


  two_groups_unpaired.mean_diff.plot();



.. image:: _images/tutorial_27_0.png


.. code-block:: python3
  :linenos:


    two_groups_unpaired.hedges_g.plot();



.. image:: _images/tutorial_28_0.png


Instead of a Gardner-Altman plot, you can produce a **Cumming estimation
plot** by setting ``float_contrast=False`` in the ``plot()`` method.
This will plot the bootstrap effect sizes below the raw data, and also
displays the the mean (gap) and ± standard deviation of each group
(vertical ends) as gapped lines. This design was inspired by Edward
Tufte’s dictum to maximise the data-ink ratio.

.. code-block:: python3
  :linenos:


    two_groups_unpaired.hedges_g.plot(float_contrast=False);



.. image:: _images/tutorial_30_0.png


The ``dabest`` package also implements a range of estimation plot
designs aimed at depicting common experimental designs.

The **multi-two-group estimation plot** tiles two or more Cumming plots
horizontally, and is created by passing a *nested tuple* to ``idx`` when
``dabest.load()`` is first invoked.

Thus, the lower axes in the Cumming plot is effectively a `forest
plot <https://en.wikipedia.org/wiki/Forest_plot>`__, used in
meta-analyses to aggregate and compare data from different experiments.

.. code-block:: python3
  :linenos:


    multi_2group = dabest.load(df, idx=(("Control 1", "Test 1",),
                                         ("Control 2", "Test 2")
                                       ))

    multi_2group.mean_diff.plot();



.. image:: _images/tutorial_35_0.png


The **shared control plot** displays another common experimental
paradigm, where several test samples are compared against a common
reference sample.

This type of Cumming plot is automatically generated if the tuple passed
to ``idx`` has more than two data columns.

.. code-block:: python3
  :linenos:


    shared_control = dabest.load(df, idx=("Control 1", "Test 1",
                                          "Test 2", "Test 3",
                                          "Test 4", "Test 5", "Test 6")
                                 )

.. code-block:: python3
  :linenos:


    shared_control




.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good evening!
    The current time is Tue Aug 31 23:39:22 2021.

    Effect size(s) with 95% confidence intervals will be computed for:
    1. Test 1 minus Control 1
    2. Test 2 minus Control 1
    3. Test 3 minus Control 1
    4. Test 4 minus Control 1
    5. Test 5 minus Control 1
    6. Test 6 minus Control 1

    5000 resamples will be used to generate the effect size bootstraps.



.. code-block:: python3
  :linenos:


    shared_control.mean_diff




.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good evening!
    The current time is Tue Aug 31 23:42:39 2021.

    The unpaired mean difference between Control 1 and Test 1 is 0.48 [95%CI 0.221, 0.768].
    The p-value of the two-sided permutation t-test is 0.001, calculated for legacy purposes only. 

    The unpaired mean difference between Control 1 and Test 2 is -0.542 [95%CI -0.914, -0.211].
    The p-value of the two-sided permutation t-test is 0.0042, calculated for legacy purposes only. 

    The unpaired mean difference between Control 1 and Test 3 is 0.174 [95%CI -0.295, 0.628].
    The p-value of the two-sided permutation t-test is 0.479, calculated for legacy purposes only. 

    The unpaired mean difference between Control 1 and Test 4 is 0.79 [95%CI 0.306, 1.31].
    The p-value of the two-sided permutation t-test is 0.0042, calculated for legacy purposes only. 

    The unpaired mean difference between Control 1 and Test 5 is 0.265 [95%CI 0.0137, 0.497].
    The p-value of the two-sided permutation t-test is 0.0404, calculated for legacy purposes only. 

    The unpaired mean difference between Control 1 and Test 6 is 0.288 [95%CI -0.00441, 0.515].
    The p-value of the two-sided permutation t-test is 0.0324, calculated for legacy purposes only. 

    5000 bootstrap samples were taken; the confidence interval is bias-corrected and accelerated.
    Any p-value reported is the probability of observing theeffect size (or greater),
    assuming the null hypothesis ofzero difference is true.
    For each p-value, 5000 reshuffles of the control and test labels were performed.

    To get the results of all valid statistical tests, use `.mean_diff.statistical_tests`



.. code-block:: python3
  :linenos:


    shared_control.mean_diff.plot();



.. image:: _images/tutorial_42_0.png


``dabest`` thus empowers you to robustly perform and elegantly present
complex visualizations and statistics.

.. code-block:: python3
  :linenos:


    multi_groups = dabest.load(df, idx=(("Control 1", "Test 1",),
                                         ("Control 2", "Test 2","Test 3"),
                                         ("Control 3", "Test 4","Test 5", "Test 6")
                                       ))


.. code-block:: python3
  :linenos:


    multi_groups




.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good evening!
    The current time is Tue Aug 31 23:47:40 2021.

    Effect size(s) with 95% confidence intervals will be computed for:
    1. Test 1 minus Control 1
    2. Test 2 minus Control 2
    3. Test 3 minus Control 2
    4. Test 4 minus Control 3
    5. Test 5 minus Control 3
    6. Test 6 minus Control 3

    5000 resamples will be used to generate the effect size bootstraps.



.. code-block:: python3
  :linenos:


    multi_groups.mean_diff




.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good evening!
    The current time is Tue Aug 31 23:48:17 2021.

    The unpaired mean difference between Control 1 and Test 1 is 0.48 [95%CI 0.221, 0.768].
    The p-value of the two-sided permutation t-test is 0.001, calculated for legacy purposes only. 

    The unpaired mean difference between Control 2 and Test 2 is -1.38 [95%CI -1.93, -0.895].
    The p-value of the two-sided permutation t-test is 0.0, calculated for legacy purposes only. 

    The unpaired mean difference between Control 2 and Test 3 is -0.666 [95%CI -1.3, -0.103].
    The p-value of the two-sided permutation t-test is 0.0352, calculated for legacy purposes only. 

    The unpaired mean difference between Control 3 and Test 4 is 0.362 [95%CI -0.114, 0.887].
    The p-value of the two-sided permutation t-test is 0.161, calculated for legacy purposes only. 

    The unpaired mean difference between Control 3 and Test 5 is -0.164 [95%CI -0.404, 0.0742].
    The p-value of the two-sided permutation t-test is 0.208, calculated for legacy purposes only. 

    The unpaired mean difference between Control 3 and Test 6 is -0.14 [95%CI -0.398, 0.102].
    The p-value of the two-sided permutation t-test is 0.282, calculated for legacy purposes only. 

    5000 bootstrap samples were taken; the confidence interval is bias-corrected and accelerated.
    Any p-value reported is the probability of observing theeffect size (or greater),
    assuming the null hypothesis ofzero difference is true.
    For each p-value, 5000 reshuffles of the control and test labels were performed.

    To get the results of all valid statistical tests, use `.mean_diff.statistical_tests`



.. code-block:: python3
  :linenos:


    multi_groups.mean_diff.plot();



.. image:: _images/tutorial_47_0.png


Using long (aka ‘melted’) data frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``dabest`` can also work with ‘melted’ or ‘long’ data. This term is so
used because each row will now correspond to a single datapoint, with
one column carrying the value and other columns carrying ‘metadata’
describing that datapoint.

More details on wide vs long or ‘melted’ data can be found in this
`Wikipedia
article <https://en.wikipedia.org/wiki/Wide_and_narrow_data>`__. The
`pandas
documentation <https://pandas.pydata.org/pandas-docs/stable/generated/pandas.melt.html>`__
gives recipes for melting dataframes.

.. code-block:: python3
  :linenos:


    x='group'
    y='metric'

    value_cols = df.columns[:-2] # select all but the "Gender" and "ID" columns.

    df_melted = pd.melt(df.reset_index(),
                        id_vars=["Gender", "ID"],
                        value_vars=value_cols,
                        value_name=y,
                        var_name=x)

    df_melted.head() # Gives the first five rows of `df_melted`.




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
          <th>Gender</th>
          <th>ID</th>
          <th>group</th>
          <th>metric</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Female</td>
          <td>1</td>
          <td>Control 1</td>
          <td>2.793984</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Female</td>
          <td>2</td>
          <td>Control 1</td>
          <td>3.236759</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Female</td>
          <td>3</td>
          <td>Control 1</td>
          <td>3.019149</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Female</td>
          <td>4</td>
          <td>Control 1</td>
          <td>2.804638</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Female</td>
          <td>5</td>
          <td>Control 1</td>
          <td>2.858019</td>
        </tr>
      </tbody>
    </table>
    </div>



When your data is in this format, you will need to specify the ``x`` and
``y`` columns in ``dabest.load()``.

.. code-block:: python3
  :linenos:


    analysis_of_long_df = dabest.load(df_melted, idx=("Control 1", "Test 1"),
                                     x="group", y="metric")

    analysis_of_long_df




.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good evening!
    The current time is Tue Aug 31 23:51:12 2021.

    Effect size(s) with 95% confidence intervals will be computed for:
    1. Test 1 minus Control 1

    5000 resamples will be used to generate the effect size bootstraps.



.. code-block:: python3
  :linenos:


    analysis_of_long_df.mean_diff.plot();



.. image:: _images/tutorial_52_0.png




Repeated-measures function
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Since v0.4.0, DABEST will support the repeated-measures feature, which allows 
the visualization of paired experiments with one control and multiple test 
groups. This is an enhanced version of the paired data computation function 
in the former version where computations involving one control group and one
test group are supported.

The repeated-measures function supports the calculation of effect sizes for
paired data, either based on sequential comparisons (group i vs group i + 1) 
or baseline comparisons (control vs group i). To use the repeated-measures function, 
you can simply declare ``paired = "sequential"`` or ``paired = "baseline"`` 
correspondingly. Same as in the previous version, you must also pass a column in 
the dataset that indicates the identity of each observation, using the 
``id_col`` keyword. (Please note that ``paired = True`` and ``paired = False``
are no longer valid in the v0.4.0.)


.. code-block:: python3
  :linenos:


    two_groups_paired_sequential = dabest.load(df, idx=("Control 1", "Test 1"),
                                               paired="sequential", id_col="ID")

.. code-block:: python3
  :linenos:


    two_groups_paired_sequential




.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good evening!
    The current time is Mon Aug 30 00:09:54 2021.

    Paired effect size(s) for the sequential design of repeated-measures experiment 
    with 95% confidence intervals will be computed for:
    1. Test 1 minus Control 1

    5000 resamples will be used to generate the effect size bootstraps.



.. code-block:: python3
  :linenos:


    two_groups_paired_baseline = dabest.load(df, idx=("Control 1", "Test 1"),
                                  paired="baseline", id_col="ID")

.. code-block:: python3
  :linenos:


    two_groups_paired_baseline




.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good evening!
    The current time is Mon Aug 30 00:13:17 2021.

    Paired effect size(s) for repeated measures against baseline 
    with 95% confidence intervals will be computed for:
    1. Test 1 minus Control 1

    5000 resamples will be used to generate the effect size bootstraps.



When only 2 paired data groups are involved, assigning either ``baseline``
or ``sequential`` to ``paired`` will give you the same numerical results.

.. code-block:: python3
  :linenos:


    two_groups_paired_sequential.mean_diff




.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good evening!
    The current time is Mon Aug 30 00:14:44 2021.

    The paired mean difference for the sequential design of repeated-measures experiment 
    between Control 1 and Test 1 is 0.48 [95%CI 0.237, 0.73].
    The p-value of the two-sided permutation t-test is 0.001, calculated for legacy purposes only. 

    5000 bootstrap samples were taken; the confidence interval is bias-corrected and accelerated.
    Any p-value reported is the probability of observing theeffect size (or greater),
    assuming the null hypothesis ofzero difference is true.
    For each p-value, 5000 reshuffles of the control and test labels were performed.

    To get the results of all valid statistical tests, use `.mean_diff.statistical_tests`



.. code-block:: python3
  :linenos:


    two_groups_paired_baseline.mean_diff




.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good evening!
    The current time is Mon Aug 30 00:18:09 2021.

    The paired mean difference for repeated measures against baseline 
    between Control 1 and Test 1 is 0.48 [95%CI 0.237, 0.73].
    The p-value of the two-sided permutation t-test is 0.001, calculated for legacy purposes only. 

    5000 bootstrap samples were taken; the confidence interval is bias-corrected and accelerated.
    Any p-value reported is the probability of observing theeffect size (or greater),
    assuming the null hypothesis ofzero difference is true.
    For each p-value, 5000 reshuffles of the control and test labels were performed.

    To get the results of all valid statistical tests, use `.mean_diff.statistical_tests`



For paired data, we use
`slopegraphs <https://www.edwardtufte.com/bboard/q-and-a-fetch-msg?msg_id=0003nk>`__
(another innovation from Edward Tufte) to connect paired observations.
Both Gardner-Altman and Cumming plots support this.

.. code-block:: python3
  :linenos:


    two_groups_paired_sequential.mean_diff.plot();



.. image:: _images/tutorial_32_0.png


.. code-block:: python3
  :linenos:


    two_groups_paired_sequential.mean_diff.plot(float_contrast=False);



.. image:: _images/tutorial_33_0.png


.. code-block:: python3
  :linenos:


    two_groups_paired_baseline.mean_diff.plot();



.. image:: _images/tutorial_32_0.png


.. code-block:: python3
  :linenos:


    two_groups_paired_baseline.mean_diff.plot(float_contrast=False);



.. image:: _images/tutorial_33_0.png

You can also create repeated-measures plots with multiple test groups.In
this case, declaring ``paired`` to be ``sequential`` or ``baseline`` will
generate different results.

.. code-block:: python3
  :linenos:

    sequential_repeated_measures = dabest.load(df, idx=("Control 1", "Test 1", "Test 2", "Test 3"),
                                               paired="sequential", id_col="ID")

.. code-block:: python3
  :linenos:
  
    sequential_repeated_measures.mean_diff


.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good evening!
    The current time is Mon Aug 30 00:51:21 2021.

    The paired mean difference for the sequential design of repeated-measures experiment 
    between Control 1 and Test 1 is 0.48 [95%CI 0.237, 0.73].
    The p-value of the two-sided permutation t-test is 0.001, calculated for legacy purposes only. 

    The paired mean difference for the sequential design of repeated-measures experiment 
    between Test 1 and Test 2 is -1.02 [95%CI -1.36, -0.716].
    The p-value of the two-sided permutation t-test is 0.0, calculated for legacy purposes only. 

    The paired mean difference for the sequential design of repeated-measures experiment 
    between Test 2 and Test 3 is 0.716 [95%CI 0.14, 1.22].
    The p-value of the two-sided permutation t-test is 0.022, calculated for legacy purposes only. 

    5000 bootstrap samples were taken; the confidence interval is bias-corrected and accelerated.
    Any p-value reported is the probability of observing theeffect size (or greater),
    assuming the null hypothesis ofzero difference is true.
    For each p-value, 5000 reshuffles of the control and test labels were performed.

    To get the results of all valid statistical tests, use `.mean_diff.statistical_tests`



.. code-block:: python3
  :linenos:

    sequential_repeated_measures.mean_diff.plot();



.. image:: _images/tutorial_103_0.png


.. code-block:: python3
  :linenos:

    baseline_repeated_measures = dabest.load(df, idx=("Control 1", "Test 1", "Test 2", "Test 3"),
                                               paired="baseline", id_col="ID")



.. code-block:: python3
  :linenos:
  
    baseline_repeated_measures.mean_diff



.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good evening!
    The current time is Mon Aug 30 00:56:37 2021.

    The paired mean difference for repeated measures against baseline 
    between Control 1 and Test 1 is 0.48 [95%CI 0.237, 0.73].
    The p-value of the two-sided permutation t-test is 0.001, calculated for legacy purposes only. 

    The paired mean difference for repeated measures against baseline 
    between Control 1 and Test 2 is -0.542 [95%CI -0.975, -0.198].
    The p-value of the two-sided permutation t-test is 0.014, calculated for legacy purposes only. 

    The paired mean difference for repeated measures against baseline 
    between Control 1 and Test 3 is 0.174 [95%CI -0.297, 0.706].
    The p-value of the two-sided permutation t-test is 0.505, calculated for legacy purposes only. 

    5000 bootstrap samples were taken; the confidence interval is bias-corrected and accelerated.
    Any p-value reported is the probability of observing theeffect size (or greater),
    assuming the null hypothesis ofzero difference is true.
    For each p-value, 5000 reshuffles of the control and test labels were performed.

    To get the results of all valid statistical tests, use `.mean_diff.statistical_tests`




.. code-block:: python3
  :linenos:

    baseline_repeated_measures.mean_diff.plot();



.. image:: _images/tutorial_104_0.png

Same as that for unpaired data, DABEST empowers you to perform complex 
visualizations and statistics for paired data as well.

.. code-block:: python3
  :linenos:

    multi_baseline_repeated_measures = dabest.load(df, idx=(("Control 1", "Test 1", "Test 2", "Test 3"),
                                                      ("Control 2", "Test 4", "Test 5", "Test 6")),
                                               paired="baseline", id_col="ID")
    multi_baseline_repeated_measures.mean_diff.plot();



.. image:: _images/tutorial_105_0.png



Mini-meta function
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In v.0.4.0, DABEST also supports the computation and plotting of weighted-average
mean differences for both paired and unpaired data.Here the weights are the 
inverses of the group variance of 2 experiment groups involved. However, this is 
limited to comparisons between 2 experiment groups or in the multi-2-groups 
situation. Also, it does not allow the computation of other weighted effect sizes. 
More details on weighted differences can be found in this 
`Wikipedia article <https://en.wikipedia.org/wiki/Weighted_arithmetic_mean>`__. 

You can calculate weighted average mean differences by setting ``mini_meta=True`` in 
``dabest.load()``.

.. code-block:: python3
  :linenos:


    mini_meta_unpaired = dabest.load(df, idx=(("Control 1", "Test 1"), 
                                              ("Control 2", "Test 2"), 
                                              ("Control 3", "Test 3")),
                                     mini_meta=True)




.. code-block:: python3
  :linenos:


    mini_meta_unpaired




Now you can see from the greeting that the calculation of weighted average is 
enabled.

.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good evening!
    The current time is Sun Aug 29 00:37:19 2021.

    Effect size(s) with 95% confidence intervals will be computed for:
    1. Test 1 minus Control 1
    2. Test 2 minus Control 2
    3. Test 3 minus Control 3
    4. weighted delta (only for mean difference)

    5000 resamples will be used to generate the effect size bootstraps.



.. code-block:: python3
  :linenos:


    mini_meta_unpaired.mean_diff




You can read off the exact weighted average as well as the confidence 
intervals from the greeting.

.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good evening! 
    The current time is Sun Aug 29 00:31:43 2021.

    The unpaired mean difference between Control 1 and Test 1 is 0.48 [95%CI 0.221, 0.768].
    The p-value of the two-sided permutation t-test is 0.001, calculated for legacy purposes only. 

    The unpaired mean difference between Control 2 and Test 2 is -1.38 [95%CI -1.93, -0.895].
    The p-value of the two-sided permutation t-test is 0.0, calculated for legacy purposes only. 

    The unpaired mean difference between Control 3 and Test 3 is -0.255 [95%CI -0.717, 0.196].
    The p-value of the two-sided permutation t-test is 0.293, calculated for legacy purposes only. 

    The weighted-average unpaired mean differences is -0.0104 [95%CI -0.222, 0.215].
    The p-value of the two-sided permutation t-test is 0.937, calculated for legacy purposes only. 

    5000 bootstrap samples were taken; the confidence interval is bias-corrected and accelerated.
    Any p-value reported is the probability of observing theeffect size (or greater),
    assuming the null hypothesis ofzero difference is true.
    For each p-value, 5000 reshuffles of the control and test labels were performed.

    To get the results of all valid statistical tests, use `.mean_diff.statistical_tests`



In a mini-meta plot, there will be a half-violin plot of the weighted averages
added on the right side of the bootstraps plots with a shared y-axis.

.. code-block:: python3
  :linenos:


    mini_meta_unpaired.mean_diff.plot()



.. image:: _images/tutorial_100_0.png


In addition, all the information of the weighted average is stored in an attribute 
named ``mini_meta_delta`` of the effect size object. 


.. code-block:: python3
  :linenos:


    mini_meta_delta = mini_meta_unpaired.mean_diff.mini_meta_delta
    mini_meta_delta



.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good evening!
    The current time is Sun Aug 29 00:53:21 2021.

    The weighted-average unpaired mean differences is -0.0104 [95%CI -0.222, 0.215].
    The p-value of the two-sided permutation t-test is 0.937, calculated for legacy purposes only. 

    5000 bootstrap samples were taken; the confidence interval is bias-corrected and accelerated.
    Any p-value reported is the probability of observing theeffect size (or greater),
    assuming the null hypothesis ofzero difference is true.
    For each p-value, 5000 reshuffles of the control and test labels were performed.


``mini_meta_delta`` has its own attributes, containing various information pertaining to the weighted average:
  - ``group_var``: the pooled group variances of each set of 2 experiment groups
  - ``difference``: the weighted mean difference calculated based on the raw data
  - ``bootstraps``: the deltas of each set of 2 experiment groups calculated based on the bootstraps
  - ``bootstraps_weighted_delta``: the weighted deltas calculated based on the bootstraps
  - ``permutations``: the deltas of each set of 2 experiment groups calculated based on the permutation data
  - ``permutations_var``: the pooled group variances of each set of 2 experiment groups calculated based on permutation data
  - ``permutations_weighted_delta``: the weighted deltas calculated based on the permutation data

``mini_meta_delta.to_dict()`` will return to you all the attributes in a dictionary format.

The calculation of weighted avaerage is supported for paired data as well. You can 
declare ``paired`` to be either ``baseline`` or ``sequential``.

.. code-block:: python3
  :linenos:

    mini_meta_paired = dabest.load(df, id_col = "ID",
                                   idx=(("Control 1", "Test 1"),
                                        ("Control 2", "Test 2"),
                                        ("Control 3", "Test 3")),
                                   paired = "baseline", mini_meta=True)
    mini_meta_paired.mean_diff




.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good morning!
    The current time is Sun Aug 29 01:24:06 2021.

    The paired mean difference for repeated measures against baseline 
    between Control 1 and Test 1 is 0.48 [95%CI 0.237, 0.73].
    The p-value of the two-sided permutation t-test is 0.001, calculated for legacy purposes only. 

    The paired mean difference for repeated measures against baseline 
    between Control 2 and Test 2 is -1.38 [95%CI -1.86, -0.899].
    The p-value of the two-sided permutation t-test is 0.0, calculated for legacy purposes only. 

    The paired mean difference for repeated measures against baseline 
    between Control 3 and Test 3 is -0.255 [95%CI -0.717, 0.235].
    The p-value of the two-sided permutation t-test is 0.323, calculated for legacy purposes only. 

    The weighted-average paired mean differences is -0.0104 [95%CI -0.246, 0.222].
    The p-value of the two-sided permutation t-test is 0.941, calculated for legacy purposes only. 

    5000 bootstrap samples were taken; the confidence interval is bias-corrected and accelerated.
    Any p-value reported is the probability of observing theeffect size (or greater),
    assuming the null hypothesis ofzero difference is true.
    For each p-value, 5000 reshuffles of the control and test labels were performed.

    To get the results of all valid statistical tests, use `.mean_diff.statistical_tests`




.. code-block:: python3
  :linenos:


    mini_meta_paired.mean_diff.plot()



.. image:: _images/tutorial_101_0.png




Delta - delta function
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since v0.4.0, DABEST also supports the testing of 2 independent categorical
variables on the dataset. 

Here, we need to create another dataset for a better illustration of this 
function. 

.. code-block:: python3
  :linenos:
    
    import numpy as np
    import pandas as pd
    from scipy.stats import norm # Used in generation of populations.
    N = 20
    np.random.seed(9999) # Fix the seed so the results are replicable.

    # Create samples
    y = norm.rvs(loc=3, scale=0.4, size=N*2)

    # Add an `experiment` column as the experiment group label
    e1 = np.repeat('Control', N).tolist()
    e2 = np.repeat('Test', N).tolist()
    experiment = e1 + e2 

    # Add a `Light` column as the first variable
    # This will be the variable plotted along the horizontal aixs
    light = []
    for i in range(N):
        light.append('L1')
        light.append('L2')

    # Add a `genotype` column as the second variable
    # This will the variable controlling the color of dots for scattered 
    # plots or the color of lines for slopegraphs
    g1 = np.repeat('G1', N/2).tolist()
    g2 = np.repeat('G2', N/2).tolist()
    g3 = np.repeat('G3', N).tolist()
    genotype = g1 + g2 + g3

    # Add an `id` column for paired data plotting.
    id_col = []
    for i in range(N):
        id_col.append(i)
        id_col.append(i)

    # Combine samples and gender into a DataFrame.
    df_delta2 = pd.DataFrame({'ID'        : id_col,
                      'Light'      : light,
                       'Genotype'  : genotype, 
                       'Experiment': experiment,
                       'Y'         : y
                    })


There are 2 experiment groups, ``control`` and ``test``. ``genotype``
differs between these two groups: objects in the ``control`` group are of ``G1`` 
or ``G2`` type, while all the objects in the ``test`` group are of ``G3`` type. 
Each group has been investigated under both light status ``L1`` and ``L2`` 
situations. ``Y`` is the experiment result obtained.


.. code-block:: python3
  :linenos:


    df_delta2.head()




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
          <th>Light</th>
          <th>Genotype</th>
          <th>Experiment</th>
          <th>Y</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>L1</td>
          <td>G1</td>
          <td>Control</td>
          <td>2.793984</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0</td>
          <td>L2</td>
          <td>G1</td>
          <td>Control</td>
          <td>3.236759</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>L1</td>
          <td>G1</td>
          <td>Control</td>
          <td>3.019149</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>L2</td>
          <td>G1</td>
          <td>Control</td>
          <td>2.804638</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2</td>
          <td>L1</td>
          <td>G1</td>
          <td>Control</td>
          <td>2.858019</td>
        </tr>
      </tbody>
    </table>
    </div>



To use the delta-delta function, you can simply set ``delta2 = True`` in the 
``dabest.load()`` function. However, here ``x`` needs to be declared as a list
consisting of 2 elements rather than 1 in most of the cases. The first element
in ``x`` will be the variable plotted along the horizontal aixs, and the second
one will determine the color of dots for scattered plots or the color of lines
for slopegraphs. The ``experiment`` is required to differentiate the group 
types.

.. code-block:: python3
  :linenos:

    unpaired_delta2 = dabest.load(data = df_delta2, x = ["Light", "Genotype"], y = "Y", 
                           delta2 = True, experiment = "Experiment")


    
.. code-block:: python3
  :linenos:

    unpaired_delta2
    

.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good evening!
    The current time is Tue Aug 31 00:56:23 2021.

    Effect size(s) with 95% confidence intervals will be computed for:
    1. L2 Control minus L1 Control
    2. L2 Test minus L1 Test
    3. Test minus Control (only for mean difference)

    5000 resamples will be used to generate the effect size bootstraps.
 


.. code-block:: python3
  :linenos:


    unpaired_delta2.mean_diff.plot()



.. image:: _images/tutorial_106_0.png

As shown by the above plot, the horizonal aixs represents the ``Light`` status
and the dot color is specified by ``Genotype`` as default. The left pair of 
scattered plots is based on the ``control`` data while the right pair is based
on the ``test`` data. The mean difference bootstraps between the ``control`` 
and ``test`` group are plotted at the right bottom with a seperate y-axis from
other bootstrap plots.

You can manipulate the orders of experiment groups as well as the horizontal
axis variable by setting ``experiment_label`` and ``x1_level``.


.. code-block:: python3
  :linenos:

    unpaired_delta2_specified = dabest.load(data = df_delta2, 
                                            x = ["Light", "Genotype"], y = "Y", 
                                            delta2 = True, experiment = "Experiment",
                                            experiment_label = ["Test", "Control"],
                                            x1_level = ["L2", "L1"])


    
.. code-block:: python3
  :linenos:

    unpaired_delta2_specified.mean_diff
    

.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good morning!
    The current time is Tue Aug 31 01:15:27 2021.

    The unpaired mean difference between L2 Test and L1 Test is -0.113 [95%CI -0.529, 0.347].
    The p-value of the two-sided permutation t-test is 0.645, calculated for legacy purposes only. 

    The unpaired mean difference between L2 Control and L1 Control is 0.366 [95%CI 0.0685, 0.706].
    The p-value of the two-sided permutation t-test is 0.0514, calculated for legacy purposes only. 

    The delta-delta between Test and Control is 0.48 [95%CI -0.103, 1.02].
    The p-value of the two-sided permutation t-test is 0.133, calculated for legacy purposes only. 

    5000 bootstrap samples were taken; the confidence interval is bias-corrected and accelerated.
    Any p-value reported is the probability of observing theeffect size (or greater),
    assuming the null hypothesis ofzero difference is true.
    For each p-value, 5000 reshuffles of the control and test labels were performed.

    To get the results of all valid statistical tests, use `.mean_diff.statistical_tests`
    


.. code-block:: python3
  :linenos:


    unpaired_delta2_specified.mean_diff.plot()



.. image:: _images/tutorial_107_0.png


This delta - delta function also supports paired data.

.. code-block:: python3
  :linenos:

    paired_delta2 = dabest.load(data = df_delta2, 
                                paired = "baseline", id_col="ID",
                                x = ["Light", "Genotype"], y = "Y", 
                                delta2 = True, experiment = "Experiment")


  
.. code-block:: python3
  :linenos:


    paired_delta2.mean_diff.plot()



.. image:: _images/tutorial_108_0.png

Since the delta-delta function is only applicable to mean differences, plots 
of other effect sizes will not include a delta-delta bootstrap plot.

.. code-block:: python3
  :linenos:


    paired_delta2.median_diff.plot()



.. image:: _images/tutorial_109_0.png

Similar as the mini-meta function, you can also find all the information 
of delta - delta by assessing the attribute named ``delta_delta`` of the 
effect size object.

.. code-block:: python3
  :linenos:

    paired_delta2.mean_diff.delta_delta


.. parsed-literal::

    DABEST v0.3.9999
    ================
                
    Good morning!
    The current time is Wed Sep  1 01:44:14 2021.

    The delta-delta between Control and Test is 0.48 [95%CI -0.188, 1.17].
    The p-value of the two-sided permutation t-test is 0.236, calculated for legacy purposes only. 

    5000 bootstrap samples were taken; the confidence interval is bias-corrected and accelerated.
    Any p-value reported is the probability of observing theeffect size (or greater),
    assuming the null hypothesis ofzero difference is true.
    For each p-value, 5000 reshuffles of the control and test labels were performed.


``delta_delta`` has its own attributes, containing various information of delta - delta.
``delta_delta.to_dict()`` will return to you all the attributes in a dictionary format.


Controlling plot aesthetics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Changing the y-axes labels.

.. code-block:: python3
  :linenos:


    two_groups_unpaired.mean_diff.plot(swarm_label="This is my\nrawdata",
                                       contrast_label="The bootstrap\ndistribtions!");



.. image:: _images/tutorial_55_0.png


Color the rawdata according to another column in the dataframe.

.. code-block:: python3
  :linenos:


    multi_2group.mean_diff.plot(color_col="Gender");



.. image:: _images/tutorial_57_0.png


.. code-block:: python3
  :linenos:


    two_groups_paired_baseline.mean_diff.plot(color_col="Gender");



.. image:: _images/tutorial_58_0.png


Changing the palette used with ``custom_palette``. Any valid matplotlib
or seaborn color palette is accepted.

.. code-block:: python3
  :linenos:


    multi_2group.mean_diff.plot(color_col="Gender", custom_palette="Dark2");



.. image:: _images/tutorial_60_0.png


.. code-block:: python3
  :linenos:


    multi_2group.mean_diff.plot(custom_palette="Paired");



.. image:: _images/tutorial_61_0.png


You can also create your own color palette. Create a dictionary where
the keys are group names, and the values are valid matplotlib colors.

You can specify matplotlib colors in a `variety of
ways <https://matplotlib.org/users/colors.html>`__. Here, I demonstrate
using named colors, hex strings (commonly used on the web), and RGB
tuples.

.. code-block:: python3
  :linenos:


    my_color_palette = {"Control 1" : "blue",
                        "Test 1"    : "purple",
                        "Control 2" : "#cb4b16",     # This is a hex string.
                        "Test 2"    : (0., 0.7, 0.2) # This is a RGB tuple.
                       }

    multi_2group.mean_diff.plot(custom_palette=my_color_palette);



.. image:: _images/tutorial_63_0.png


By default, ``dabest.plot()`` will
`desaturate <https://en.wikipedia.org/wiki/Colorfulness#Saturation>`__
the color of the dots in the swarmplot by 50%. This draws attention to
the effect size bootstrap curves.

You can alter the default values with the ``swarm_desat`` and
``halfviolin_desat`` keywords.

.. code-block:: python3
  :linenos:


    multi_2group.mean_diff.plot(custom_palette=my_color_palette,
                                swarm_desat=0.75,
                                halfviolin_desat=0.25);



.. image:: _images/tutorial_65_0.png


You can also change the sizes of the dots used in the rawdata swarmplot,
and those used to indicate the effect sizes.

.. code-block:: python3
  :linenos:


    multi_2group.mean_diff.plot(raw_marker_size=3,
                                es_marker_size=12);



.. image:: _images/tutorial_67_0.png


Changing the y-limits for the rawdata axes, and for the contrast axes.

.. code-block:: python3
  :linenos:


    multi_2group.mean_diff.plot(swarm_ylim=(0, 5),
                                contrast_ylim=(-2, 2));



.. image:: _images/tutorial_69_0.png


If your effect size is qualitatively inverted (ie. a smaller value is a
better outcome), you can simply invert the tuple passed to
``contrast_ylim``.

.. code-block:: python3
  :linenos:


    multi_2group.mean_diff.plot(contrast_ylim=(2, -2),
                                contrast_label="More negative is better!");



.. image:: _images/tutorial_71_0.png


The contrast axes share the same y-limits as that of the delta - delta plot
and thus the y axis of the delta - delta plot changes as well.

.. code-block:: python3
  :linenos:


    paired_delta2.mean_diff.plot(contrast_ylim=(3, -3),
                                 contrast_label="More negative is better!");



.. image:: _images/tutorial_112_0.png


You can also change the y-limits and y-label for the delta - delta plot.

.. code-block:: python3
  :linenos:


    paired_delta2.mean_diff.plot(delta2_ylim=(3, -3),
                                 delta2_label="More negative is better!");



.. image:: _images/tutorial_111_0.png

 


You can add minor ticks and also change the tick frequency by accessing
the axes directly.

Each estimation plot produced by ``dabest`` has 2 axes. The first one
contains the rawdata swarmplot; the second one contains the bootstrap
effect size differences.

.. code-block:: python3
  :linenos:


    import matplotlib.ticker as Ticker

    f = two_groups_unpaired.mean_diff.plot()

    rawswarm_axes = f.axes[0]
    contrast_axes = f.axes[1]

    rawswarm_axes.yaxis.set_major_locator(Ticker.MultipleLocator(1))
    rawswarm_axes.yaxis.set_minor_locator(Ticker.MultipleLocator(0.5))

    contrast_axes.yaxis.set_major_locator(Ticker.MultipleLocator(0.5))
    contrast_axes.yaxis.set_minor_locator(Ticker.MultipleLocator(0.25))



.. image:: _images/tutorial_73_0.png


.. code-block:: python3
  :linenos:


    f = multi_2group.mean_diff.plot(swarm_ylim=(0,6),
                                   contrast_ylim=(-3, 1))

    rawswarm_axes = f.axes[0]
    contrast_axes = f.axes[1]

    rawswarm_axes.yaxis.set_major_locator(Ticker.MultipleLocator(2))
    rawswarm_axes.yaxis.set_minor_locator(Ticker.MultipleLocator(1))

    contrast_axes.yaxis.set_major_locator(Ticker.MultipleLocator(0.5))
    contrast_axes.yaxis.set_minor_locator(Ticker.MultipleLocator(0.25))



.. image:: _images/tutorial_74_0.png



For mini-meta plots, you can hide the weighted avergae plot by setting 
``show_mini_meta=False`` in the ``plot()`` function.

.. code-block:: python3
  :linenos:


    mini_meta_paired.mean_diff.plot(show_mini_meta=False)

.. image:: _images/tutorial_102_0.png


Similarly, you can also hide the delta-delta plot by setting 
``show_delta2=False`` in the ``plot()`` function.

.. code-block:: python3
  :linenos:


    paired_delta2.mean_diff.plot(show_delta2=False)

.. image:: _images/tutorial_113_0.png


Creating estimation plots in existing axes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Implemented in v0.2.6 by Adam Nekimken*.

``dabest.plot`` has an ``ax`` keyword that accepts any Matplotlib
``Axes``. The entire estimation plot will be created in the specified
``Axes``.

.. code-block:: python3
  :linenos:


    from matplotlib import pyplot as plt
    f, axx = plt.subplots(nrows=2, ncols=2,
                          figsize=(15, 15),
                          gridspec_kw={'wspace': 0.25} # ensure proper width-wise spacing.
                         )

    two_groups_unpaired.mean_diff.plot(ax=axx.flat[0]);

    two_groups_paired.mean_diff.plot(ax=axx.flat[1]);

    multi_2group.mean_diff.plot(ax=axx.flat[2]);

    multi_2group_paired.mean_diff.plot(ax=axx.flat[3]);



.. image:: _images/tutorial_76_0.png


In this case, to access the individual rawdata axes, use
``name_of_axes`` to manipulate the rawdata swarmplot axes, and
``name_of_axes.contrast_axes`` to gain access to the effect size axes.

.. code-block:: python3
  :linenos:


    topleft_axes = axx.flat[0]
    topleft_axes.set_ylabel("New y-axis label for rawdata")
    topleft_axes.contrast_axes.set_ylabel("New y-axis label for effect size")

    f




.. image:: _images/tutorial_78_0.png


Applying style sheets
~~~~~~~~~~~~~~~~~~~~~

*Implemented in v0.2.0*.

``dabest`` can apply `matplotlib style
sheets <https://matplotlib.org/tutorials/introductory/customizing.html>`__
to estimation plots. You can refer to this
`gallery <https://matplotlib.org/3.0.3/gallery/style_sheets/style_sheets_reference.html>`__
of style sheets for reference.

.. code-block:: python3
  :linenos:


    import matplotlib.pyplot as plt
    plt.style.use("dark_background")

.. code-block:: python3
  :linenos:


    multi_2group.mean_diff.plot();



.. image:: _images/tutorial_81_0.png
