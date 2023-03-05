.. _Controlling Plot Aesthetics Tutorial:

=====================================
Tutorial: Controlling Plot Aesthetics
=====================================


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