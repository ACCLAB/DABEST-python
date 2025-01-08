import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dabest.forest_plot import forest_plot
from data.mocked_data_test_forestplot import default_forestplot_kwargs

def test_forest_plot_no_input_parameters():
    error_msg = "The `data` argument must be a non-empty list of dabest objects."
    with pytest.raises(ValueError) as excinfo:
        forest_plot(data = None)
    
    assert error_msg in str(excinfo.value)

@pytest.mark.parametrize("param_name, param_value, error_msg, error_type", [
    ("data", [], "The `data` argument must be a non-empty list of dabest objects.", ValueError),
    ("idx", 123, "`idx` must be a tuple or list of integers.", TypeError),
    ("ax", "axes", "The `ax` must be a `matplotlib.axes.Axes` instance or `None`.", TypeError),
    ("effect_size", 456, "The `effect_size` argument must be a string and please choose from the following effect sizes: `mean_diff`, `median_diff`, `cliffs_delta`, `cohens_d`, and `hedges_g`.", TypeError),
    ("horizontal", "sideways", "`horizontal` must be a boolean value.", TypeError),
    ("marker_size", "large", "`marker_size` must be a positive integer or float.", TypeError),
    ("custom_palette", 123, "The `custom_palette` must be either a dictionary, list, string, or `None`.", TypeError),
    ("halfviolin_alpha", "opaque", "`halfviolin_alpha` must be a float between 0 and 1.", TypeError),
    ("halfviolin_desat", "yes", "`halfviolin_desat` must be a float between 0 and 1 or an int (1).", TypeError),
    ("labels", ["valid", 123], "The `labels` must be a list of strings or `None`.", TypeError),
    ("labels", ['valid', 'valid'], "`labels` must match the number of `data` provided.", ValueError),
    ("labels_fontsize", "big", "`labels_fontsize` must be an integer or float.", TypeError),
    ("labels_rotation", "right", "`labels_rotation` must be an integer or float between 0 and 360.", TypeError),
    ("title", 123, "The `title` argument must be a string.", TypeError),
    ("title_fontsize", "big", "`title_fontsize` must be an integer or float.", TypeError),
    ("ylabel", 789, "The `ylabel` argument must be a string.", TypeError),
    ("ylabel_fontsize", "big", "`ylabel_fontsize` must be an integer or float.", TypeError),
    ("ylim", "auto", "`ylim` must be a tuple or list of two floats.", TypeError),
    ("ylim", [0, 1, 2], "`ylim` must be a tuple or list of two floats.", ValueError),
    ("yticks", "auto", "`yticks` must be a tuple or list of floats.", TypeError),
    ("yticklabels", "auto", "`yticklabels` must be a tuple or list of strings.", TypeError),
    ("yticklabels", [532, 123], "`yticklabels` must be a list of strings.", TypeError),
    ("remove_spines", "yes", "`remove_spines` must be a boolean value.", TypeError),
])

def test_forest_plot_input_error_handling(param_name, param_value, error_msg, error_type):
    # Setup: Define a base set of valid inputs to forest_plot
    valid_inputs = default_forestplot_kwargs.copy()

    # Replace the tested parameter with the invalid value
    valid_inputs[param_name] = param_value

    # Perform the test
    with pytest.raises(error_type) as excinfo:
        forest_plot(**valid_inputs)
    
    # Check the error message
    assert error_msg in str(excinfo.value)
