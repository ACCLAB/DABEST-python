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

idx_msg1 = "The `idx` argument must have the same length as the number of dabest objects. "
idx_msg2 = "E.g., If two dabest objects are supplied, there should be two lists within `idx`. "
idx_msg3 = "E.g., `idx` = [[1,2],[0,1]]."
    
@pytest.mark.parametrize("param_name, param_value, error_msg, error_type", [
    ("data", [], "The `data` argument must be a non-empty list of dabest objects.", ValueError),
    ("idx", 123, "`idx` must be a tuple or list of integers.", TypeError),
    ("idx", ((0,1),(0,1),(0,1),(0,1),(0,1)), idx_msg1+idx_msg2+idx_msg3, ValueError),
    ("ax", "axes", "The `ax` must be a `matplotlib.axes.Axes` instance or `None`.", TypeError),
    ("fig_size", "huge", "`fig_size` must be a tuple or list of two positive integers.", TypeError),
    ("effect_size", 456, "The `effect_size` argument must be a string and please choose from the following effect sizes: 'mean_diff', 'median_diff', 'cohens_d', 'cohens_h', 'cliffs_delta', 'hedges_g', 'delta_g'.", TypeError),
    ("ci_type", 'linear', "`ci_type` must be either 'bca' or 'pct'.", TypeError),
    ("horizontal", "sideways", "`horizontal` must be a boolean value.", TypeError),
    ("marker_size", "large", "`marker_size` must be a positive integer or float.", TypeError),
    ("custom_palette", 123, "The `custom_palette` must be either a dictionary, list, string, or `None`.", TypeError),
    ("custom_palette", "test_palette", "The specified `custom_palette` test_palette is not a recognized Matplotlib palette.", ValueError),
    ("contrast_alpha", "opaque", "`contrast_alpha` must be a float between 0 and 1.", TypeError),
    ("contrast_desat", "yes", "`contrast_desat` must be a float between 0 and 1 or an int (1).", TypeError),
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
    ("reference_band", "yes", "`reference_band` must be a list/tuple of indices (ints).", TypeError),
    ("reference_band", [0.1, 0.5], "`reference_band` must be a list/tuple of indices (ints).", TypeError),
    ("reference_band", [10,], "Index [10] chosen is out of range for the contrast objects.", ValueError),
    ("delta_text", "auto", "`delta_text` must be a boolean value.", TypeError),
    ("contrast_bars", "auto", "`contrast_bars` must be a boolean value.", TypeError),
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
