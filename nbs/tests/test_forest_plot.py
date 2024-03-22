import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dabest.forest_plot import load_plot_data, extract_plot_data, forest_plot
from data.mocked_data_test_forestplot import dummy_contrasts, default_forestplot_kwargs

def test_forest_plot_no_input_parameters():
    error_msg = "The `contrasts` parameter cannot be None"
    with pytest.raises(ValueError) as excinfo:
        forest_plot(contrasts = None)
    
    assert error_msg in str(excinfo.value)

@pytest.mark.parametrize("param_name, param_value, error_msg, error_type", [
    ("contrasts", None, "The `contrasts` parameter cannot be None", ValueError),
    ("contrasts", [], "The `contrasts` argument must be a non-empty list.", ValueError),
    ("selected_indices", "not a list or None", "The `selected_indices` must be a list of integers or `None`.", TypeError),
    ("contrast_type", 123, "The `contrast_type` argument must be a string.", TypeError),
    ("xticklabels", [123, 456], "The `xticklabels` must be a list of strings or `None`.", TypeError),
    ("effect_size", 456, "The `effect_size` argument must be a string.", TypeError),
    ("contrast_labels", ["valid", 123], "The `contrast_labels` must be a list of strings or `None`.", TypeError),
    ("ylabel", 789, "The `ylabel` argument must be a string.", TypeError),
    ("custom_palette", 123, "The `custom_palette` must be either a dictionary, list, string, or `None`.", TypeError),
    ("fontsize", "big", "`fontsize` must be an integer or float.", TypeError),
    ("marker_size", "large", "`marker_size` must be a positive integer or float.", TypeError),
    ("ci_line_width", "thick", "`ci_line_width` must be a positive integer or float.", TypeError),
    ("zero_line_width", "thin", "`zero_line_width` must be a positive integer or float.", TypeError),
    ("remove_spines", "yes", "`remove_spines` must be a boolean value.", TypeError),
    ("rotation_for_xlabels", "right", "`rotation_for_xlabels` must be an integer or float between 0 and 360.", TypeError),
    ("alpha_violin_plot", "opaque", "`alpha_violin_plot` must be a float between 0 and 1.", TypeError),
    ("horizontal", "sideways", "`horizontal` must be a boolean value.", TypeError),
    ("contrast_type", "unknown", "Invalid contrast_type: unknown. Available options: [`delta2`, `mini_meta`]", ValueError),
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
