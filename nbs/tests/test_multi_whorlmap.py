"""
Unit tests for whorlmap() function in multi.py.

Tests input validation, visualization parameters, and return types
for the whorlmap spiral heatmap visualization.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from dabest.multi import whorlmap, combine
from data.mocked_data_test_multi import (
    default_whorlmap_kwargs,
    valid_multi_contrast_1d,
    valid_multi_contrast_2d,
    valid_multi_contrast_mixed,
    standard_contrast_1,
    standard_contrast_2,
    delta2_contrast_1,
    delta2_contrast_2
)



def test_whorlmap_no_multi_contrast():
    """Test that whorlmap raises error when multi_contrast is None."""
    with pytest.raises(AttributeError):
        whorlmap(multi_contrast=None)


def test_whorlmap_invalid_multi_contrast_type():
    """Test that whorlmap raises error with invalid multi_contrast type."""
    with pytest.raises(AttributeError):
        whorlmap(multi_contrast="not_a_multicontrast")


@pytest.mark.parametrize("param_name, param_value, error_msg, error_type", [
    # n parameter validation
    ("n", "large", "object cannot be interpreted as an integer", TypeError),
    ("n", -5, "All axis/value parameters must be positive", ValueError),
    ("n", 0, "All axis/value parameters must be positive", ValueError),
    ("n", [21], "object cannot be interpreted as an integer", TypeError),
    
    # sort_by validation
    ("sort_by", "invalid", "'str' object is not subscriptable", TypeError),
    ("sort_by", 123, "'int' object is not subscriptable", TypeError),
    
    # cmap validation
    ("cmap", 123, "Colormap 123 is not recognized", ValueError),
    ("cmap", ["vlag"], "Colormap ['vlag'] is not recognized", ValueError),
    
    # vmax/vmin validation
    ("vmax", "high", "Cannot cast array data", TypeError),
    ("vmin", "low", "Cannot cast array data", TypeError),
    ("vmax", [10], "Cannot cast array data", TypeError),
    ("vmin", [-10], "Cannot cast array data", TypeError),
    
    # Boolean parameter validation
    ("reverse_neg", "yes", "object cannot be interpreted as an integer", TypeError),
    ("reverse_neg", 1, "object cannot be interpreted as an integer", TypeError),
    ("abs_rank", "yes", "object cannot be interpreted as an integer", TypeError),
    ("abs_rank", 1, "object cannot be interpreted as an integer", TypeError),
    
    # chop_tail validation
    ("chop_tail", "some", "must be real number, not str", TypeError),
    ("chop_tail", [5], "must be real number, not list", TypeError),
    ("chop_tail", -5, "cannot be negative", ValueError),
    ("chop_tail", 101, "must be between 0 and 100", ValueError),
    
    # ax validation
    ("ax", "axes", "`ax` must be a `matplotlib.axes.Axes` instance or `None`", TypeError),
    ("ax", 123, "`ax` must be a `matplotlib.axes.Axes` instance or `None`", TypeError),
    
    # fig_size validation
    ("fig_size", "large", "`fig_size` must be a tuple or list of two positive numbers", TypeError),
    ("fig_size", 10, "`fig_size` must be a tuple or list of two positive numbers", TypeError),
    ("fig_size", [10], "`fig_size` must have exactly 2 elements", ValueError),
    ("fig_size", [10, 5, 3], "`fig_size` must have exactly 2 elements", ValueError),
    ("fig_size", [-5, 10], "`fig_size` values must be positive", ValueError),
    
    # whorlmap_title validation
    ("whorlmap_title", 123, "`whorlmap_title` must be a string or None", TypeError),
    ("whorlmap_title", ["title"], "`whorlmap_title` must be a string or None", TypeError),
    
    # heatmap_kwargs validation
    ("heatmap_kwargs", "options", "`heatmap_kwargs` must be a dictionary or None", TypeError),
    ("heatmap_kwargs", [{"cmap": "viridis"}], "`heatmap_kwargs` must be a dictionary or None", TypeError),
])
def test_whorlmap_input_validation(param_name, param_value, error_msg, error_type):
    """Test input validation for whorlmap() parameters."""
    valid_inputs = default_whorlmap_kwargs.copy()
    valid_inputs[param_name] = param_value
    
    with pytest.raises(error_type) as excinfo:
        whorlmap(**valid_inputs)
    
    assert error_msg in str(excinfo.value)

