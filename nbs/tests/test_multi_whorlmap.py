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
    two_group_contrast_1,
    two_group_contrast_2,
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
    ("n", "large", "'str' object cannot be interpreted as an integer", TypeError),
    ("n", -5, "negative dimensions are not allowed", ValueError),
    
    # sort_by validation
    ("sort_by", 123, "'int' object is not subscriptable", TypeError),
    
    # cmap validation
    ("cmap", 123, "'int' object is not callable", TypeError),
    ("cmap", ["vlag"], "Invalid RGBA argument: 'vlag'", ValueError),
    
    # vmax/vmin validation
    ("vmax", "high", "unsupported operand type(s) for -: 'str' and 'int'", TypeError),
    ("vmin", "low", "unsupported operand type(s) for -: 'int' and 'str'", TypeError),
    
    # chop_tail validation
    ("chop_tail", ['str'], "unsupported operand type(s) for /: 'list' and 'int'", TypeError),
    
    # ax validation
    ("ax", "axes", "'str' object has no attribute 'spines'", AttributeError),
    ("ax", 123, "'int' object has no attribute 'spines'", AttributeError),
    
])

def test_whorlmap_input_validation(param_name, param_value, error_msg, error_type):
    """Test input validation for whorlmap() parameters."""
    valid_inputs = default_whorlmap_kwargs.copy()
    valid_inputs[param_name] = param_value
    
    with pytest.raises(error_type) as excinfo:
        whorlmap(**valid_inputs)
    
    assert error_msg in str(excinfo.value)

