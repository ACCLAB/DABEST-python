import pytest
import pandas as pd
from dabest.plot_tools import get_swarm_spans, width_determine, error_bar, check_data_matches_labels
import numpy as np


def test_get_swarm_spans_wrong_parameters():
    error_msg = "The collection `coll` parameter cannot be None"
    with pytest.raises(ValueError) as excinfo:
        get_swarm_spans(None)

    assert error_msg in str(excinfo.value)


def test_width_determine():
    error_msg = "The `labels` parameter cannot be None"
    with pytest.raises(ValueError) as excinfo:
        width_determine(None, [])

    assert error_msg in str(excinfo.value)

    error_msg = "The `data` parameter cannot be None"
    with pytest.raises(ValueError) as excinfo:
        width_determine("some_labels", None)

    assert error_msg in str(excinfo.value)


def test_error_bar():
    data = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B'],
        'value': [1, 2, 3, 4]
    })
    error_msg = "`gap_width_percent` must be between 0 and 100."
    with pytest.raises(ValueError) as excinfo:
        error_bar(
        data=data,
        x='group',
        y='value',
        type='mean_sd',
        gap_width_percent=-10  # Invalid as it's less than 0
    )

    assert error_msg in str(excinfo.value)

    error_msg = "Invalid `method`. Must be one of 'gapped_lines', \
                         'proportional_error_bar', or 'sankey_error_bar'."
    with pytest.raises(ValueError) as excinfo:
        error_bar(
        data=data,
        x='group',
        y='value',
        type='mean_sd',
        method='invalid_method'  # Invalid as it's not one of the accepted values
    )

    assert error_msg in str(excinfo.value)

    error_msg = "Only accepted values for type are ['mean_sd', 'median_quartiles']"
    with pytest.raises(ValueError) as excinfo:
        error_bar(
        data=data,
        x='group',
        y='value',
        type='invalid_type'
    )

    assert error_msg in str(excinfo.value)


def test_check_data_matches_labels():
    wrong_labels = ['A', 'B', 'C']
    wrong_data = pd.Series(['A', 'B', 'D'])
    error_msg = "labels and data do not match."
    with pytest.raises(Exception) as excinfo:
        check_data_matches_labels(wrong_labels, wrong_data, side='left')

    assert error_msg in str(excinfo.value)