import pytest
from dabest.plot_tools import get_swarm_spans, width_determine
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
