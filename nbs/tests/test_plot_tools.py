import pytest
import numpy as np
import matplotlib.pyplot as plt
from dabest.plot_tools import get_swarm_spans, width_determine, swarmplot
from data.mocked_data_test_swarmplot import dummy_df, default_swarmplot_kwargs


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


# swarmplot() UNIT TESTS
# fmt: off
@pytest.mark.parametrize("param_name, param_value, error_msg, error_type", [
    # Basic input validation checks
    ("data", None, "`data` must be a Pandas Dataframe.", ValueError),
    ("x", None, "`x` must be a string.", ValueError),
    ("y", None, "`y` must be a string.", ValueError),
    ("ax", None, "`ax` must be a Matplotlib AxesSubplot. The current `ax` is a <class 'NoneType'>", ValueError),
    ("order", 5, "`order` must be either an Iterable or None.", ValueError),
    ("hue", 5, "`hue` must be either a string or None.", ValueError),
    ("palette", None, "`palette` must be either a string indicating a color name or an Iterable.", ValueError),
    ("zorder", None, "`zorder` must be a scalar or float.", ValueError),
    ("size", None, "`size` must be a scalar or float.", ValueError),
    ("side", None, "Invalid `side`. Must be one of 'center', 'right', or 'left'.", ValueError),
    ("jitter", None, "`jitter` must be a scalar or float.", ValueError),
    ("is_drop_gutter", None, "`is_drop_gutter` must be a boolean.", ValueError),
    ("gutter_limit", None, "`gutter_limit` must be a scalar or float.", ValueError),
    ("filled", 1, "`filled` must be a boolean, list or tuple.", ValueError),

    # More thorough input validation checks
    ("x", "a", "a is not a column in `data`.", IndexError),
    ("y", "b", "b is not a column in `data`.", IndexError),
    ("hue", "c", "c is not a column in `data`.", IndexError),
    ("order", ["Control 1", "Test 2"], "Test 2 in `order` is not in the 'group' column of `data`.", IndexError),
    ("palette", " ", "`palette` cannot be an empty string. It must be either a string indicating a color name or an Iterable.", ValueError),
    ("palette", {"Control 1": " "}, "The color mapping for Control 1 in `palette` is an empty string. It must contain a color name.", ValueError),
    ("palette", {"Control 3": "black"}, "Control 3 in `palette` is not in the 'group' column of `data`.", IndexError),
    # TODO: to add palette validation testing for when color_col is hue
    ("side", "top", "Invalid `side`. Must be one of 'center', 'right', or 'left'.", ValueError),
    ("filled", [True, "a"], "All values in `filled` must be a boolean.", ValueError),
    ("filled", [True], "There are 2 unique values in `x` column in `data` but `filled` has a length of 1.", ValueError),
])
def test_swarmplot_input_error_handling(param_name, param_value, error_msg, error_type):
    with pytest.raises(error_type) as excinfo:
        my_data = swarmplot(
            data=dummy_df if param_name != "data" else param_value,
            x="group" if param_name != "x" else param_value,
            y="value" if param_name != "y" else param_value,
            ax=plt.gca() if param_name != "ax" else param_value,
            order=["Control 1", "Test 1"] if param_name != "order" else param_value,
            hue=None if param_name != "hue" else param_value,
            palette="black" if param_name != "palette" else param_value,
            zorder=1 if param_name != "zorder" else param_value,
            size=5 if param_name != "size" else param_value,
            side="center" if param_name != "side" else param_value,
            jitter=1 if param_name != "jitter" else param_value,
            filled=True if param_name != "filled" else param_value,
            is_drop_gutter=True if param_name != "is_drop_gutter" else param_value,
            gutter_limit=0.5 if param_name != "gutter_limit" else param_value,
        )

    assert error_msg in str(excinfo.value)


def test_swarmplot_warnings():
    warning_msg = (
        "{0:.1f}% of the points cannot be placed. "
        "You might want to decrease the size of the markers."
    )
    with pytest.warns(UserWarning) as warn_rec:
        my_data = swarmplot(size=100, **default_swarmplot_kwargs)

    assert warning_msg.format(10) in str(warn_rec[0].message)
    assert warning_msg.format(20) in str(warn_rec[1].message)

    warning_msg = (
        "unique values in '{0}' column in `data` "
        "and `palette` do not have the same length. Number of unique values is {1} "
        "while length of palette is {2}. The assignment of the colors in the "
        "palette will be cycled."
    )
    with pytest.warns(UserWarning) as warn_rec:
        my_data = swarmplot(palette=["black"], **default_swarmplot_kwargs)

    assert warning_msg.format("group", 2, 1) in str(warn_rec[0].message)


def test_swarmplot_order_params():
    # `order` should be able to handle customised order -> swapping of params in `order` list
    swarmplot(order=["Control 1", "Test 1"], **default_swarmplot_kwargs)
    swarmplot(order=["Test 1", "Control 1"], **default_swarmplot_kwargs)

    # `order` should be able to handle None, where it will then be autogenerated
    swarmplot(order=None, **default_swarmplot_kwargs)


def test_swarmplot_hue_params():
    swarmplot(hue="gender", **default_swarmplot_kwargs)


@pytest.mark.parametrize("hue, palette", [
    # `palette` can be a string, list, tuple or a dict
    # Testing `palette` when color of swarms is based on `x` value
    (None, "black"),
    (None, ("black", "red")),
    (None, {"Control 1": "black", "Test 1": "red"}),

    # Testing `palette` when color of swarms is based on `hue` value
    ("gender", "black"),
    ("gender", ["black", "red"]),
    ("gender", ("black", "red")),
    ("gender", {"Female": "black", "Male": "red"}),

    # Testing auto assignment of `palette` when `palette` is:
    # (list | tuple) and len(palette) != len(unique_color_groups)
    (None, ["black"]),
])
def test_swarmplot_palette_params(hue, palette):
    swarmplot(hue=hue, palette=palette, **default_swarmplot_kwargs)


def test_swarmplot_side_params():
    swarmplot(side="center", **default_swarmplot_kwargs)
    swarmplot(side="right", **default_swarmplot_kwargs)
    swarmplot(side="left", **default_swarmplot_kwargs)
