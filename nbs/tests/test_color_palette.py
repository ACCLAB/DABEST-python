"""
Tests for `get_color_palette`, which decides how the raw data, the slopegraph
and the bootstrap distributions are coloured.

Regression coverage for issue #218: combining `color_col` with a
`custom_palette` on a paired plot used to raise a `KeyError`, because the
palette is keyed by the `color_col` categories while the bootstraps were
still being coloured by the x-axis group.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dabest import load
from dabest.misc_tools import get_color_palette


N = 20
IDX = ("Control 1", "Test 1")
ALL_PLOT_GROUPS = list(IDX)
COLOR_GROUPS = ["Female", "Male"]


@pytest.fixture
def df():
    np.random.seed(9999)
    return pd.DataFrame(
        {
            "Control 1": np.random.normal(3, 0.4, N),
            "Test 1": np.random.normal(3.5, 0.5, N),
            "Gender": ["Female"] * (N // 2) + ["Male"] * (N // 2),
            "ID": range(1, N + 1),
        }
    )


@pytest.fixture
def plot_data(df):
    return df.melt(
        id_vars=["Gender", "ID"], var_name="group", value_name="value"
    )


def make_plot_kwargs(color_col=None, custom_palette=None):
    return {
        "color_col": color_col,
        "custom_palette": custom_palette,
        "empty_circle": False,
        "raw_desat": 1.0,
        "contrast_desat": 1.0,
    }


def call(plot_data, color_col=None, custom_palette=None, show_pairs=True):
    return get_color_palette(
        plot_kwargs=make_plot_kwargs(color_col, custom_palette),
        plot_data=plot_data,
        xvar="group",
        show_pairs=show_pairs,
        idx=IDX,
        all_plot_groups=ALL_PLOT_GROUPS,
        delta2=False,
        proportional=False,
    )


def test_paired_color_col_with_list_palette(plot_data):
    # The palette is keyed by the `color_col` categories, so the bootstraps
    # must not be coloured by the x-axis group.
    (color_col, bootstraps_color_by_group, n_groups, _, _,
     plot_palette_raw, plot_palette_contrast, _) = call(
        plot_data, color_col="Gender", custom_palette=["red", "blue"]
    )

    assert color_col == "Gender"
    assert bootstraps_color_by_group is False
    assert n_groups == 2
    assert list(plot_palette_raw.keys()) == COLOR_GROUPS
    assert list(plot_palette_contrast.keys()) == COLOR_GROUPS


def test_paired_color_col_with_dict_palette(plot_data):
    palette = {"Female": "red", "Male": "blue"}
    (_, bootstraps_color_by_group, _, _, _,
     plot_palette_raw, _, _) = call(
        plot_data, color_col="Gender", custom_palette=palette
    )

    assert bootstraps_color_by_group is False
    assert list(plot_palette_raw.keys()) == COLOR_GROUPS


def test_paired_dict_palette_missing_color_raises(plot_data):
    with pytest.raises(ValueError) as excinfo:
        call(plot_data, color_col="Gender", custom_palette={"Female": "red"})

    assert "missing colors" in str(excinfo.value)
    assert "Male" in str(excinfo.value)


def test_paired_custom_palette_without_color_col(plot_data):
    # Issue #207: without a `color_col`, a custom palette colours the paired
    # groups and the bootstraps follow the x-axis group.
    (color_col, bootstraps_color_by_group, _, _, _,
     plot_palette_raw, _, _) = call(plot_data, custom_palette=["red", "blue"])

    assert color_col is None
    assert bootstraps_color_by_group is True
    assert list(plot_palette_raw.keys()) == ALL_PLOT_GROUPS


def test_paired_without_custom_palette(plot_data):
    _, bootstraps_color_by_group, _, _, _, _, _, _ = call(plot_data)
    assert bootstraps_color_by_group is False


def test_unpaired_color_col_with_custom_palette(plot_data):
    _, bootstraps_color_by_group, _, _, _, plot_palette_raw, _, _ = call(
        plot_data,
        color_col="Gender",
        custom_palette=["red", "blue"],
        show_pairs=False,
    )

    assert bootstraps_color_by_group is False
    assert list(plot_palette_raw.keys()) == COLOR_GROUPS


@pytest.mark.parametrize(
    "custom_palette",
    [["red", "blue"], {"Female": "red", "Male": "blue"}, "Dark2"],
)
@pytest.mark.parametrize("paired", ["baseline", "sequential"])
def test_paired_plot_with_color_col_and_custom_palette(df, custom_palette, paired):
    # Issue #218: this used to raise `KeyError: 'Test 1'`.
    loaded = load(df, idx=IDX, paired=paired, id_col="ID")
    fig = loaded.mean_diff.plot(color_col="Gender", custom_palette=custom_palette)
    assert fig is not None
    plt.close(fig)
