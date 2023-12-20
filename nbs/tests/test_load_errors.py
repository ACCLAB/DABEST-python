import pytest
from dabest._api import load
from data.mocked_data_test_load_errors import dummy_df, N


def test_wrong_params_combinations():
    error_msg = "`proportional` and `mini_meta` cannot be True at the same time."
    with pytest.raises(ValueError) as excinfo:
        my_data = load(
            dummy_df, idx=("Control 1", "Test 1"), proportional=True, mini_meta=True
        )

    assert error_msg in str(excinfo.value)

    error_msg = (
        "If `delta2` is True. `x` parameter cannot be None. String or list expected"
    )
    with pytest.raises(ValueError) as excinfo:
        my_data = load(
            dummy_df,
            idx=("Control 1", "Test 1"),
            delta2=True,
        )
    assert error_msg in str(excinfo.value)

    error_msg = "`delta2` and `mini_meta` cannot be True at the same time."
    with pytest.raises(ValueError) as excinfo:
        my_data = load(
            dummy_df,
            x=["Control 1", "Control 1"],
            y="Test 1",
            delta2=True,
            mini_meta=True,
        )

    assert error_msg in str(excinfo.value)

    error_msg = "`proportional` and `delta` cannot be True at the same time."
    with pytest.raises(ValueError) as excinfo:
        my_data = load(
            dummy_df,
            x=["Control 1", "Control 1"],
            y="Test 1",
            delta2=True,
            proportional=True,
        )

    assert error_msg in str(excinfo.value)

    error_msg = "`idx` should not be specified when `delta2` is True.".format(N)
    with pytest.raises(ValueError) as excinfo:
        my_data = load(
            dummy_df,
            x=["Control 1", "Control 1"],
            idx=("Control 1", "Test 1"),
            delta2=True,
        )

    assert error_msg in str(excinfo.value)
