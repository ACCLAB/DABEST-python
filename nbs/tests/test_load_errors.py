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
            delta2=True
        )
    assert error_msg in str(excinfo.value)

    error_msg = "`delta2` and `mini_meta` cannot be True at the same time."
    with pytest.raises(ValueError) as excinfo:
        my_data = load(
            dummy_df,
            x=["Control 1", "Control 1"],
            y="Test 1",
            delta2=True,
            mini_meta=True
        )

    assert error_msg in str(excinfo.value)

    error_msg = "`idx` should not be specified when `delta2` is True.".format(N)
    with pytest.raises(ValueError) as excinfo:
        my_data = load(
            dummy_df,
            x=["Control 1", "Control 1"],
            idx=("Control 1", "Test 1"),
            delta2=True
        )

    assert error_msg in str(excinfo.value)

    error_msg = "`id_col` must be specified if `paired` is assigned with a not NoneType value."
    with pytest.raises(IndexError) as excinfo:
        my_data = load(
            dummy_df, idx=("Control 1", "Test 1"), paired="baseline"
        )

    assert error_msg in str(excinfo.value)

    error_msg = "`delta2` is True but `y` is not indicated."
    with pytest.raises(ValueError) as excinfo:
        my_data = load(
            dummy_df,
            x=["Control 1", "Control 1"],
            delta2=True
        )


def test_param_validations():
    error_msg = "`idx` contains duplicated groups. Please remove any duplicates and try again.".format(N)
    with pytest.raises(ValueError) as excinfo:
        my_data = load(
            dummy_df, idx=("Control 1", "Control 1")
        )

    assert error_msg in str(excinfo.value)

    err0 = "Groups are repeated across tuples,"
    err1 = " or a tuple has repeated groups in it."
    err2 = " Please remove any duplicates and try again."
    error_msg = err0 + err1 + err2
    with pytest.raises(ValueError) as excinfo:
        my_data = load(
            dummy_df, idx=(("Control 1", "Control 1", "Test 1"), ("Control 2", "Test 2"))
        )

    assert error_msg in str(excinfo.value)

    wrong_idx = ("Control 1", ("Control 1", "Test 1"))
    error_msg = "There seems to be a problem with the idx you " "entered--{}.".format(wrong_idx)
    with pytest.raises(ValueError) as excinfo:
        my_data = load(
            dummy_df, idx=wrong_idx
        )

    assert error_msg in str(excinfo.value)

    wrong_paired = 'not_valid'
    error_msg = "'{}' assigned for `paired` is not valid. Please use either 'baseline' or 'sequential'.".format(wrong_paired)
    with pytest.raises(ValueError) as excinfo:
        my_data = load(
            dummy_df, idx=("Control 1", "Test 1"), paired=wrong_paired, id_col="ID"
        )

    assert error_msg in str(excinfo.value)


    wrong_id_col = 'not_valid'
    error_msg = "`id_col` was given as '{}'; however, '{}' is not a column in `data`.".format(wrong_id_col, wrong_id_col)
    with pytest.raises(IndexError) as excinfo:
        my_data = load(
            dummy_df, idx=("Control 1", "Test 1"), paired="baseline", id_col=wrong_id_col
        )

    assert error_msg in str(excinfo.value)

    wrong_idx_mmeta = ("Control 1", "Test 1", "Test 2")
    err0 = "`mini_meta` is True, but `idx` ({})".format(wrong_idx_mmeta)
    err1 = "does not contain exactly 2 unique columns."
    error_msg = err0 + err1
    with pytest.raises(ValueError) as excinfo:
        my_data = load(
            dummy_df, idx=wrong_idx_mmeta, mini_meta=True
        )

    assert error_msg in str(excinfo.value)

    wrong_idx_mmeta = (("Control 1", "Test 1", "Test 2"), ("Control 1", "Control 2", "Test 3"))
    err0 = "`mini_meta` is True, but `idx` ({})".format(wrong_idx_mmeta)
    err1 = "does not contain exactly 2 unique columns."
    error_msg = err0 + err1
    with pytest.raises(ValueError) as excinfo:
        my_data = load(
            dummy_df, idx=wrong_idx_mmeta, mini_meta=True
        )

    assert error_msg in str(excinfo.value)

    wrong_x = ["Control 1", "Control 1", "Control 2"]
    error_msg = "`delta2` is True but the number of variables indicated by `x` is {}.".format(len(wrong_x))
    with pytest.raises(ValueError) as excinfo:
        my_data = load(
            dummy_df, x=wrong_x, y="Test 1", delta2=True
        )

    assert error_msg in str(excinfo.value)

    wrong_x = ["Control 4", "Control 5"]
    error_msg = "is not a column in `data`. Please check."
    with pytest.raises(IndexError) as excinfo:
        my_data = load(
            dummy_df, x=wrong_x, y="Test 1", delta2=True
        )

    assert error_msg in str(excinfo.value)

    wrong_y = "Test 3"
    error_msg = "is not a column in `data`. Please check."
    with pytest.raises(IndexError) as excinfo:
        my_data = load(
            dummy_df, x=["Control 1", "Control 2"], y=wrong_y, delta2=True
        )

    assert error_msg in str(excinfo.value)

    wrong_experiment = "not_valid"
    error_msg = "is not a column in `data`. Please check."
    with pytest.raises(IndexError) as excinfo:
        my_data = load(
            dummy_df,
            x=["Control 1", "Control 1"],
            y="Test 1",
            delta2=True,
            experiment=wrong_experiment
        )

    assert error_msg in str(excinfo.value)

    #TODO experiment and experiment_label are different

    wrong_experiment_label = ["A", "B", "C"]
    error_msg = "`experiment_label` does not have a length of 2."
    with pytest.raises(ValueError) as excinfo:
        my_data = load(
            dummy_df,
            x=["Control 1", "Control 1"],
            y="Test 1",
            delta2=True,
            experiment="Control 1",
            experiment_label=wrong_experiment_label
        )

    assert error_msg in str(excinfo.value)

    wrong_x1_level = "not_valid"
    error_msg = "`x1_level` does not have a length of 2."
    with pytest.raises(ValueError) as excinfo:
        my_data = load(
            dummy_df,
            x=["Control 1", "Control 1"],
            y="Test 1",
            delta2=True,
            experiment="Control 1",
            x1_level=wrong_x1_level
        )

    assert error_msg in str(excinfo.value)