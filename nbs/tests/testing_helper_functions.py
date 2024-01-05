import pytest
from typing import Callable, Any, Dict


def check_exceptions(fn: Callable, *args: Any, **kwargs: Dict) -> None:
    """
    Check if a function raises an exception and fail the test if it does.

    Parameters
    ----------
    fn : Callable
        The function to be called and checked for exceptions.
    *args : Any
        Positional arguments to be passed to the function.
    **kwargs : Dict
        Keyword arguments to be passed to the function.

    Raises
    ------
    pytest.fail:
        If the function raises any exception, the test fails with a descriptive message.
    """
    try:
        fn(*args, **kwargs)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")
