# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/API/misc_tools.ipynb.

# %% auto 0
__all__ = ['merge_two_dicts', 'unpack_and_add', 'print_greeting', 'get_varname']

# %% ../nbs/API/misc_tools.ipynb 4
import datetime as dt
from numpy import repeat

# %% ../nbs/API/misc_tools.ipynb 5
def merge_two_dicts(
    x: dict, y: dict
) -> dict:  # A dictionary containing a union of all keys in both original dicts.
    """
    Given two dicts, merge them into a new dict as a shallow copy.
    Any overlapping keys in `y` will override the values in `x`.

    Taken from [here](https://stackoverflow.com/questions/38987/
    how-to-merge-two-python-dictionaries-in-a-single-expression)

    """
    z = x.copy()
    z.update(y)
    return z


def unpack_and_add(l, c):
    """Convenience function to allow me to add to an existing list
    without altering that list."""
    t = [a for a in l]
    t.append(c)
    return t


def print_greeting():
    """
    Generates a greeting message based on the current time, along with the version information of DABEST.

    This function dynamically generates a greeting ('Good morning', 'Good afternoon', 'Good evening')
    based on the current system time. It also retrieves and displays the version of DABEST (Data Analysis
    using Bootstrap-Coupled ESTimation). The message includes a header with the DABEST version and the
    current time formatted in a user-friendly manner.

    Returns:
    str: A formatted string containing the greeting message, DABEST version, and current time.
    """
    from .__init__ import __version__

    line1 = "DABEST v{}".format(__version__)
    header = "".join(repeat("=", len(line1)))
    spacer = "".join(repeat(" ", len(line1)))

    now = dt.datetime.now()
    if 0 < now.hour < 12:
        greeting = "Good morning!"
    elif 12 < now.hour < 18:
        greeting = "Good afternoon!"
    else:
        greeting = "Good evening!"

    current_time = "The current time is {}.".format(now.ctime())

    return "\n".join([line1, header, spacer, greeting, current_time])


def get_varname(obj):
    matching_vars = [k for k, v in globals().items() if v is obj]
    if len(matching_vars) > 0:
        return matching_vars[0]
    return ""

def unique_with_types(lst):
    """
    Given a list, return a new list with only the unique elements of the original list and preserve
    the type of each element.
    """
    unique_list = []
    seen_types = {}
    for item in lst:
        # Create a type-specific key
        key = (type(item), item)
        if key not in seen_types:
            seen_types[key] = True
            unique_list.append(item)
    return unique_list
