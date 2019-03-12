#!/usr/bin/python
# -*-coding: utf-8 -*-
# Author: Joses Ho
# Email : joseshowh@gmail.com

# CONVENIENCE FUNCTIONS THAT DON'T DIRECTLY DEAL WITH PLOTTING OR
# BOOTSTRAP COMPUTATIONS ARE PLACED HERE.

def merge_two_dicts(x, y):
    """
    Given two dicts, merge them into a new dict as a shallow copy.
    Any overlapping keys in `y` will override the values in `x`.

    Taken from https://stackoverflow.com/questions/38987/
    how-to-merge-two-python-dictionaries-in-a-single-expression

    Parameters:
        x, y: dicts

    Returns:
        A dictionary containing a union of all keys in both original dicts.
    """
    z = x.copy()
    z.update(y)
    return z



def unpack_and_add(l, c):
    """Convenience function to allow me to add to an existing list
    without altering that list."""
    t = [a for a in l]
    t.append(c)
    return(t)



def print_greeting():
    from .__init__ import __version__
    import datetime as dt
    import numpy as np

    line1 = "DABEST v{}".format(__version__)
    header = "".join(np.repeat("=", len(line1)))
    spacer = "".join(np.repeat(" ", len(line1)))

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
    matching_vars = [k for k,v in globals().items() if v is obj]
    if len(matching_vars) > 0:
        return matching_vars[0]
    else:
        return ""
