# CONVENIENCE FUNCTIONS THAT DON'T DIRECTLY DEAL WITH PLOTTING OR
# BOOTSTRAP COMPUTATIONS ARE PLACED HERE.

def merge_two_dicts(x, y):
    """
    Given two dicts, merge them into a new dict as a shallow copy.
    Any overlapping keys in `y` will override the values in `x`.

    Taken from https://stackoverflow.com/questions/38987/
    how-to-merge-two-python-dictionaries-in-a-single-expression

    Keywords:
        x, y: dicts

    Returns:
        A dictionary containing a union of all keys in both original dicts.
    """
    z = x.copy()
    z.update(y)
    return z
