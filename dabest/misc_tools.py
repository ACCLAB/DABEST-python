# CONVENIENCE FUNCTIONS THAT DON'T DIRECTLY DEAL WITH PLOTTING OR
# BOOTSTRAP COMPUTATIONS ARE PLACED HERE.

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy.
    Taken from https://stackoverflow.com/questions/38987/
    how-to-merge-two-python-dictionaries-in-a-single-expression"""
    z = x.copy()
    z.update(y)
    return z
