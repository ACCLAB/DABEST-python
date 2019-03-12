def create_dummy_dataset(seed=None, n=30, base_mean=0,
                         plus_minus=5, expt_groups=7,
                         scale_means=1., scale_std=1.):
    """
    Creates a dummy dataset for plotting.
    Returns the seed used to generate the random numbers,
    the maximum possible difference between mean differences,
    and the dataset itself.
    """
    import numpy as np
    import scipy as sp
    import pandas as pd

    # Set a random seed.
    if seed is None:
        random_seed = np.random.randint(low=1, high=1000, size=1)[0]
    else:
        if isinstance(seed, int):
            random_seed = seed
        else:
            raise TypeError('{} is not an integer.'.format(seed))

    # Generate a set of random means
    np.random.seed(random_seed)
    MEANS = np.repeat(base_mean, expt_groups) + \
            np.random.uniform(base_mean-plus_minus, base_mean+plus_minus,
                              expt_groups) * scale_means
    SCALES = np.random.random(size=expt_groups) * scale_std

    max_mean_diff = np.ptp(MEANS)

    dataset = list()
    for i, m in enumerate(MEANS):
        pop = sp.stats.norm.rvs(loc=m, scale=SCALES[i], size=10000)
        sample = np.random.choice(pop, size=n, replace=False)
        dataset.append(sample)

    df = pd.DataFrame(dataset).T
    df["idcol"] = pd.Series(range(1, n+1))
    df.columns = [str(c) for c in df.columns]

    return random_seed, max_mean_diff, df



def get_swarm_yspans(coll, round_result=False, decimals=12):
    """
    Given a matplotlib Collection, will obtain the y spans
    for the collection. Will return None if this fails.
    Modified from `get_swarm_spans` in plot_tools.py.
    """
    import numpy as np
    _, y = np.array(coll.get_offsets()).T
    try:
        if round_result:
            return np.around(y.min(), decimals), np.around(y.max(),decimals)
        else:
            return y.min(), y.max()
    except ValueError:
        return None
