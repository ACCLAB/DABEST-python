def create_demo_dataset(seed=9999, N=20):
    
    import numpy as np
    import pandas as pd
    from scipy.stats import norm # Used in generation of populations.

    np.random.seed(9999) # Fix the seed so the results are replicable.
    # pop_size = 10000 # Size of each population.

    # Create samples
    c1 = norm.rvs(loc=3, scale=0.4, size=N)
    c2 = norm.rvs(loc=3.5, scale=0.75, size=N)
    c3 = norm.rvs(loc=3.25, scale=0.4, size=N)

    t1 = norm.rvs(loc=3.5, scale=0.5, size=N)
    t2 = norm.rvs(loc=2.5, scale=0.6, size=N)
    t3 = norm.rvs(loc=3, scale=0.75, size=N)
    t4 = norm.rvs(loc=3.5, scale=0.75, size=N)
    t5 = norm.rvs(loc=3.25, scale=0.4, size=N)
    t6 = norm.rvs(loc=3.25, scale=0.4, size=N)


    # Add a `gender` column for coloring the data.
    females = np.repeat('Female', N/2).tolist()
    males = np.repeat('Male', N/2).tolist()
    gender = females + males

    # Add an `id` column for paired data plotting. 
    id_col = pd.Series(range(1, N+1))

    # Combine samples and gender into a DataFrame.
    df = pd.DataFrame({'Control 1' : c1,     'Test 1' : t1,
                       'Control 2' : c2,     'Test 2' : t2,
                       'Control 3' : c3,     'Test 3' : t3,
                       'Test 4'    : t4,     'Test 5' : t5, 'Test 6' : t6,
                       'Gender'    : gender, 'ID'  : id_col
                      })
                      
    return df



def create_demo_dataset_rm(seed=9999, N=20):
    
    import numpy as np
    import pandas as pd
    from scipy.stats import norm # Used in generation of populations.

    np.random.seed(9999) # Fix the seed so the results are replicable.
    # pop_size = 10000 # Size of each population.

    # Create samples
    timepoint0 = norm.rvs(loc=3, scale=0.4, size=N)
    timepoint1 = norm.rvs(loc=3.5, scale=0.75, size=N)
    timepoint2 = norm.rvs(loc=3.25, scale=0.4, size=N)
    timepoint3 = norm.rvs(loc=3.5, scale=0.5, size=N)
    timepoint4 = norm.rvs(loc=2.5, scale=0.6, size=N)
    timepoint5 = norm.rvs(loc=3, scale=0.75, size=N)
    timepoint6 = norm.rvs(loc=3.5, scale=0.75, size=N)
    timepoint7 = norm.rvs(loc=3.25, scale=0.4, size=N)
    timepoint8 = norm.rvs(loc=3.25, scale=0.4, size=N)


    # Add a `gender` column for coloring the data.
    grp1 = np.repeat('Group 1', N/2).tolist()
    grp2 = np.repeat('Group 2', N/2).tolist()
    grp = grp1 + grp2

    # Add an `id` column for paired data plotting. 
    id_col = pd.Series(range(1, N+1))

    # Combine samples and gender into a DataFrame.
    df = pd.DataFrame({'Time Point 0' : timepoint0,     
                       'Time Point 1' : timepoint1,
                       'Time Point 2' : timepoint2,     
                       'Time Point 3' : timepoint3,
                       'Time Point 4' : timepoint4,     
                       'Time Point 5' : timepoint5,
                       'Time Point 6' : timepoint6,     
                       'Time Point 7' : timepoint7, 
                       'Time Point 8' : timepoint8,
                       'Group'        : grp, 
                       'ID'           : id_col
                      })
                      
    return df


def create_demo_dataset_delta(seed=9999, N=20):
    
    import numpy as np
    import pandas as pd
    from scipy.stats import norm # Used in generation of populations.

    np.random.seed(seed) # Fix the seed so the results are replicable.
    # pop_size = 10000 # Size of each population.

    from scipy.stats import norm # Used in generation of populations.

    # Create samples
    y = norm.rvs(loc=3, scale=0.4, size=N*4)
    y[N:2*N] = y[N:2*N]+1
    y[2*N:3*N] = y[2*N:3*N]-0.5

    # Add drug column
    t1 = np.repeat('Placebo', N*2).tolist()
    t2 = np.repeat('Drug', N*2).tolist()
    treatment = t1 + t2 

    # Add a `rep` column as the first variable for the 2 replicates of experiments done
    rep = []
    for i in range(N*2):
        rep.append('Rep1')
        rep.append('Rep2')

    # Add a `genotype` column as the second variable
    wt = np.repeat('W', N).tolist()
    mt = np.repeat('M', N).tolist()
    wt2 = np.repeat('W', N).tolist()
    mt2 = np.repeat('M', N).tolist()


    genotype = wt + mt + wt2 + mt2

    # Add an `id` column for paired data plotting.
    id = list(range(0, N*2))
    id_col = id + id 


    # Combine all columns into a DataFrame.
    df = pd.DataFrame({'ID'        : id_col,
                      'Rep'      : rep,
                       'Genotype'  : genotype, 
                       'Treatment': treatment,
                       'Y'         : y
                    })
    return df


def create_demo_prop_dataset(seed=9999, N=40):
    import numpy as np
    import pandas as pd

    np.random.seed(9999)  # Fix the seed so the results are replicable.
    # Create samples
    n = 1
    c1 = np.random.binomial(n, 0.2, size=N)
    c2 = np.random.binomial(n, 0.2, size=N)
    c3 = np.random.binomial(n, 0.8, size=N)

    t1 = np.random.binomial(n, 0.5, size=N)
    t2 = np.random.binomial(n, 0.2, size=N)
    t3 = np.random.binomial(n, 0.3, size=N)
    t4 = np.random.binomial(n, 0.4, size=N)
    t5 = np.random.binomial(n, 0.5, size=N)
    t6 = np.random.binomial(n, 0.6, size=N)

    # Add a `gender` column for coloring the data.
    females = np.repeat('Female', N / 2).tolist()
    males = np.repeat('Male', N / 2).tolist()
    gender = females + males

    # Add an `id` column for paired data plotting.
    id_col = pd.Series(range(1, N + 1))

    # Combine samples and gender into a DataFrame.
    df = pd.DataFrame({'Control 1': c1, 'Test 1': t1,
                       'Control 2': c2, 'Test 2': t2,
                       'Control 3': c3, 'Test 3': t3,
                       'Test 4': t4, 'Test 5': t5, 'Test 6': t6,
                       'Gender': gender, 'ID': id_col
                       })

    return df


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
        
        
        
# def create_dummy_dataset(seed=None, n=30, base_mean=0,
#                          plus_minus=5, expt_groups=7,
#                          scale_means=1., scale_std=1.):
#     """
#     Creates a dummy dataset for plotting.
#     Returns the seed used to generate the random numbers,
#     the maximum possible difference between mean differences,
#     and the dataset itself.
#     """
#     import numpy as np
#     import scipy as sp
#     import pandas as pd
# 
#     # Set a random seed.
#     if seed is None:
#         random_seed = np.random.randint(low=1, high=1000, size=1)[0]
#     else:
#         if isinstance(seed, int):
#             random_seed = seed
#         else:
#             raise TypeError('{} is not an integer.'.format(seed))
# 
#     # Generate a set of random means
#     np.random.seed(random_seed)
#     MEANS = np.repeat(base_mean, expt_groups) + \
#             np.random.uniform(base_mean-plus_minus, base_mean+plus_minus,
#                               expt_groups) * scale_means
#     SCALES = np.random.random(size=expt_groups) * scale_std
# 
#     max_mean_diff = np.ptp(MEANS)
# 
#     dataset = list()
#     for i, m in enumerate(MEANS):
#         pop = sp.stats.norm.rvs(loc=m, scale=SCALES[i], size=10000)
#         sample = np.random.choice(pop, size=n, replace=False)
#         dataset.append(sample)
# 
#     df = pd.DataFrame(dataset).T
#     df["idcol"] = pd.Series(range(1, n+1))
#     df.columns = [str(c) for c in df.columns]
# 
#     return random_seed, max_mean_diff, df