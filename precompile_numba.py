# import numpy as np
# from dabest import effsize, ci_2g

# def main():
#     def test_ci_2g():
#         data = np.array([1, 2, 3, 4, 5])
#         is_paired_flag = 'baseline'
#         is_paired_none = None
#         x0_len = 5
#         x1_len = 5
#         resamples = 10
#         random_seed = 42

#         x1 = np.array([1, 2, 3, 4, 5])
#         x2 = np.array([2, 3, 4, 5, 6])
#         x3 = np.array([3, 4, 5, 6, 7])
#         x4 = np.array([4, 5, 6, 7, 8])
#         pooled_sd = 1
#         rng_seed = 42
        
#         ci_2g.create_jackknife_indexes(data)

#         ci_2g.create_repeated_indexes(data)

#         ci_2g.bootstrap_indices(is_paired_flag, x0_len, x1_len, resamples, random_seed)
#         ci_2g.bootstrap_indices(is_paired_none, x0_len, x1_len, resamples, random_seed)

#         ci_2g.delta2_bootstrap_loop(x1, x2, x3, x4, resamples, pooled_sd, rng_seed, is_paired_flag)

#         z = 0.5
#         bias = 0.1
#         acceleration = 0.2
#         ci_2g._compute_quantile(z, bias, acceleration)

#         control_var = 2
#         control_N = 10
#         test_var = 3
#         test_N = 15
#         ci_2g.calculate_group_var(control_var, control_N, test_var, test_N)

#         ci_2g.calculate_weighted_delta(x1, x2)


#     def test_es():
#         control = np.array([1, 2, 3, 4, 5])
#         test = np.array([2, 3, 4, 5, 6])
#         effsize.cohens_d(control, test)
#         effsize.cohens_d(control, test, is_paired='baseline')

#         effsize._compute_standardizers(control, test)
#         effsize.weighted_delta(control, test)

#         control = np.array([1, 0, 1, 0, 1])
#         test = np.array([0, 1, 0, 1, 0])
#         effsize.cohens_h(control, test)

#     test_ci_2g()
#     test_es()


# if __name__ == '__main__':
#     main()