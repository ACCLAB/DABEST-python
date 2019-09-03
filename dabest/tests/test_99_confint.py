def test_paired_mean_diff_ci():    
    # See Altman et al., Statistics with Confidence: 
    # Confidence Intervals and Statistical Guidelines (Second Edition). Wiley, 2000.
    # Pg 31.
    # Added in v0.2.5.
    blood_pressure = {"before": [148, 142, 136, 134, 138, 140, 132, 144,
                                128, 170, 162, 150, 138, 154, 126, 116],
                      "after" : [152, 152, 134, 148, 144, 136, 144, 150, 
                                146, 174, 162, 162, 146, 156, 132, 126],
                     "subject_id" : np.arange(1, 17)}
    exercise_bp = pd.DataFrame(blood_pressure)


    ex_bp = load(data=exercise_bp, idx=("before", "after"), 
                 paired=True, id_col="subject_id")
    paired_mean_diff = ex_bp.mean_diff.results
    
    assert pytest.approx(3.625) == paired_mean_diff.bca_low[0]
    assert pytest.approx(9.125) == paired_mean_diff.bca_high[0]


# def test_paired_median_diff_ci():    
#     # See Altman et al., Statistics with Confidence: 
#     # Confidence Intervals and Statistical Guidelines (Second Edition). Wiley, 2000.
#     # Pg 31.
#     endorphin = {"before": [10.6, 5.2, 8.4, 9.0, 6.6, 4.6,
#                             14.1, 5.2, 4.4, 17.4, 7.2],
#                   "after" : [14.6, 15.6, 20.2, 20.9, 24.0,
#                             25.0, 35.2, 30.2, 30.0, 46.2, 37.0],
#                  "subject_id" : np.arange(1, 12)}
#     marathon = pd.DataFrame(endorphin)
# 
# 
#     endorphin_marathon = load(data=marathon, idx=("before", "after"), 
#                                paired=True, id_col="subject_id")
#     paired_median_diff = endorphin_marathon.median_diff.results
# 
#     assert pytest.approx(10.4) == paired_median_diff.bca_low[0]
