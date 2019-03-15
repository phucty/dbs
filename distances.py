import numpy as np
from scipy import stats
from numpy import percentile
import io_worker


def get_transformation(ori_values, trans_size=100):
    if (len(ori_values)) < 1:
        return ori_values
    trans_values = ori_values
    trans_values.sort()
    len_trans_values = len(trans_values)
    trans_values = np.array([trans_values[int(len_trans_values * 1. * i / trans_size)] for i in range(trans_size)])
    return trans_values


def ks_test(s1, s2):
    if len(s1) > 1 and len(s2) > 1:
        d, p_value = stats.ks_2samp(s1, s2)
        return d, p_value
    return 0, 0


def t_test(s1, s2):
    if len(s1) > 1 and len(s2) > 1:
        t, p_value = stats.ttest_ind(s1, s2)
        return t, p_value
    return 0, 0


def u_test_hist(s1, s2):
    if len(s1) > 1 and len(s2) > 1:
        if s2[-1] != 0 and s1[-1] != 0:
            hist_s1, _ = np.histogram(s1, bins=100)
            hist_s2, _ = np.histogram(s2, bins=100)
            u, p_value = stats.mannwhitneyu(hist_s1, hist_s2)
            return u, p_value
    return 0, 0


def u_test(s1, s2):
    if len(s1) > 1 and len(s2) > 1:
        u, p_value = stats.mannwhitneyu(s1, s2)
        return u, p_value
    return 0, 0


def distance_num_jaccard(s1, s2):
    inf_value = 1
    if len(s1) > 1 and len(s2) > 1:
        max1 = percentile(s1, 75)
        min1 = percentile(s1, 25)
        max2 = percentile(s2, 75)
        min2 = percentile(s2, 25)
        max3 = max(max1, max2)
        min3 = min(min1, min2)
        min4 = min(max1, max2)
        max4 = max(min1, min2)
        if min2 > max1 or min1 > max2:
            return inf_value
        elif max3 == min3:
            return inf_value
        else:
            result = 1 - (min4 - max4) * 1.0 / (max3 - min3)
            return result
    return inf_value


def distance_ks_test_p(s1, s2):
    return 1 - ks_test(s1, s2)[1]


def distance_ks_test_d(s1, s2):
    return ks_test(s1, s2)[0]


def distance_t_test_p(s1, s2):
    return 1 - t_test(s1, s2)[1]


def distance_t_test_t(s1, s2):
    return t_test(s1, s2)[0]


def distance_u_test_hist_p(s1, s2):
    return 1 - u_test_hist(s1, s2)[1]


def distance_u_test_hist_u(s1, s2):
    return u_test_hist(s1, s2)[0]


def distance_u_test_p(s1, s2):
    return 1 - u_test(s1, s2)[1]


def distance_u_test_u(s1, s2):
    return u_test(s1, s2)[0]


def get_dsl_vector(s1, s2):
    return (distance_num_jaccard(s1, s2),
            distance_ks_test_p(s1, s2),
            distance_u_test_hist_p(s1, s2))


def distance_dsl(s1, s2, model, feature_extraction_func):
    feature_vector = feature_extraction_func(s1, s2)
    distance = model.predict_proba([feature_vector])
    return 1 - distance[0][1]


def distance_numerical(s1, s2, distance_function):
    try:
        import time
        if "1" in distance_function:
            res_dis = np.linalg.norm(s1 - s2, ord=1)
        elif "2" in distance_function:
            res_dis = np.linalg.norm(s1 - s2, ord=2)
        else:
            res_dis = np.linalg.norm(s1 - s2, ord=np.inf)
        return res_dis
    except Exception as message:
        io_worker.print_status(message, show_on_screen=False)
        return 0
