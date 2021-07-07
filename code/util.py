import pickle
import numpy as np

def load_pickle(path):
    """

    :param path:
    :return:
    """
    f = open(path, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def dump_pickle(path, data):
    """

    :param path:
    :param data:
    :return:
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def cal_ic_spearman(r_proposed, r_actual):
    """
    Spearman correlation based on equ(5), (objective func / fitness func)
    :param r_proposed: shape (I) score ranking of stock i by the proposed model at time t
    :param r_actual: shape (I) actual return ranking in the next period t+1
    :return: a floating number indicating normalized similarity
    """
    assert r_proposed.shape == r_actual.shape
    return np.cov(r_proposed, r_actual)[0, 1] / np.sqrt(np.var(r_proposed) * np.var(r_actual))


def cal_ic_pearson(S, returns):
    """
    Pearson correlation of Score and next period returns
    :param S: shape (I), scores of stock i by the proposed model at time t
    :param returns: shape (I), actual returns in the next period t+1
    :return: a floating number indicating normalized similarity
    """
    assert S.shape == returns.shape
    return np.cov(S, returns)[0, 1] / np.sqrt(np.var(S) * np.var(returns))