import numpy as np
from code.util import load_pickle


def z_score_normalization(feature_data, indicator):
    """
    do z score normalization, based on equ(1)/(2)
    :param feature_data: V in equ(1), D-dimensional feature of I stocks in T timestamp,
                    a list of numpy array with shape (I, D), with length T
    :param indicator: np array with shape (D), elements in {-1, 1}
    :return: a list of numpy array with shape (I, D), with length T
    """
    assert feature_data[0].shape[1] == indicator.shape[0]

    Y = list()

    for array in feature_data:

        # cal mean and standard deviation
        v_mean = np.mean(array, axis=0)  # shape (D)
        v_std = np.std(array, axis=0)  # shape (D)

        # cal Y in equ (1)/(2)
        y = (array - v_mean) / v_std

        # consider the directional indicator
        y = y * np.array(indicator)  # broadcasting
        Y.append(y)

    return Y


def return_to_rank(returns):
    """
    convert return to the rank of stokes every quarter
    :param returns: a list of numpy array with shape (I) of length T
    :return: a list of numpy array with shape (I) of length T
    """
    ranks = []
    for return_i in returns:

        I = return_i.shape[0]

        # get the rank r_{i,t} of actual return R_{i,t}
        sorted_index = np.argsort(-return_i, axis=0)  # a list of index (desc)
        r = np.zeros_like(sorted_index)  # index list to rank list
        r[sorted_index] = np.arange(1, I + 1)
        ranks.append(r)

    return ranks


# def price_to_return(price):
#     """
#     convert price data to the return of stokes every quarter
#     R_{i,t} = ln(P_{i,t}/P_{i,t-1})
#     :param price_data: price of I stokes in T+1 timestamps,
#                     a list of numpy array with shape (I), with length T+1
#     :return: numpy array with shape (I, T)
#     """
#
#
#     return np.log(price[:, 1:] / price[:, :-1])


def data_preprocess(feature_data, return_data, indicator):
    """
    main function of data preparation. generate rank_data and Y based on the input data
    :param feature_data: V in equ(1), D-dimensional feature of I stocks in T timestamp,
                    a list of numpy array with shape (I, D), with length T
    :param return_data: return of I stokes in T timestamps,
                    a list of numpy array with shape (I), with length T
    :param indicator: np array with shape (D), elements in {-1, 1}
    :return: Y, rank_data, return_data
    """

    # assertion
    assert feature_data[0].shape[1] == indicator.shape[0]  # D
    assert len(feature_data) == len(return_data)  # T

    # V->Y
    Y = z_score_normalization(feature_data, indicator)

    # return->rank
    rank_data = return_to_rank(return_data)

    # # use Y_{1....t-1} and rank_data_{2...t} and return_data_{2...t}
    # Y = Y[:-1]
    # rank_data = rank_data[1:]
    # return_data = return_data[1:]
    return Y, rank_data, return_data


def get_all_data(mode='input'):
    """
    load input data and convert them to the input of the model
    :return:
    """

    # path
    if mode == 'input':
        feature_path = 'data/input_data/monthly/features.pkl'
        # price_path = 'data/input_data/monthly/price.pkl'
        indicator_path = 'data/input_data/indicators.pkl'
        return_path = 'data/input_data/monthly/returns.pkl'
    else:
        feature_path = 'test/features.pkl'
        indicator_path = 'test/indicators.pkl'
        return_path = 'test/returns.pkl'

    # load
    feature_data = load_pickle(feature_path)
    # price_data = load_pickle(price_path)
    indicator_list = load_pickle(indicator_path)
    return_data = load_pickle(return_path)

    # call data_preprocess
    return data_preprocess(feature_data, return_data, indicator_list)
