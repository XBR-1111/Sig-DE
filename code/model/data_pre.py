import numpy as np
from code.util import load_pickle
import os


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
        print(array)
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


def data_preprocess(feature_data, return_data, indicator, market_cap_data):
    """
    main function of data preparation. generate rank_data and Y based on the input data
    :param feature_data: V in equ(1), D-dimensional feature of I stocks in T timestamp,
                    a list of numpy array with shape (I, D), with length T
    :param return_data: return of I stokes in T timestamps,
                    a list of numpy array with shape (I), with length T
    :param indicator: np array with shape (D), elements in {-1, 1}
    :param market_cap_data:
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
    return Y, rank_data, return_data, market_cap_data


def get_all_data(num_feat_timestamps, padding_or_ignore, mode='small'):
    """
    load input data and convert them to the input of the model
    :return:
    """

    # path
    if mode == 'full':
        feature_path = 'data/input_data/monthly/features.pkl'
        indicator_path = 'data/input_data/indicators.pkl'
        return_path = 'data/input_data/monthly/returns.pkl'
        market_cap_path = 'data/input_data/monthly/market_caps.pkl'
        stock_names_path = 'data/input_data/monthly/stock_names.pkl'
        zz500_path = 'data/input_data/monthly/zz500.pkl'
        hs300_path = 'data/input_data/monthly/hs300.pkl'
    elif mode == 'small':
        feature_path = 'data/input_data/monthly/features_s.pkl'
        indicator_path = 'data/input_data/indicators_s.pkl'
        return_path = 'data/input_data/monthly/returns_s.pkl'
        market_cap_path = 'data/input_data/monthly/market_caps_s.pkl'
        stock_names_path = 'data/input_data/monthly/stock_names_s.pkl'
        zz500_path = 'data/input_data/monthly/zz500_s.pkl'
        hs300_path = 'data/input_data/monthly/hs300_s.pkl'
    else:
        assert False
        # feature_path = 'test/features.pkl'
        # indicator_path = 'test/indicators.pkl'
        # return_path = 'test/returns.pkl'
        # market_cap_path = 'test/market_caps.pkl'
        # stock_names_path = 'test/stock_names.pkl'

    # load
    feature_data = load_pickle(feature_path)
    return_data = load_pickle(return_path)
    market_cap_data = load_pickle(market_cap_path)
    stock_names = load_pickle(stock_names_path)
    zz500_data = load_pickle(zz500_path)
    hs300_data = load_pickle(hs300_path)

    # if no indicator file, generate one
    if os.path.exists(indicator_path):
        indicator_list = load_pickle(indicator_path)
    else:
        indicator_list = np.array([1 for _ in range(feature_data[0].shape[1])])

    # call data_preprocess
    Y, rank_data, return_data, market_cap_data = \
        data_preprocess(feature_data, return_data, indicator_list, market_cap_data)

    # deal with time sequence of features
    if padding_or_ignore == 'padding':
        # new Y
        Y_new = []

        for i in range(num_feat_timestamps - 1, len(Y)):
            Y_new_i = [] # with shape (I, D * num_feat_timestamps)
            Y_i = Y[i]  # shape (I,D)
            Y_new_i.append(Y_i)
            stock_names_i = stock_names[i]

            for j in range(1, num_feat_timestamps):
                Y_i_j_new = np.zeros_like(Y_i)  # shape (I,D)
                Y_i_j = Y[i-j]
                stock_names_i_j = stock_names[i-j]

                for index, stock in enumerate(stock_names_i):
                    if stock in stock_names_i_j:
                        Y_i_j_new[index, :] = Y_i_j[np.where(stock_names_i_j == stock), :]
                    else:
                        Y_i_j_new[index, :] = Y_new_i[0][index, :]

                Y_new_i.insert(0, Y_i_j_new)

            Y_new_i = np.concatenate(Y_new_i, axis=1)
            Y_new.append(Y_new_i)
        rank_data_new = rank_data[num_feat_timestamps-1:]
        return_data_new = return_data[num_feat_timestamps-1:]
        market_cap_data_new = market_cap_data[num_feat_timestamps-1:]
    else:
        Y_new = []
        # not complete yet
        assert False

    return Y_new, rank_data_new, return_data_new, market_cap_data_new, zz500_data, hs300_data
