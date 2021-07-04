import numpy as np
from code.util import load_pickle


def z_score_normalization(feature_data, indicator):
    """
    do z score normalization, based on equ(1)/(2)
    :param feature_data: V in equ(1), D-dimensional feature of I stocks in T timestamp, np array  with shape (I, D, T)
    :param indicator: np array with shape (D), elements in {-1, 1}
    :return:
    """
    assert feature_data.shape[1] == len(indicator)

    # cal mean and standard deviation
    V_mean = np.mean(feature_data, axis=0)  # shape (D, T)
    V_std = np.std(feature_data, axis=0) # shape (D, T)

    # cal Y in equ (1)/(2)
    Y = (feature_data - V_mean) / V_std

    # consider the directional indicator
    Y = Y * np.expand_dims(np.array(indicator), axis=1)  # broadcasting

    return Y


def return_to_rank(returns):
    """
    convert return to the rank of stokes every quarter
    :param returns: numpy array with shape shape (I, T)
    :return: numpy array with shape shape (I, T)
    """

    I, T = returns.shape

    # get the rank r_{i,t} of actual return R_{i,t}
    sorted_index = np.argsort(-returns, axis=0)  # a list of index (desc)
    r = np.zeros_like(sorted_index)  # index list to rank list

    for t in range(T):
        r[sorted_index[:, t], t] = np.arange(1, I + 1)
    # r: each column contains (1,2,3,...,)

    return r


def price_to_return(price):
    """
    convert price data to the return of stokes every quarter
    R_{i,t} = ln(P_{i,t}/P_{i,t-1})
    :param price: numpy array with shape (I, T+1)
    :return: numpy array with shape (I, T)
    """
    return np.log(price[:, 1:] / price[:, :-1])


def data_preprocess(feature_data, price_data, indicator):
    """
    main function of data preparation. generate rank_data and Y based on the input data
    :param feature_data: V in equ(1), D-dimensional feature of I stocks in T timestamp, np array  with shape (I, D, T)
    :param price_data: price of I stokes in T+1 timestamps, np array with shape (I, T+1)
    :param indicator: np array with shape (D), elements in {-1, 1}
    :return: Y, rank_data, return_data
    """

    # assertion
    print(feature_data.shape)
    print(price_data.shape)
    assert feature_data.shape[0] == price_data.shape[0]
    assert feature_data.shape[1] == indicator.shape[0]
    assert feature_data.shape[2] == price_data.shape[1] - 1

    # V->Y
    Y = z_score_normalization(feature_data, indicator)

    # price->rank
    return_data = price_to_return(price_data)
    rank_data = return_to_rank(return_data)

    # use Y_{1....t-1} and rank_data_{2...t} and return_data_{2...t}
    Y = Y[:, :, :-1]
    rank_data = rank_data[:, 1:]
    return_data = return_data[:, 1:]
    return Y, rank_data, return_data


def get_all_data():
    """
    load input data and convert them to the input of the model
    :return:
    """

    # path
    feature_path = 'data/input_data/monthly/features.pkl'
    price_path = 'data/input_data/monthly/price.pkl'
    indicator_path = 'data/input_data/indicator.pkl'

    # load
    feature_data = load_pickle(feature_path)
    price_data = load_pickle(price_path)
    indicator_list = load_pickle(indicator_path)
    print(feature_data)
    # call data_preprocess
    return data_preprocess(feature_data, price_data, indicator_list)
