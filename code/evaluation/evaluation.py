import numpy as np


def measure_performance_R(feat_param, Y, returns, m):
    """
    calculate R^{p}_{t+1} which is the performance of the portfolio constructed by the proposed model
    at time t+1 based on equ(4)
    :param feat_param: numpy array with shape (2*D), feature selection of the DE algorithm
    :param Y: numpy array with shape (I,D), normalised features at time t
    :param returns: numpy array with shape (I), contains the return of each stock
    :param m: number of the selected stocks
    :return:    1: a floating point number of R^{p}_{t+1} portfolio return of the proposed model
                2: a floating point number of R1 portfolio return of all candidates
    """
    # print('in measure')
    # print(Y)
    # print(returns)
    D = feat_param.shape[0] // 2
    S = np.einsum('j,ij->i', feat_param[:D] * feat_param[D:], Y)

    sorted_index = np.argsort(-S)  # a list of index(desc)
    m_stocks = sorted_index[:m]

    return returns[m_stocks].mean(), returns.mean()


def cal_MR(port_ret):
    """
    calculate MR based on equ (16)
    :param port_ret: numpy 1d array of portfolio return for each case (Logarithmic rate of return)
    :return: a floating point number of MR
    """

    return port_ret.mean()


# if __name__ == '__main__':
#     # port_ret = np.array([1, 3, 5])
#     # print(cal_MR(port_ret))
#     a = np.array([1, 2, 3, 4])
#     returns = a
#     m_stocks = np.array([0, 2])
#     print(measure_performance_R(m_stocks, returns))
