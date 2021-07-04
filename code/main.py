from model.Sig_DE import SigDE
from evaluation.evaluation import measure_performance_R, cal_MR
from config.config import get_config
import numpy as np
from model.data_pre import get_all_data
import matplotlib.pyplot as plt
from util import dump_pickle

def train_and_predict():

    print('preparing data...')

    # get data
    Y_data, rank_data, returns = get_all_data()
    # print(Y_data)
    # print(rank_data)
    # print(returns)  # true

    # Y: (I, D, T) rank_data: (I, T) returns: (I, T) , here rank and returns are at next timestamp
    assert returns.shape == rank_data.shape
    assert Y_data.shape[0] == rank_data.shape[0] and Y_data.shape[2] == rank_data.shape[1]

    # get configuration of hyper parameter
    config = get_config()

    print('begin to train the Sig-DE model...')

    # there are (T - min_train_periods) cases
    I, D, T = Y_data.shape
    min_train_periods = config.get('min_train_periods', 2)
    print('num of cases:%d' % (T - min_train_periods))

    # the portfolio returns of each case
    portfolio_returns = np.zeros(shape=T - min_train_periods)
    R1_returns = np.zeros(shape=T - min_train_periods)
    random_returns = np.zeros(shape=T - min_train_periods)
    # for each case
    for t in range(min_train_periods, T):
        # select Y_train and r_train
        Y_train = Y_data[:, :, :t]
        r_train = rank_data[:, :t]

        # init the model
        model = SigDE(config=config, Y=Y_train, r=r_train)

        # run the model and return selected stokes and the feature parameters
        print('begin to run on case %d...' % (t - min_train_periods + 1))
        m_stocks, feat_param = model.run()
        # print('selected stocks:', m_stocks)
        # print(feat_param)
        print('finish running on case %d...' % (t - min_train_periods + 1))

        # measure performance of testing period
        m = max(int(config.get('m_rate', 0.2) * I), 1)
        R, R1 = measure_performance_R(feat_param, Y_data[:, :, t], returns[:, t], m)
        print('portfolio_returns of the proposed method:%f' % R)
        print('portfolio_returns of all candidates(R1):%f' % R1)
        # R_random = measure_performance_R(np.random.randint(0, I, size=m_stocks.shape), returns[:, t])
        # print('portfolio_returns of random pick:%f' % R_random)
        portfolio_returns[t - min_train_periods] = R
        R1_returns[t - min_train_periods] = R1
        # random_returns[t - min_train_periods] = R_random
        print('--------------------------------------------------------------')

    MR = cal_MR(portfolio_returns)
    R1_MR = cal_MR(R1_returns)
    dump_pickle('show result/portfolio_returns.pkl', portfolio_returns)
    dump_pickle('show result/R1_returns.pkl', R1_returns)
    print('MR:%f' % MR)
    print('R1:%f' % R1_MR)
    plt.figure()
    plt.plot(portfolio_returns, label="R", color="#F08080")
    plt.plot(R1_returns, label="R1", color="#0B7093")
    plt.legend()
    plt.savefig('./test2.jpg')
    plt.show()



if __name__ == '__main__':
    train_and_predict()
