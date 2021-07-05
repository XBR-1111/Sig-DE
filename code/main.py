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
    # Y: (I, D) * T rank_data: (I) * T returns: (I) * T , here rank and returns are at next timestamp
    assert [_.shape for _ in returns] == [_.shape for _ in rank_data]  # I
    # print([_.shape[0] for _ in Y_data])
    # print([_.shape[0] for _ in rank_data])
    assert [_.shape[0] for _ in Y_data] == [_.shape[0] for _ in rank_data]  # I
    assert len(Y_data) == len(rank_data)  # T
    assert len(Y_data) == len(returns)  # T

    # get configuration of hyper parameter
    config = get_config()

    print('begin to train the Sig-DE model...')

    # there are (T - min_train_periods) cases
    T = len(Y_data)
    D = Y_data[0].shape[1]
    I = [_.shape[0] for _ in Y_data]
    min_train_periods = config.get('min_train_periods', 2)
    print('num of cases:%d' % (T - min_train_periods))

    # the portfolio returns of each case
    portfolio_returns = np.zeros(shape=T - min_train_periods)
    R1_returns = np.zeros(shape=T - min_train_periods)

    times_per_case = config.get('times_per_case', 1)

    # for each case
    for t in range(min_train_periods, T):
        # select Y_train and r_train and m
        Y_train = Y_data[:t]
        r_train = rank_data[:t]
        m = max(int(config.get('m_rate', 0.2) * I[t]), 1)

        print('begin to run on case %d...' % (t - min_train_periods + 1))

        Rs = np.zeros(shape=times_per_case)
        R1s = np.zeros(shape=times_per_case)

        for times in range(times_per_case):
            # init the model
            model = SigDE(config=config, Y=Y_train, r=r_train, silent=True)

            # run the model and return feature parameters
            feat_param = model.run()

            # measure performance of testing period
            R, R1 = measure_performance_R(feat_param, Y_data[t], returns[t], m)
            Rs[times] = R
            R1s[times] = R1

        portfolio_returns[t - min_train_periods] = Rs.mean()
        R1_returns[t - min_train_periods] = R1s.mean()
        print('portfolio_returns of the proposed method:%f' % portfolio_returns[t - min_train_periods])
        print('portfolio_returns of all candidates(R1):%f' % R1_returns[t - min_train_periods])

        print('finish running on case %d...' % (t - min_train_periods + 1))
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
    plt.savefig('./test.jpg')
    plt.show()


if __name__ == '__main__':
    train_and_predict()
