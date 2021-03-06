from model.Sig_DE import SigDE
from evaluation.evaluation import Evaluation
from config.config import get_config
from model.data_pre import get_all_data
from result.show_result import show_result


def train_and_predict(mode="small"):
    # get configuration of hyper parameter
    config = get_config()
    min_train_periods = config.get('min_train_periods', 2)
    times_per_case = config.get('times_per_case', 1)
    num_feat_timestamps = config.get('num_feat_timestamps', 1)
    padding_or_ignore = config.get('padding_or_ignore', 'padding')
    sliding_window = config.get('sliding_window', False)
    window_size = config.get('window_size', 5)

    # get data
    # Y: (I, D) * T rank_data,returns,market_caps: (I) * T , here rank and returns are at next timestamp
    print('preparing data...')
    Y_data, rank_data, returns, market_caps, zz500, hs300 = get_all_data(num_feat_timestamps, padding_or_ignore, mode)
    T = len(Y_data)
    I = [_.shape[0] for _ in Y_data]
    assert [_.shape for _ in returns] == [_.shape for _ in rank_data]  # returns = rank_data
    assert [_.shape for _ in returns] == [_.shape for _ in market_caps]  # returns = market_caps
    assert [_.shape[0] for _ in Y_data] == [_.shape[0] for _ in rank_data]  # I
    assert len(Y_data) == len(rank_data)  # T

    print('begin to train the Sig-DE model...')
    print('num of cases:%d' % (T - min_train_periods))

    # evaluation
    my_eval = Evaluation(config=config)

    # for each case
    for t in range(min_train_periods, T):

        # select Y_train and r_train and m
        if sliding_window:
            Y_train = Y_data[max(t - window_size, 0):t]
            r_train = rank_data[max(t - window_size, 0):t]
            returns_train = returns[max(t - window_size, 0):t]
            zz500_train = zz500[max(t - window_size, 0):t]
            hs300_train = hs300[max(t - window_size, 0):t]
        else:
            Y_train = Y_data[:t]
            r_train = rank_data[:t]
            returns_train = returns[:t]
            zz500_train = zz500[:t]
            hs300_train = hs300[:t]
        m = max(int(config.get('m_rate', 0.2) * I[t]), 1)

        print('begin to run on case %d...' % (t - min_train_periods + 1))

        # eval
        my_eval.case_begin()

        for times in range(times_per_case):
            # init the model
            model = SigDE(config=config, Y=Y_train, r=r_train, returns=returns_train, zz500=zz500_train, hs300=hs300_train, m=m, silent=True)

            # run the model and return feature parameters
            feat_param = model.run()

            # evaluation
            my_eval.eval_per_time(feat_param, Y_data[t], returns[t], m, market_caps[t], rank_data[t])

        # eval
        my_eval.case_end()

        print('finish running on case %d...' % (t - min_train_periods + 1))
        print('--------------------------------------------------------------')

    my_eval.final_eval()
    my_eval.print_evals()
    my_eval.dump_evals('./result/evals.pkl')

    show_result(T - min_train_periods, mode)


if __name__ == '__main__':
    train_and_predict('full')
