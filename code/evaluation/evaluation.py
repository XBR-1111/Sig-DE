import numpy as np
from code.util import cal_ic_spearman, cal_ic_pearson, dump_pickle


def eval_6(port_ret, TC, N_Y, MAR):
    # T
    T = len(port_ret)

    # APR Annualized Percentage Rate
    A_T = np.array([((port_ret - TC)[:t + 1]).mean() for t in range(T)])
    APR_T = A_T * N_Y
    APR = APR_T[-1]

    # AVOL Annualized Volatility
    V = port_ret.std()
    AVOL = V * np.sqrt(N_Y)

    # ASR Annualized Sharpe Ratio
    ASR = APR / AVOL

    # MDD  Maximum DrawDown
    MDD = \
        np.array(
            [
                np.array([(APR_T[t] - APR_T[tao]) / APR_T[t] for t in range(tao + 1)]).max() for tao in range(T)
            ]
        ).max()

    # CR Calmar Ratio
    CR = APR / MDD

    # DDR Downside Deviation Ratio
    DDR = APR / np.sqrt(np.square(np.minimum(port_ret, MAR)).mean())

    return APR, AVOL, ASR, MDD, CR, DDR


class Evaluation:
    def __init__(self, config):
        # params
        self.TC = config.get('TC', 0.001)  # transaction cost = 0.1%
        self.N_Y = config.get('N_Y', 12)  # number of holding periods in a year.default: monthly
        self.MAR = config.get('MAR', 0)  # minimum acceptable return. default:0

        # IC
        self.IC_pearson = list()
        self.IC_spearman = list()

        # equal weight portfolio returns
        self.port_ret_equal_weight = list()  # portfolio returns
        self.APR_equal_weight = 0
        self.AVOL_equal_weight = 0
        self.ASR_equal_weight = 0
        self.MDD_equal_weight = 0
        self.CR_equal_weight = 0
        self.DDR_equal_weight = 0

        # market cap weight
        self.port_ret_market_cap = list()  # portfolio returns
        self.APR_market_cap = 0
        self.AVOL_market_cap = 0
        self.ASR_market_cap = 0
        self.MDD_market_cap = 0
        self.CR_market_cap = 0
        self.DDR_market_cap = 0

        # times average
        self.port_ret_equal_weight_t = list()
        self.port_ret_market_cap_t = list()
        self.IC_pearson_t = list()
        self.IC_spearman_t = list()

    def case_begin(self):
        """
        mark the beginning of a case
        :return:
        """
        self.port_ret_equal_weight_t = list()
        self.port_ret_market_cap_t = list()
        self.IC_pearson_t = list()
        self.IC_spearman_t = list()

    def case_end(self):
        """
        mark the end of a case
        :return:
        """
        self.port_ret_equal_weight.append(np.array(self.port_ret_equal_weight_t).mean())
        self.port_ret_market_cap.append(np.array(self.port_ret_market_cap_t).mean())
        self.IC_pearson.append(np.array(self.IC_pearson_t).mean())
        self.IC_spearman.append(np.array(self.IC_spearman_t).mean())
        print("port_ret_equal_weight: % f" % self.port_ret_equal_weight[-1])
        print("port_ret_market_cap: % f" % self.port_ret_market_cap[-1])

    def eval_per_time(self, feat_param, Y, returns, m, market_caps, rank_act):
        """
        evaluation for every output of the Sig-DE model
        :param feat_param: numpy array with shape (2*D), feature selection of the DE algorithm
        :param Y: numpy array with shape (I,D), normalised features at time t
        :param returns: numpy array with shape (I), contains the return of each stock
        :param m: number of the selected stocks
        :param market_caps: np.array with shape (I), market caps at time t
        :param rank_act: np.array with shape (I), actual rank at time t
        """

        D = feat_param.shape[0] // 2
        I = Y.shape[0]

        # Score
        S = np.einsum('j,ij->i', feat_param[:D] * feat_param[D:], Y)

        # rank
        sorted_index = np.argsort(-S)  # a list of index(desc)
        rank = np.zeros_like(sorted_index)
        rank[sorted_index] = np.arange(1, I + 1)

        # cal ICs
        self.IC_pearson_t.append(cal_ic_pearson(S, returns))
        self.IC_spearman_t.append(cal_ic_spearman(rank, rank_act))

        # port_ret
        m_stock_indexes = sorted_index[:m]
        m_returns = returns[m_stock_indexes]
        m_market_caps = market_caps[m_stock_indexes]

        self.port_ret_equal_weight_t.append(m_returns.mean())
        self.port_ret_market_cap_t.append(np.dot(m_returns, np.exp(m_market_caps/m_market_caps.mean()) / np.exp(m_market_caps/m_market_caps.mean()).sum()))
        # self.port_ret_market_cap_t.append(np.dot(m_returns, m_market_caps / m_market_caps.sum()))

    def final_eval(self):
        """
        evaluation after all cases
        """

        # equal weight evaluations
        self.port_ret_equal_weight = np.array(self.port_ret_equal_weight)
        self.APR_equal_weight, self.AVOL_equal_weight, self.ASR_equal_weight, \
            self.MDD_equal_weight, self.CR_equal_weight, self.DDR_equal_weight = \
            eval_6(self.port_ret_equal_weight, self.TC, self.N_Y, self.MAR)

        # market cap evaluations
        self.port_ret_market_cap = np.array(self.port_ret_market_cap)
        self.APR_market_cap, self.AVOL_market_cap, self.ASR_market_cap, \
            self.MDD_market_cap, self.CR_market_cap, self.DDR_market_cap = \
            eval_6(self.port_ret_market_cap, self.TC, self.N_Y, self.MAR)

    def print_evals(self):
        """
        print evaluations
        :return:
        """
        print('----------------------------------')
        print('IC evaluations:')
        print('average IC pearson: %f' % np.array(self.IC_pearson).mean())
        print('IC_spearman: %f' % np.array(self.IC_spearman).mean())

        print('----------------------------------')
        print('equal weight evaluations:')
        print('average portfolio return: %f' % np.array(self.port_ret_equal_weight).mean())
        print('APR: %f' % self.APR_equal_weight)
        print('AVOL: %f' % self.AVOL_equal_weight)
        print('ASR: %f' % self.ASR_equal_weight)
        print('MDD: %f' % self.MDD_equal_weight)
        print('CR: %f' % self.CR_equal_weight)
        print('DDR: %f' % self.DDR_equal_weight)

        print('----------------------------------')
        print('market cap evaluations:')
        print('average portfolio return: %f' % np.array(self.port_ret_market_cap).mean())
        print('APR: %f' % self.APR_market_cap)
        print('AVOL: %f' % self.AVOL_market_cap)
        print('ASR: %f' % self.ASR_market_cap)
        print('MDD: %f' % self.MDD_market_cap)
        print('CR: %f' % self.CR_market_cap)
        print('DDR: %f' % self.DDR_market_cap)

    def dump_evals(self, path):
        """
        generate a dictionary and dump it with pickle
        :param path:
        :return:
        """
        eval_dct = dict()
        eval_dct['IC_pearson'] = self.IC_pearson
        eval_dct['IC_spearman'] = self.IC_spearman

        eval_dct['port_ret_equal_weight'] = self.port_ret_equal_weight
        eval_dct['APR_equal_weight'] = self.APR_equal_weight
        eval_dct['AVOL_equal_weight'] = self.AVOL_equal_weight
        eval_dct['ASR_equal_weight'] = self.ASR_equal_weight
        eval_dct['MDD_equal_weight'] = self.MDD_equal_weight
        eval_dct['CR_equal_weight'] = self.CR_equal_weight
        eval_dct['DDR_equal_weight'] = self.DDR_equal_weight

        eval_dct['port_ret_market_cap'] = self.port_ret_market_cap
        eval_dct['APR_market_cap'] = self.APR_market_cap
        eval_dct['AVOL_market_cap'] = self.AVOL_market_cap
        eval_dct['ASR_market_cap'] = self.ASR_market_cap
        eval_dct['MDD_market_cap'] = self.MDD_market_cap
        eval_dct['CR_market_cap'] = self.CR_market_cap
        eval_dct['DDR_market_cap'] = self.DDR_market_cap

        dump_pickle(path, eval_dct)

# def measure_performance_R(feat_param, Y, returns, m):
#     """
#     calculate R^{p}_{t+1} which is the performance of the portfolio constructed by the proposed model
#     at time t+1 based on equ(4)
#     :param feat_param: numpy array with shape (2*D), feature selection of the DE algorithm
#     :param Y: numpy array with shape (I,D), normalised features at time t
#     :param returns: numpy array with shape (I), contains the return of each stock
#     :param m: number of the selected stocks
#     :return:    1: a floating point number of R^{p}_{t+1} portfolio return of the proposed model
#                 2: a floating point number of R1 portfolio return of all candidates
#     """
#     D = feat_param.shape[0] // 2
#     S = np.einsum('j,ij->i', feat_param[:D] * feat_param[D:], Y)
#
#     sorted_index = np.argsort(-S)  # a list of index(desc)
#     m_stocks = sorted_index[:m]
#
#     return returns[m_stocks].mean(), returns.mean()


# if __name__ == '__main__':
#     # port_ret = np.array([1, 3, 5])
#     # print(cal_MR(port_ret))
#     a = np.array([1, 2, 3, 4])
#     returns = a
#     m_stocks = np.array([0, 2])
#     print(measure_performance_R(m_stocks, returns))
