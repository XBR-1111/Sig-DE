import numpy as np
import matplotlib.pyplot as plt
from code.util import load_pickle
from code.evaluation.evaluation import eval_6

def show_result(num, mode):
    """

    :param num: num of cases
    :return:
    """

    # get_data
    eval_dct = load_pickle('./result/evals.pkl')
    print(eval_dct)
    port_ret_equal_weight = eval_dct['port_ret_equal_weight']
    port_ret_market_cap = eval_dct['port_ret_market_cap']
    if mode == 'full':
        zz500_path = './data/input_data/monthly/zz500.pkl'
        hs300_path = './data/input_data/monthly/hs300.pkl'
    else:
        zz500_path = './data/input_data/monthly/zz500_s.pkl'
        hs300_path = './data/input_data/monthly/hs300_s.pkl'

    zz500 = load_pickle(zz500_path)
    hs300 = load_pickle(hs300_path)
    zz500 = zz500[zz500.shape[0]-num:]
    hs300 = hs300[hs300.shape[0]-num:]
    assert port_ret_equal_weight.shape == port_ret_market_cap.shape
    assert zz500.shape == hs300.shape
    assert zz500.shape == port_ret_equal_weight.shape

    # accu port_ret_equal_weight port_ret_market_cap zz500 hs300
    accu_ew, accu_mc = np.zeros(zz500.shape[0] + 1), np.zeros(zz500.shape[0] + 1)
    accu_zz500, accu_hs300 = np.zeros(zz500.shape[0] + 1), np.zeros(zz500.shape[0] + 1)
    accu_ew[0], accu_mc[0], accu_zz500[0], accu_hs300[0] = 1, 1, 1, 1
    for i in range(1, accu_ew.shape[0]):
        accu_ew[i] = accu_ew[i-1] + port_ret_equal_weight[i-1]
        accu_mc[i] = accu_mc[i-1] + port_ret_market_cap[i-1]
        accu_zz500[i] = accu_zz500[i-1] + zz500[i-1]
        accu_hs300[i] = accu_hs300[i-1] + hs300[i-1]

    # figure 1
    plt.figure()
    plt.plot(accu_ew, label="proposed method(ew)")
    plt.plot(accu_mc, label="proposed method(sm)")
    plt.plot(accu_zz500, label="zz500")
    plt.plot(accu_hs300, label="hs300")
    if mode == 'full':
        plt.title("accumulative return from 2005 July to 2020 December")  # 图形标题
    else:
        plt.title("accumulative return from 2017 Jan to 2020 December")  # 图形标题
    plt.xlabel("period")  # x轴名称
    plt.ylabel("accumulative return")  # y 轴名称
    plt.legend()
    plt.savefig('./figure1.jpg')
    plt.show()

    # figure 2
    plt.figure()
    plt.plot(accu_ew-accu_zz500, label="hedge_ret_500(ew)")
    plt.plot(accu_mc-accu_zz500, label="hedge_ret_500(sm)")
    plt.plot(accu_ew-accu_hs300, label="hedge_ret_300(ew)")
    plt.plot(accu_mc-accu_hs300, label="hedge_ret_300(sm)")
    if mode == 'full':
        plt.title("hedge_ret from 2005 July to 2020 December")  # 图形标题
    else:
        plt.title("hedge_ret from 2017 Jan to 2020 December")  # 图形标题
    plt.xlabel("period")  # x轴名称
    plt.ylabel("accumulative return")  # y 轴名称
    plt.legend()
    plt.savefig('./figure2.jpg')
    plt.show()


def eval_zz500_hs300():
    zz500 = load_pickle('../data/input_data/monthly/zz500.pkl')
    hs300 = load_pickle('../data/input_data/monthly/hs300.pkl')
    print(eval_6(zz500, 0.001, 12, 0))
    print(eval_6(hs300, 0.001, 12, 0))


# def show_difference():
#     R = load_pickle('portfolio_returns.pkl')
#     R1 = load_pickle('R1_returns.pkl')
#     a = R-R1
#     b = np.zeros_like(a)
#     plt.figure()
#     plt.plot(a, label="proposed model", color="#F08080")
#     plt.plot(b, color="#0B7093")
#     plt.title("accumulative return differences")  # 图形标题
#     plt.xlabel("period")  # x轴名称
#     plt.ylabel("R-R1")  # y 轴名称
#     # plt.legend()
#     plt.savefig('./test4.jpg')
#     plt.show()


def show_sw():
    """
    draw sliding window result
    :return:
    """
    eval_path_5 = 'history/A0_sw/A0_sw_5/evals.pkl'
    eval_path_10 = 'history/A0_sw/A0_sw_10/evals.pkl'
    eval_path_20 = 'history/A0_sw/A0_sw_20/evals.pkl'
    eval_path_50 = 'history/A0_sw/A0_sw_50/evals.pkl'
    eval_path_infty = './history/A0_sm/evals.pkl'

    eval_dct_5 = load_pickle(eval_path_5)
    eval_dct_10 = load_pickle(eval_path_10)
    eval_dct_20 = load_pickle(eval_path_20)
    eval_dct_50 = load_pickle(eval_path_50)
    eval_dct_infty = load_pickle(eval_path_infty)

    port_ret_ew_5 = eval_dct_5['port_ret_equal_weight']
    port_ret_mc_5 = eval_dct_5['port_ret_market_cap']
    port_ret_ew_10 = eval_dct_10['port_ret_equal_weight']
    port_ret_mc_10 = eval_dct_10['port_ret_market_cap']
    port_ret_ew_20 = eval_dct_20['port_ret_equal_weight']
    port_ret_mc_20 = eval_dct_20['port_ret_market_cap']
    port_ret_ew_50 = eval_dct_50['port_ret_equal_weight']
    port_ret_mc_50 = eval_dct_50['port_ret_market_cap']
    port_ret_ew_infty = eval_dct_infty['port_ret_equal_weight']
    port_ret_mc_infty = eval_dct_infty['port_ret_market_cap']

    #
    accu_port_ret_ew_5 = np.zeros(port_ret_ew_5.shape[0] + 1)
    accu_port_ret_mc_5 = np.zeros_like(accu_port_ret_ew_5)
    accu_port_ret_ew_10 = np.zeros_like(accu_port_ret_ew_5)
    accu_port_ret_mc_10 = np.zeros_like(accu_port_ret_ew_5)
    accu_port_ret_ew_20 = np.zeros_like(accu_port_ret_ew_5)
    accu_port_ret_mc_20 = np.zeros_like(accu_port_ret_ew_5)
    accu_port_ret_ew_50 = np.zeros_like(accu_port_ret_ew_5)
    accu_port_ret_mc_50 = np.zeros_like(accu_port_ret_ew_5)
    accu_port_ret_ew_infty = np.zeros_like(accu_port_ret_ew_5)
    accu_port_ret_mc_infty = np.zeros_like(accu_port_ret_ew_5)

    accu_port_ret_ew_5[0], accu_port_ret_mc_5[0] = 1, 1
    accu_port_ret_ew_10[0], accu_port_ret_mc_10[0] = 1, 1
    accu_port_ret_ew_20[0], accu_port_ret_mc_20[0] = 1, 1
    accu_port_ret_ew_50[0], accu_port_ret_mc_50[0] = 1, 1
    accu_port_ret_ew_infty[0], accu_port_ret_mc_infty[0] = 1, 1

    for i in range(1, accu_port_ret_ew_5.shape[0]):
        accu_port_ret_ew_5[i] = accu_port_ret_ew_5[i-1] + port_ret_ew_5[i-1]
        accu_port_ret_mc_5[i] = accu_port_ret_mc_5[i-1] + port_ret_mc_5[i-1]
        accu_port_ret_ew_10[i] = accu_port_ret_ew_10[i-1] + port_ret_ew_10[i-1]
        accu_port_ret_mc_10[i] = accu_port_ret_mc_10[i-1] + port_ret_mc_10[i-1]
        accu_port_ret_ew_20[i] = accu_port_ret_ew_20[i-1] + port_ret_ew_20[i-1]
        accu_port_ret_mc_20[i] = accu_port_ret_mc_20[i-1] + port_ret_mc_20[i-1]
        accu_port_ret_ew_50[i] = accu_port_ret_ew_50[i-1] + port_ret_ew_50[i-1]
        accu_port_ret_mc_50[i] = accu_port_ret_mc_50[i-1] + port_ret_mc_50[i-1]
        accu_port_ret_ew_infty[i] = accu_port_ret_ew_infty[i-1] + port_ret_ew_infty[i-1]
        accu_port_ret_mc_infty[i] = accu_port_ret_mc_infty[i-1] + port_ret_mc_infty[i-1]

    # figure 1
    plt.figure()
    plt.plot(accu_port_ret_ew_5, label="sw size=5")
    plt.plot(accu_port_ret_ew_10, label="sw size=10")
    plt.plot(accu_port_ret_ew_20, label="sw size=20")
    plt.plot(accu_port_ret_ew_50, label="sw size=50")
    plt.plot(accu_port_ret_ew_infty, label="sw size=infty")
    plt.title("accumulative return from 2005 July to 2020 December(ew)")  # 图形标题
    plt.xlabel("period")  # x轴名称
    plt.ylabel("accumulative return")  # y 轴名称
    plt.legend()
    plt.savefig('./figure1.jpg')
    plt.show()

    # figure 2
    plt.figure()
    plt.plot(accu_port_ret_mc_5, label="sw size=5")
    plt.plot(accu_port_ret_mc_10, label="sw size=10")
    plt.plot(accu_port_ret_mc_20, label="sw size=20")
    plt.plot(accu_port_ret_mc_50, label="sw size=50")
    plt.plot(accu_port_ret_mc_infty, label="sw size=infty")
    plt.title("accumulative return from 2005 July to 2020 December(mc)")  # 图形标题
    plt.xlabel("period")  # x轴名称
    plt.ylabel("accumulative return")  # y 轴名称
    plt.legend()
    plt.savefig('./figure2.jpg')
    plt.show()



def show_m():
    """
    draw sliding window result
    :return:
    """
    eval_path_10 = 'history/A1_m/A1_m_10/evals.pkl'
    eval_path_15 = 'history/A1_m/A1_m_15/evals.pkl'
    eval_path_20 = 'history/A0_sm/evals.pkl'
    eval_path_25 = 'history/A1_m/A1_m_25/evals.pkl'

    eval_dct_10 = load_pickle(eval_path_10)
    eval_dct_15 = load_pickle(eval_path_15)
    eval_dct_20 = load_pickle(eval_path_20)
    print(eval_dct_20)
    exit(0)
    eval_dct_25 = load_pickle(eval_path_25)


    port_ret_ew_10 = eval_dct_10['port_ret_equal_weight']
    port_ret_mc_10 = eval_dct_10['port_ret_market_cap']
    port_ret_ew_15 = eval_dct_15['port_ret_equal_weight']
    port_ret_mc_15 = eval_dct_15['port_ret_market_cap']
    port_ret_ew_20 = eval_dct_20['port_ret_equal_weight']
    port_ret_mc_20 = eval_dct_20['port_ret_market_cap']
    port_ret_ew_25 = eval_dct_25['port_ret_equal_weight']
    port_ret_mc_25 = eval_dct_25['port_ret_market_cap']

    #
    accu_port_ret_ew_10 = np.zeros(port_ret_ew_10.shape[0] + 1)
    accu_port_ret_mc_10 = np.zeros_like(accu_port_ret_ew_10)
    accu_port_ret_ew_15 = np.zeros_like(accu_port_ret_ew_10)
    accu_port_ret_mc_15 = np.zeros_like(accu_port_ret_ew_10)
    accu_port_ret_ew_20 = np.zeros_like(accu_port_ret_ew_10)
    accu_port_ret_mc_20 = np.zeros_like(accu_port_ret_ew_10)
    accu_port_ret_ew_25 = np.zeros_like(accu_port_ret_ew_10)
    accu_port_ret_mc_25 = np.zeros_like(accu_port_ret_ew_10)


    accu_port_ret_ew_10[0], accu_port_ret_mc_10[0] = 1, 1
    accu_port_ret_ew_15[0], accu_port_ret_mc_15[0] = 1, 1
    accu_port_ret_ew_20[0], accu_port_ret_mc_20[0] = 1, 1
    accu_port_ret_ew_25[0], accu_port_ret_mc_25[0] = 1, 1

    for i in range(1, accu_port_ret_ew_10.shape[0]):
        accu_port_ret_ew_10[i] = accu_port_ret_ew_10[i-1] + port_ret_ew_10[i-1]
        accu_port_ret_mc_10[i] = accu_port_ret_mc_10[i-1] + port_ret_mc_10[i-1]
        accu_port_ret_ew_15[i] = accu_port_ret_ew_15[i-1] + port_ret_ew_15[i-1]
        accu_port_ret_mc_15[i] = accu_port_ret_mc_15[i-1] + port_ret_mc_15[i-1]
        accu_port_ret_ew_20[i] = accu_port_ret_ew_20[i-1] + port_ret_ew_20[i-1]
        accu_port_ret_mc_20[i] = accu_port_ret_mc_20[i-1] + port_ret_mc_20[i-1]
        accu_port_ret_ew_25[i] = accu_port_ret_ew_25[i-1] + port_ret_ew_25[i-1]
        accu_port_ret_mc_25[i] = accu_port_ret_mc_25[i-1] + port_ret_mc_25[i-1]


    # figure 1
    plt.figure()
    plt.plot(accu_port_ret_ew_10, label="m=0.1")
    plt.plot(accu_port_ret_ew_15, label="m=0.15")
    plt.plot(accu_port_ret_ew_20, label="m=0.2")
    plt.plot(accu_port_ret_ew_25, label="m=0.25")
    plt.title("accumulative return from 2005 July to 2020 December(ew)")  # 图形标题
    plt.xlabel("period")  # x轴名称
    plt.ylabel("accumulative return")  # y 轴名称
    plt.legend()
    plt.savefig('./figure1.jpg')
    plt.show()

    # figure 2
    plt.figure()
    plt.plot(accu_port_ret_mc_10, label="m=0.1")
    plt.plot(accu_port_ret_mc_15, label="m=0.15")
    plt.plot(accu_port_ret_mc_20, label="m=0.2")
    plt.plot(accu_port_ret_mc_25, label="m=0.25")
    plt.title("accumulative return from 2005 July to 2020 December(ew)")  # 图形标题
    plt.xlabel("period")  # x轴名称
    plt.ylabel("accumulative return")  # y 轴名称
    plt.legend()
    plt.savefig('./figure2.jpg')
    plt.show()


def show_A2():
    """
    draw a2 result
    :return:
    """
    eval_path_return = 'history/A2/A2_f_return/evals.pkl'
    eval_path_sharpe = 'history/A2/A2_f_sharpe/evals.pkl'
    eval_path_IC = 'history/A2/A2_f_IC/evals.pkl'
    eval_path_sharpe_500 = 'history/A2/A2_f_sharpe_500/evals.pkl'

    eval_dct_return = load_pickle(eval_path_return)
    eval_dct_sharpe = load_pickle(eval_path_sharpe)
    eval_dct_IC = load_pickle(eval_path_IC)
    eval_dct_sharpe_500 = load_pickle(eval_path_sharpe_500)


    port_ret_ew_return = eval_dct_return['port_ret_equal_weight']
    port_ret_mc_return = eval_dct_return['port_ret_market_cap']
    port_ret_ew_sharpe = eval_dct_sharpe['port_ret_equal_weight']
    port_ret_mc_sharpe = eval_dct_sharpe['port_ret_market_cap']
    port_ret_ew_IC = eval_dct_IC['port_ret_equal_weight']
    port_ret_mc_IC = eval_dct_IC['port_ret_market_cap']
    port_ret_ew_sharpe_500 = eval_dct_sharpe_500['port_ret_equal_weight']
    port_ret_mc_sharpe_500 = eval_dct_sharpe_500['port_ret_market_cap']

    #
    accu_port_ret_ew_return = np.zeros(port_ret_ew_return.shape[0] + 1)
    accu_port_ret_mc_return = np.zeros_like(accu_port_ret_ew_return)
    accu_port_ret_ew_sharpe = np.zeros_like(accu_port_ret_ew_return)
    accu_port_ret_mc_sharpe = np.zeros_like(accu_port_ret_ew_return)
    accu_port_ret_ew_IC = np.zeros_like(accu_port_ret_ew_return)
    accu_port_ret_mc_IC = np.zeros_like(accu_port_ret_ew_return)
    accu_port_ret_ew_sharpe_500 = np.zeros_like(accu_port_ret_ew_return)
    accu_port_ret_mc_sharpe_500 = np.zeros_like(accu_port_ret_ew_return)


    accu_port_ret_ew_return[0], accu_port_ret_mc_return[0] = 1, 1
    accu_port_ret_ew_sharpe[0], accu_port_ret_mc_sharpe[0] = 1, 1
    accu_port_ret_ew_IC[0], accu_port_ret_mc_IC[0] = 1, 1
    accu_port_ret_ew_sharpe_500[0], accu_port_ret_mc_sharpe_500[0] = 1, 1

    for i in range(1, accu_port_ret_ew_return.shape[0]):
        accu_port_ret_ew_return[i] = accu_port_ret_ew_return[i-1] + port_ret_ew_return[i-1]
        accu_port_ret_mc_return[i] = accu_port_ret_mc_return[i-1] + port_ret_mc_return[i-1]
        accu_port_ret_ew_sharpe[i] = accu_port_ret_ew_sharpe[i-1] + port_ret_ew_sharpe[i-1]
        accu_port_ret_mc_sharpe[i] = accu_port_ret_mc_sharpe[i-1] + port_ret_mc_sharpe[i-1]
        accu_port_ret_ew_IC[i] = accu_port_ret_ew_IC[i-1] + port_ret_ew_IC[i-1]
        accu_port_ret_mc_IC[i] = accu_port_ret_mc_IC[i-1] + port_ret_mc_IC[i-1]
        accu_port_ret_ew_sharpe_500[i] = accu_port_ret_ew_sharpe_500[i-1] + port_ret_ew_sharpe_500[i-1]
        accu_port_ret_mc_sharpe_500[i] = accu_port_ret_mc_sharpe_500[i-1] + port_ret_mc_sharpe_500[i-1]


    # figure 1
    plt.figure()
    plt.plot(accu_port_ret_ew_return, label="return")
    plt.plot(accu_port_ret_ew_sharpe, label="sharpe")
    plt.plot(accu_port_ret_ew_IC, label="IC")
    plt.plot(accu_port_ret_ew_sharpe_500, label="sharpe(zz500)")
    plt.title("accumulative return from 2005 July to 2020 December(ew)")  # 图形标题
    plt.xlabel("period")  # x轴名称
    plt.ylabel("accumulative return")  # y 轴名称
    plt.legend()
    plt.savefig('./figure1.jpg')
    plt.show()

    # figure 2
    plt.figure()
    plt.plot(accu_port_ret_mc_return, label="return")
    plt.plot(accu_port_ret_mc_sharpe, label="sharpe")
    plt.plot(accu_port_ret_mc_IC, label="IC")
    plt.plot(accu_port_ret_mc_sharpe_500, label="sharpe(zz500)")
    plt.title("accumulative return from 2005 July to 2020 December(ew)")  # 图形标题
    plt.xlabel("period")  # x轴名称
    plt.ylabel("accumulative return")  # y 轴名称
    plt.legend()
    plt.savefig('./figure2.jpg')
    plt.show()


if __name__ == '__main__':
    # eval_zz500_hs300()
    # show_m()
    show_A2()
