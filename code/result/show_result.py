import numpy as np
import matplotlib.pyplot as plt
from code.util import load_pickle
from code.evaluation.evaluation import eval_6

def show_result(num):
    """

    :param num: num of cases
    :return:
    """

    # get_data
    eval_dct = load_pickle('./result/evals.pkl')
    print(eval_dct)
    port_ret_equal_weight = eval_dct['port_ret_equal_weight']
    port_ret_market_cap = eval_dct['port_ret_market_cap']
    zz500 = load_pickle('./data/input_data/monthly/zz500.pkl')
    hs300 = load_pickle('./data/input_data/monthly/hs300.pkl')
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
    plt.plot(accu_mc, label="proposed method(mc)")
    plt.plot(accu_zz500, label="zz500")
    plt.plot(accu_hs300, label="hs300")
    plt.title("accumulative return from 2005 July to 2020 December")  # 图形标题
    plt.xlabel("period")  # x轴名称
    plt.ylabel("accumulative return")  # y 轴名称
    plt.legend()
    plt.savefig('./figure1.jpg')
    plt.show()

    # figure 2
    plt.figure()
    plt.plot(accu_ew-accu_zz500, label="hedge_ret_500(ew)")
    plt.plot(accu_mc-accu_zz500, label="hedge_ret_500(mc)")
    plt.plot(accu_ew-accu_hs300, label="hedge_ret_300(ew)")
    plt.plot(accu_mc-accu_hs300, label="hedge_ret_300(mc)")
    plt.title("hedge_ret from 2005 July to 2020 December")  # 图形标题
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


if __name__ == '__main__':
    # eval_zz500_hs300()
    R = load_pickle('./history/A0/evals.pkl')
    print(R)