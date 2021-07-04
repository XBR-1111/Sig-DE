import numpy as np
import matplotlib.pyplot as plt
from code.util import load_pickle


def show_result():
    R = load_pickle('portfolio_returns.pkl')
    R1 = load_pickle('R1_returns.pkl')
    accu_R = np.zeros(shape=len(R)+1)
    accu_R1 = np.zeros(shape=len(R)+1)
    accu_R[0] = 100
    accu_R1[0] = 100
    for i in range(len(R)):
        accu_R[i+1] = accu_R[i] * (1+R[i])
        accu_R1[i+1] = accu_R1[i] * (1+R1[i])

    plt.figure()
    plt.plot(accu_R, label="proposed model", color="#F08080")
    plt.plot(accu_R1, label="market average return", color="#0B7093")
    plt.title("accumulative return from 2007 March to 2020 December")  # 图形标题
    plt.xlabel("period")  # x轴名称
    plt.ylabel("accumulative return(%)")  # y 轴名称
    plt.legend()
    plt.savefig('./test3.jpg')
    plt.show()


def show_difference():
    R = load_pickle('portfolio_returns.pkl')
    R1 = load_pickle('R1_returns.pkl')
    a = R-R1
    b = np.zeros_like(a)
    plt.figure()
    plt.plot(a, label="proposed model", color="#F08080")
    plt.plot(b, color="#0B7093")
    plt.title("accumulative return differences")  # 图形标题
    plt.xlabel("period")  # x轴名称
    plt.ylabel("R-R1")  # y 轴名称
    # plt.legend()
    plt.savefig('./test4.jpg')
    plt.show()

if __name__ == '__main__':
    show_difference()
