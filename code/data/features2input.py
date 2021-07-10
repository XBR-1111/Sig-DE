import numpy as np
from code.util import dump_pickle
from code.data.get_raw_data import get_month_data_feature45, load_pickle

factor_list = ['ret', 'ret_max', 'std_ret', 'std_vol', 'pe_ratio', 'turnover_ratio',
               'pb_ratio', 'ps_ratio', 'pcf_ratio', 'capitalization', 'market_cap',
               'circulating_cap', 'circulating_market_cap', 'pe_ratio_lyr', 'eps',
               'adjusted_profit', 'operating_profit', 'value_change_profit', 'roe',
               'inc_return', 'roa', 'net_profit_margin', 'gross_profit_margin',
               'expense_to_total_revenue', 'operation_profit_to_total_revenue',
               'net_profit_to_total_revenue', 'operating_expense_to_total_revenue',
               'ga_expense_to_total_revenue', 'financing_expense_to_total_revenue',
               'operating_profit_to_profit', 'invesment_profit_to_profit',
               'adjusted_profit_to_profit', 'goods_sale_and_service_to_revenue',
               'ocf_to_revenue', 'ocf_to_operating_profit',
               'inc_total_revenue_year_on_year', 'inc_total_revenue_annual',
               'inc_revenue_year_on_year', 'inc_revenue_annual',
               'inc_operation_profit_year_on_year', 'inc_operation_profit_annual',
               'inc_net_profit_year_on_year', 'inc_net_profit_annual',
               'inc_net_profit_to_shareholders_year_on_year',
               'inc_net_profit_to_shareholders_annual']


def features2input():
    # get raw data
    feature_data = get_month_data_feature45()

    # result
    returns = []  # a list of np.array (I)
    features = []  # a list of np.array (I, D)
    stock_names = []  # a list of np.array (I)
    market_caps = []  # a list of np.array (I)

    for t in range(len(feature_data) - 1):
        print(t)
        current_feature = feature_data[t]
        next_feature = feature_data[t + 1]
        features_t = []
        returns_t = []
        stock_names_t = []

        for ts_code in current_feature.index.tolist():

            if ts_code.split('.')[1] != 'XSHG':
                continue

            features_t_i = []
            for factor in factor_list:
                if ts_code in current_feature[factor].keys():
                    features_t_i.append(current_feature[factor][ts_code])
                else:
                    break

            if len(features_t_i) == len(factor_list):
                if ts_code in next_feature['ret'].keys():
                    # valid stock at time t
                    features_t.append(features_t_i)
                    returns_t.append(next_feature['ret'][ts_code])
                    stock_names_t.append(ts_code)

        features.append(np.array(features_t))
        returns.append(np.array(returns_t))
        stock_names.append(np.array(stock_names_t))

    # features to market_caps
    for array in features:
        market_caps.append(array[:, 10])

    # path
    features_path = './input_data/monthly/features.pkl'
    returns_path = './input_data/monthly/returns.pkl'
    stock_names_path = './input_data/monthly/stock_names.pkl'
    market_caps_path = './input_data/monthly/market_caps.pkl'

    # dump
    dump_pickle(features_path, features)
    dump_pickle(returns_path, returns)
    dump_pickle(stock_names_path, stock_names)
    dump_pickle(market_caps_path, market_caps)


def input_20():
    """
    remove stocks with 20% smallest market cap
    :return:
    """
    # path
    features_path = './input_data/monthly/features.pkl'
    returns_path = './input_data/monthly/returns.pkl'
    stock_names_path = './input_data/monthly/stock_names.pkl'
    market_caps_path = './input_data/monthly/market_caps.pkl'

    # load
    feature_data = load_pickle(features_path)
    return_data = load_pickle(returns_path)
    stock_name_data = load_pickle(stock_names_path)
    market_cap_data = load_pickle(market_caps_path)

    # ratio
    ratio = 0.2

    # result
    new_feature_data = []
    new_return_data = []
    new_stock_name_data = []
    new_market_cap_data = []

    for i, market_cap in enumerate(market_cap_data):
        m = int(market_cap.shape[0] * (1 - ratio))
        sorted_index = np.argsort(-market_cap)  # desc
        selected_index = sorted_index[:m]
        new_feature_data.append(feature_data[i][selected_index, :])
        new_return_data.append(return_data[i][selected_index])
        new_stock_name_data.append(stock_name_data[i][selected_index])
        new_market_cap_data.append(market_cap_data[i][selected_index])

    new_features_path = './input_data/monthly/features_20.pkl'
    new_returns_path = './input_data/monthly/returns_20.pkl'
    new_stock_names_path = './input_data/monthly/stock_names_20.pkl'
    new_market_caps_path = './input_data/monthly/market_caps_20.pkl'

    dump_pickle(new_features_path, new_feature_data)
    dump_pickle(new_returns_path, new_return_data)
    dump_pickle(new_stock_names_path, new_stock_name_data)
    dump_pickle(new_market_caps_path, new_market_cap_data)


def input_50():
    """
    construct a smaller training/testing dataset
    :return:
    """
    # path
    features_path = './input_data/monthly/features.pkl'
    returns_path = './input_data/monthly/returns.pkl'
    stock_names_path = './input_data/monthly/stock_names.pkl'
    market_caps_path = './input_data/monthly/market_caps.pkl'
    hs300_path = './input_data/monthly/hs300.pkl'
    zz500_path = './input_data/monthly/zz500.pkl'

    # load
    feature_data = load_pickle(features_path)
    return_data = load_pickle(returns_path)
    stock_name_data = load_pickle(stock_names_path)
    market_cap_data = load_pickle(market_caps_path)
    hs300_data = load_pickle(hs300_path)
    zz500_data = load_pickle(zz500_path)

    # T
    T = 50
    length = len(feature_data)
    feature_data = feature_data[length-T:length]
    return_data = return_data[length-T:length]
    stock_name_data = stock_name_data[length-T:length]
    market_cap_data = market_cap_data[length-T:length]
    hs300_data = hs300_data[length-T:length]
    zz500_data = zz500_data[length-T:length]

    # new path
    new_features_path = './input_data/monthly/features_s.pkl'
    new_returns_path = './input_data/monthly/returns_s.pkl'
    new_stock_names_path = './input_data/monthly/stock_names_s.pkl'
    new_market_caps_path = './input_data/monthly/market_caps_s.pkl'
    new_hs300_path = './input_data/monthly/hs300_s.pkl'
    new_zz500_path = './input_data/monthly/zz500_s.pkl'

    # sump
    dump_pickle(new_features_path, feature_data)
    dump_pickle(new_returns_path, return_data)
    dump_pickle(new_stock_names_path, stock_name_data)
    dump_pickle(new_market_caps_path, market_cap_data)
    dump_pickle(new_hs300_path, hs300_data)
    dump_pickle(new_zz500_path, zz500_data)


if __name__ == '__main__':
    # features2input()

    # names = load_pickle('./input_data/monthly/stock_names.pkl')
    # print([_.shape[0] for _ in names])

    input_50()
