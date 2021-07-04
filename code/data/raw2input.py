from code.data.get_raw_data import get_price_raw, get_month_data_feature45
from code.util import load_pickle, dump_pickle
import numpy as np


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

indicator_list = [1 for i in range(len(factor_list))]

def get_all_ts_code(price_df):
    """
    get all ts_code from the price dataframe
    :param price_df: the price dataframe
    :return: a list of all ts_code
    """
    return price_df['ts_code'].drop_duplicates(inplace=False).values.tolist()


def select_ts_code(ts_list, extra_info=None):
    """
    first,select Shanghai A share stokes, then exclude the stocks in financial industry
    and those ever labeled as Special Treatment (ST)
    :param ts_list: a list of all ts_code
    :param extra_info: ST info + financial industry info
    :return: a new list of selected ts_code
    """

    ts_sh_a = []
    # The Shanghai A share market ts_code start with 600/601/603
    for ts in ts_list:
        num, label = ts.split('.')
        if label == 'SH' and str(num).startswith(('600', '601', '603')):
            ts_sh_a.append(ts)

    # TODO the second step:  exclude ST and the stocks in financial industry
    return ts_sh_a


def get_month_range():
    # 确定月范围， 考虑price和feature的范围
    start_ym = (2007, 3)
    end_ym = (2020, 12)
    return start_ym, end_ym


def get_index_range():
    # 0 ->2005 5
    # 24->2007 5
    # 22->2007 3
    # 187->2020 12
    # 190->2021 3
    begin_index = 22
    end_index = 187
    return begin_index, end_index


def get_stocks():
    # 确定股票范围,price和feature重叠

    # price 部分
    price_df = get_price_raw()
    price_df = price_df[['trade_date', 'ts_code', 'close']]
    all_ts_code = get_all_ts_code(price_df)
    ts_codes_a = select_ts_code(all_ts_code)

    first_monthly_day = 20070228

    price_df = price_df[
        price_df['ts_code'].isin(ts_codes_a) & (price_df['trade_date'] == first_monthly_day)]

    ts_codes_price = []

    for ts in ts_codes_a:
        r = price_df[price_df['ts_code'] == ts].close.values
        if len(r) != 0:
            ts_codes_price.append(ts)
    print(len(ts_codes_price))
    print(ts_codes_price)


    # feature 部分
    ts_codes_feat = []
    for ts in ts_codes_price:
        ts_codes_feat.append(ts.split('.')[0] + '.XSHG')
    print(ts_codes_feat)
    begin_index = 24

    feature_data = get_month_data_feature45()
    ts_code_final = []

    t_df = feature_data[begin_index]
    j = 0
    factor = factor_list[j]
    t_f_df = t_df[factor]
    for i in range(len(ts_codes_feat)):
        ts = ts_codes_feat[i]
        if ts in t_f_df.index.tolist():
            ts_code_final.append(ts)

    print(len(ts_code_final))
    return ts_code_final


def select_trading_day(price_df):
    """
    select trading_day from price_df
    :param price_df: price dataframe
    :return: a list of trading_day with format yyyymmdd
    """
    return price_df['trade_date'][0].values


def select_month_days(trading_days, start_ym, end_ym):
    """
    select last day of each month
    :param trading_days: a list of all trading days
    :return: day_indexes: a list of integers with indexes of trading_days
            intervals: a list of (year, month) with length t
    """
    # start at the earliest point of time
    current_month = (trading_days[0] % 10000) // 100

    # select t+1 days and t intervals
    intervals = []
    day_indexes = []

    end_ym = (end_ym[0] if end_ym[1] != 12 else end_ym[0] + 1, end_ym[1]+1 if end_ym[1] != 12 else 1)
    for i in range(len(trading_days)):
        num = trading_days[i]
        y = num // 10000
        m = (num % 10000) // 100
        if m != current_month and start_ym[0] * 100 + start_ym[1] <= y * 100 + m <= end_ym[0] * 100 + end_ym[1] :
            day_indexes.append(i-1)
            intervals.append((y, m))
            current_month = m

    return np.array(day_indexes), np.array(intervals[:-1])


def price2input(start_ym, end_ym, stocks):

    #
    price_df = get_price_raw()
    price_df = price_df[['trade_date', 'ts_code', 'close']]

    # get all trading day
    trading_day = select_trading_day(price_df)
    print("trading days:", trading_day)

    # select a day for a month
    month_days_index, months = select_month_days(trading_day, start_ym, end_ym)
    print(month_days_index)
    print(months)
    print(len(month_days_index))
    print(len(months))
    exit(0)
    # stocks conversion
    stocks_price = []
    for stock in stocks:
        stocks_price.append(stock.split('.')[0] + '.SH')
    stocks_price = np.array(stocks_price)

    # select by ts_code candidates and trade_date
    price_df = price_df[
        price_df['ts_code'].isin(stocks_price) & price_df['trade_date'].isin(trading_day[month_days_index])
        ]

    print('begin to select data')
    price = []  # list of list with shape (num_ts, num_quarters)
    final_ts_codes = []
    count = 0
    for i in range(len(stocks_price)):
        count += 1
        if count % 50 == 0:
            print("finish %d stocks" % count)
        ts = stocks_price[i]

        day_list = []  # for a ts_code
        for j in range(len(month_days_index)):
            day_index = month_days_index[j]
            day = trading_day[day_index]
            list_of_r = price_df[(price_df['trade_date'] == day) & (price_df['ts_code'] == ts)][
                'close'].values
            if len(list_of_r) == 1:
                day_list.append(list_of_r[0])
            else:
                if len(day_list) > 0:
                    day_list.append(day_list[-1])
                else:
                    day_list.append(0)
        assert len(day_list) == len(month_days_index)
        # print(day_list)
        price.append(day_list)
        final_ts_codes.append(ts)

    print('%d final stokes are selected due to data integrality' % (len(final_ts_codes)))
    return np.array(price), np.array(final_ts_codes), np.array(months), np.array(trading_day), np.array(month_days_index)


def features2input(stocks, trading_day, month_days_index):
    # get factor raw data
    factor_raw = get_month_data_feature45()
    begin_index, end_index = get_index_range()
    factor_array = np.zeros(shape=[len(stocks), len(factor_list), end_index-begin_index+1])
    # for i in range(len(stocks):

    for k in range(end_index-begin_index+1):
        t_df = factor_raw[k+begin_index]
        for j in range(len(factor_list)):
            factor = factor_list[j]
            t_f_df = t_df[factor]
            for i in range(len(stocks)):
                ts = stocks[i]
                if ts in t_f_df.index.tolist():
                    factor_array[i, j, k] = t_f_df[ts]
                else:
                    factor_array[i, j, k] = factor_array[i, j, k-1]
    return factor_array


if __name__ == '__main__':
    # start_ym, end_ym = get_month_range()
    # stocks = get_stocks()
    # price, ts_codes, months, trading_day, month_days_index = price2input(start_ym, end_ym, stocks)
    # dump_pickle('./input_data/monthly/months.pkl', months)
    # dump_pickle('./input_data/monthly/price.pkl', price)
    # dump_pickle('./input_data/monthly/month_days_index.pkl', month_days_index)
    # dump_pickle('./input_data/ts_codes.pkl', ts_codes)
    # dump_pickle('./input_data/trading_day.pkl', trading_day)
    #
    # print(price.shape)
    # print(ts_codes.shape)
    # print(months.shape)
    # print(trading_day.shape)
    # print(month_days_index.shape)

    stocks = get_stocks()
    trading_day = load_pickle('./input_data/trading_day.pkl')
    month_days_index = load_pickle('./input_data/monthly/month_days_index.pkl')
    factor_array = features2input(stocks, trading_day, month_days_index)
    dump_pickle('./input_data/monthly/features.pkl', factor_array)

    # dump_pickle('./input_data/indicator.pkl', np.array(indicator_list))
