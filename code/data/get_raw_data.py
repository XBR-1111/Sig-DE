from code.util import load_pickle


def get_fundamental_data():
    # a dictionary, with keys ranging from 0 to 187
    # 0 means 2005 May
    # values are <class 'pandas.core.frame.DataFrame'>s

    # Index(['id', 'code', 'pe_ratio', 'turnover_ratio', 'pb_ratio', 'ps_ratio',
    #        'pcf_ratio', 'capitalization', 'market_cap', 'circulating_cap',
    #        'circulating_market_cap', 'day', 'pe_ratio_lyr', 'id.1', 'code.1',
    #        'day.1', 'pubDate', 'statDate', 'eps', 'adjusted_profit',
    #        'operating_profit', 'value_change_profit', 'roe', 'inc_return', 'roa',
    #        'net_profit_margin', 'gross_profit_margin', 'expense_to_total_revenue',
    #        'operation_profit_to_total_revenue', 'net_profit_to_total_revenue',
    #        'operating_expense_to_total_revenue', 'ga_expense_to_total_revenue',
    #        'financing_expense_to_total_revenue', 'operating_profit_to_profit',
    #        'invesment_profit_to_profit', 'adjusted_profit_to_profit',
    #        'goods_sale_and_service_to_revenue', 'ocf_to_revenue',
    #        'ocf_to_operating_profit', 'inc_total_revenue_year_on_year',
    #        'inc_total_revenue_annual', 'inc_revenue_year_on_year',
    #        'inc_revenue_annual', 'inc_operation_profit_year_on_year',
    #        'inc_operation_profit_annual', 'inc_net_profit_year_on_year',
    #        'inc_net_profit_annual', 'inc_net_profit_to_shareholders_year_on_year',
    #        'inc_net_profit_to_shareholders_annual'],
    #       dtype='object')
    fundamental_data_path = 'raw_data/fundamental_data.pkl'
    fundamental_data = load_pickle(fundamental_data_path)
    return fundamental_data


def get_market_data():
    # a dictionary, with keys ranging from 0 to 187
    # 0 means 2005 May
    # values are <class 'pandas.core.frame.DataFrame'>s
    market_data_path = 'raw_data/market_data.pkl'
    market_data = load_pickle(market_data_path)
    # print(type(market_data[0]))
    return market_data


def get_month_data_feature45():
    # a dictionary, with keys ranging from 0 to 187
    # 0 means 2005 May
    # values are <class 'pandas.core.frame.DataFrame'>s

    # Index(['ret', 'ret_max', 'std_ret', 'std_vol', 'pe_ratio', 'turnover_ratio',
    #        'pb_ratio', 'ps_ratio', 'pcf_ratio', 'capitalization', 'market_cap',
    #        'circulating_cap', 'circulating_market_cap', 'pe_ratio_lyr', 'eps',
    #        'adjusted_profit', 'operating_profit', 'value_change_profit', 'roe',
    #        'inc_return', 'roa', 'net_profit_margin', 'gross_profit_margin',
    #        'expense_to_total_revenue', 'operation_profit_to_total_revenue',
    #        'net_profit_to_total_revenue', 'operating_expense_to_total_revenue',
    #        'ga_expense_to_total_revenue', 'financing_expense_to_total_revenue',
    #        'operating_profit_to_profit', 'invesment_profit_to_profit',
    #        'adjusted_profit_to_profit', 'goods_sale_and_service_to_revenue',
    #        'ocf_to_revenue', 'ocf_to_operating_profit',
    #        'inc_total_revenue_year_on_year', 'inc_total_revenue_annual',
    #        'inc_revenue_year_on_year', 'inc_revenue_annual',
    #        'inc_operation_profit_year_on_year', 'inc_operation_profit_annual',
    #        'inc_net_profit_year_on_year', 'inc_net_profit_annual',
    #        'inc_net_profit_to_shareholders_year_on_year',
    #        'inc_net_profit_to_shareholders_annual'],
    #       dtype='object')
    month_data_path = 'raw_data/month_data_feature45_new.pkl'
    month_data = load_pickle(month_data_path)
    # print(month_data[0].columns)
    return month_data


def get_factor_raw():
    factor_data_path = 'raw_data/factor_data.pkl'
    factor_data = load_pickle(factor_data_path)
    #print(factor_data[0].columns)
    return factor_data


def get_price_raw():
    price_data_path = 'raw_data/price.pkl'
    price_data = load_pickle(price_data_path)
    # print(price_data[0].columns)
    return price_data


if __name__ == '__main__':
    get_month_data_feature45()
