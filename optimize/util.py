# -*- coding: utf-8 -*-
import copy

import pandas as pd

from Function import read_mt4_data
from TradeTest import Symbol
from optimization import config


def data_load(symbols, start_date, end_date):
    """
    回测数据加载
    :param symbols: 回测品种列表
    :param start_date: 回测开始日期
    :param end_date: 回测结束日期
    :return: 回测数据
    """
    file_dir = "Data//TestData//"
    file_paths = [file_dir+symbol+'1.csv' for symbol in symbols]
    data = read_mt4_data(file_paths, start=start_date, end=end_date)
    return data


def parameter_load(path="Data//Backresult//parameters.csv"):
    """
    历史最优参数加载
    :param path: 参数路径
    :return: 参数，类型为dataframe
    """
    head = ['Symbol', 'TestPeriod', 'Months', 'BiasPeriod', 'Buy', 'Sell', 'Short', 'Cover', 'StopProfitTime']
    try:
        parameters = pd.read_csv(path)
    except IOError:
        parameters = None
        print("Parameters file doesn't exit, please have a check")

    return parameters


def parameter_match(symbols, avg):
    """
    生成策略参数
    :param symbols: 测试外汇品种，type为list,例如['XAUUSD','USDJPY']
    :param avg:品种对应的参数列表
    :return:TradeTest可以处理的策略参数
    """
    symbols_upper = [symbol.upper() for symbol in symbols]
    parameters = []
    for i, symbol in enumerate(symbols_upper):
        base_num = i+1
        if symbol == 'XAUUSD':
            parameter = {'symbol': Symbol('XAUUSD', config.xauusd_base_info_low[1]),
                         'strategy_id': base_num*50000, 'bias_period': int(avg[0+i*6]),
                         'bias_buy': avg[1+i*6], 'bias_short': avg[2+i*6],
                         'bias_sell': avg[3+i*6], 'bias_cover': avg[4+i*6],  'is_check_outTime': True,
                         'outTime': avg[5+i*6], 'limit_profit': config.stop_profit}
            parameters.append(copy.deepcopy(parameter))
        if symbol == 'GBPUSD':
            parameter = {'symbol': Symbol('GBPUSD', config.gbpusd_base_info_low[1]),
                         'strategy_id': base_num * 50000, 'bias_period': int(avg[0 + i * 6]),
                         'bias_buy': avg[1 + i * 6], 'bias_short': avg[2 + i * 6],
                         'bias_sell': avg[3 + i * 6], 'bias_cover': avg[4 + i * 6], 'is_check_outTime': True,
                         'outTime': avg[5 + i * 6], 'limit_profit': config.stop_profit}
            parameters.append(copy.deepcopy(parameter))
        if symbol == 'OILUSD':
            parameter = {'symbol': Symbol('OILUSD', config.oilusd_base_info_low[1]),
                         'strategy_id': base_num * 50000, 'bias_period': int(avg[0 + i * 6]),
                         'bias_buy': avg[1 + i * 6], 'bias_short': avg[2 + i * 6],
                         'bias_sell': avg[3 + i * 6], 'bias_cover': avg[4 + i * 6], 'is_check_outTime': True,
                         'outTime': avg[5 + i * 6], 'limit_profit': config.stop_profit}
            parameters.append(copy.deepcopy(parameter))
        if symbol == 'USDJPY':
            parameter = {'symbol': Symbol('USDJPY', config.usdjpy_base_info_low[1]),
                         'strategy_id': base_num * 50000, 'bias_period': int(avg[0 + i * 6]),
                         'bias_buy': avg[1 + i * 6], 'bias_short': avg[2 + i * 6],
                         'bias_sell': avg[3 + i * 6], 'bias_cover': avg[4 + i * 6], 'is_check_outTime': True,
                         'outTime': avg[5 + i * 6], 'limit_profit': config.stop_profit}
            parameters.append(copy.deepcopy(parameter))
        if symbol == 'USDCHF':
            parameter = {'symbol': Symbol('USDCHF', config.usdchf_base_info_low[1]),
                         'strategy_id': base_num * 50000, 'bias_period': int(avg[0 + i * 6]),
                         'bias_buy': avg[1 + i * 6], 'bias_short': avg[2 + i * 6],
                         'bias_sell': avg[3 + i * 6], 'bias_cover': avg[4 + i * 6], 'is_check_outTime': True,
                         'outTime': avg[5 + i * 6], 'limit_profit': config.stop_profit}
            parameters.append(copy.deepcopy(parameter))

    return parameters


def strategy_evaluate(order_list):
    """
    策略校验
    :param order_list: 订单列表，类型为list
    :return: 资金净值与盈亏比的乘积
    """
    gross_profit = 0
    gross_loss = 0
    pnlList = [t[1] for t in order_list]
    for pnl in pnlList:
        if pnl >= 0:
            gross_profit += pnl
        else:
            gross_loss += pnl
        if gross_loss != 0:
            profit_factor = abs(gross_profit / gross_loss)
    return sum(pnlList) * profit_factor


if __name__ == "__main__":
    parameter_load()



