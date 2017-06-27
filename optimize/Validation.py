# -*- coding: utf-8 -*-
import copy

import numpy

from TradeTest import TradeTest
from optimization.util import parameter_match, parameter_load


class StrategyValidation(object):
    """
    验证类，主要是验证第一个月优化后的参数
    """
    def __init__(self, symbols):
        """Constructor"""
        self.data = numpy.zeros(4)  # 回测数据
        self.symbols = symbols

    def capital(self, strategy_avg):
        """
        本函数为优化目标函数，返回优化参考值

        :param strategy_avg: 回测参数
        :return: capital，资金净值；profit_factor,盈亏比
        """
        gross_loss = 0.0
        gross_profit = 0.0
        profit_factor = 0.0
        parameter = parameter_match(self.symbols, strategy_avg)
        test = TradeTest(parameter)
        test.init_data = self.data

        pnldict = test.on_test()
        pnlList = [t[1] for t in pnldict]

        for pnl in pnlList:
            if pnl >= 0:
                gross_profit += pnl
            else:
                gross_loss += pnl
            if gross_loss != 0:
                profit_factor = abs(gross_profit / gross_loss)
        return sum(pnlList) * profit_factor

    def parameter_exchange(self):
        """
      本函数为优化目标函数，返回优化参考值

      :param strategy_avg: 回测参数
      :return: capital，资金净值；profit_factor,盈亏比
      """
        val_dict = {}
        parameter = parameter_load()
        parameters = parameter.loc[:, ['BiasPeriod', 'Buy', 'Sell', 'Short', 'Cover', 'StopProfitTime']]
        _parameters = parameters.values
        parameter_num = len(parameters)
        for j in range(parameter_num):
            immediate = self.capital(_parameters[j])
            val_dict[immediate] = copy.deepcopy(_parameters[j])
        immediate_keys = val_dict.keys()
        immediate_keys.sort()
        avg = val_dict[immediate_keys[-1]]
        print ("immediate_keys: ", immediate_keys)

        return avg, immediate_keys[-1]







