# coding=utf-8
"""读取外汇数据并分析，计算技术指标"""
from __future__ import division
from pandas.core.series import Series
from pandas.core.api import DataFrame
import numpy as np


class Indicator(object):
    def __init__(self, data_set=None, applied_price=None):
        self.data_set = data_set
        self.applied_price = 'close' if applied_price is None else applied_price

    @classmethod
    def i_rsi(cls, data_set, period, applied_price):
        """
        计算rsi指标
        Args:
        data_set: [dict(symbol=DataFrame) or DataFrame], 待分析的数据集是一个以品种名为key, value是DataFrame或者是一个DataFrame
        period: [int],指标的计算周期
        applied_price:[str, default 'close', other 'open','high','low','Ask','Bid'] 使用的价格

        Returns:
            Series, 指标序列
        """
        def rsi(x):
            # x<0的值为负数，取绝对值相加
            sum_p = np.sum(x[x > 0])
            sum_n = np.sum(x[x < 0])
            if sum_n != 0.0:
                return 100 - 100 / (1 + sum_p / np.abs(sum_n))
            else:
                if sum_p != 0.0:
                    return 100.0
                else:
                    return 50.0

        return cls.cal_indicator("RSI", data_set, rsi, period, applied_price)

    @staticmethod
    def cal_indicator(i_name, data_set, calculator, period, applied_price, is_rolling=True):
        """
        计算指标函数
        Args:
            i_name: 指标名称，用于分类识别
            data_set: dict(symbol=DataFrame)或DataFrame, 待分析的数据集是一个以品种名为key, value是DataFrame或者是一个DataFrame
            calculator: [func],指标计算的函数
            period: [int], 指标周期
            applied_price: [str, default 'close', other 'open','high','low','Ask','Bid'] 使用的价格
            is_rolling:是否采用rolling应用函数
        Returns:
            Series, 指标序列
        """
        ind_dict = {}
        if isinstance(data_set, dict):
            for key in data_set:
                if is_rolling:
                    diff = data_set[key][applied_price] - data_set[key][applied_price].shift(1)
                    ind = diff.rolling(window=period - 1).apply(calculator)
                    ind.name = i_name
                else:
                    ind = calculator(i_name, data_set[key], period, applied_price)
                ind_dict[key] = ind
            return ind_dict
        elif isinstance(data_set, DataFrame):
            if is_rolling:
                diff = data_set[applied_price] - data_set[applied_price].shift(1)
                ind = diff.rolling(window=period - 1).apply(calculator)
                ind.name = i_name
            else:
                ind = calculator(i_name, data_set, period, applied_price)
            return ind
        else:
            raise ValueError(u"数据集输入类型输入错误")

    @classmethod
    def mt4_rsi(cls, data_set, period, applied_price):
        """
        计算mt4中的rsi指标

        Args:
            data_set: [DataFrame],价格数据
            period: 指标周期
            applied_price: 应用的价格
        Returns:
            Series
        
        """

        def rsi(ind_name, data, i_period, i_applied_price):
            rsi_buffer = []
            pos_buffer = []
            neg_buffer = []
            close = data[i_applied_price]
            c = close.values
            sump = 0.0
            sumn = 0.0

            for i in range(len(close)):
                if i < i_period:
                    rsi_buffer.append(0.0)
                    pos_buffer.append(0.0)
                    neg_buffer.append(0.0)
                    if i > 0:
                        diff = c[i] - c[i - 1]
                        if diff > 0:
                            sump += diff
                        else:
                            sumn -= diff

                elif i == i_period:
                    pos_buffer.append(sump / i_period)
                    neg_buffer.append(sumn / i_period)
                    if neg_buffer[i] != 0.0:
                        rsi_buffer.append(100.0 - 100.0 / (1.0 + pos_buffer[i] / neg_buffer[i]))
                    else:
                        if pos_buffer[i] != 0.0:
                            rsi_buffer.append(100.0)
                        else:
                            rsi_buffer.append(50.0)

                else:
                    diff = c[i] - c[i - 1]
                    pos_buffer.append((pos_buffer[i - 1] * (i_period - 1) + (diff if diff > 0.0 else 0.0)) / i_period)
                    neg_buffer.append(
                        (neg_buffer[i - 1] * (i_period - 1) + (-1 * diff if diff < 0.0 else 0.0)) / i_period)
                    if neg_buffer[i] != 0.0:
                        rsi_buffer.append(100.0 - 100.0 / (1.0 + pos_buffer[i] / neg_buffer[i]))
                    else:
                        if pos_buffer[i] != 0.0:
                            rsi_buffer.append(100.0)
                        else:
                            rsi_buffer.append(50.0)
            return Series(data=rsi_buffer, index=data.index, name=ind_name)

        return cls.cal_indicator("RSI", data_set, rsi, period, applied_price, is_rolling=False)

    def i_bias(self, period):
        def _bias(i_name, data_set, period_, applied_price):
            ma = data_set[applied_price].rolling(window=period_).mean()
            bias = (data_set[applied_price] - ma) * 100 / ma
            bias.name = i_name
            return bias
        return self.cal_indicator("Bias", self.data_set, _bias, period, self.applied_price, is_rolling=False)




