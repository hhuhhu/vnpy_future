# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: indicator_learn.py
@time: 2017/10/23 10:20
"""
import os
import sys

import talib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from research.rnn.rnn import fix_data, data_frequency_transfer


class IndicatorAnalysis:
    """
    指标分析
    """

    def __init__(self):
        self.local_path = sys.path[1]
        self.file_path = os.path.join(self.local_path, 'research/rnn', '白银88.csv')
        self.data = fix_data(self.file_path)
        self.data_1d = data_frequency_transfer(self.data, '1d')
        self.high = self.data_1d['high'].values
        self.low = self.data_1d['low'].values
        self.open = self.data_1d['open'].values
        self.close = self.data_1d['close'].values
        self.figure = plt.figure()
        self.ax1 = self.figure.add_subplot(111)
        self.ax1.plot(self.data_1d['close'], color='b', marker='o', label='close')
        self.ax1.legend(bbox_to_anchor=(0.98, 0.98))
        self.ax2 = self.ax1.twinx()

    def apo_analysis(self):
        figure = plt.figure()
        ax1 = figure.add_subplot(111)
        ax1.plot(self.data_1d['close'], color='b', marker='o', label='close')
        self.data_1d['APO'] = talib.APO(self.close, fastperiod=12, slowperiod=26)
        self.data_1d['MA26'] = self.data_1d['close'].rolling(26).mean()
        self.data_1d['MA12'] = self.data_1d['close'].rolling(12).mean()
        print("data: ", self.data_1d)
        self.data_1d.to_csv('data.csv')
        ax2 = ax1.twinx()
        ax2.plot(self.data_1d['APO'], color='r', marker='*', label='APO')
        ax1.legend(bbox_to_anchor=(0.98, 0.98))
        ax2.legend(bbox_to_anchor=(0.98, 0.96))
        plt.title('AG88_2014-2016_Daily')
        plt.show()

    def aroon_analysis(self):
        self.data_1d['AROONDown'], self.data_1d['AROONUp'] = talib.AROON(self.high, self.low, timeperiod=14)
        self.data_1d['AROONOSC'] = talib.AROONOSC(self.high, self.low, timeperiod=14)
        figure = plt.figure()
        ax1 = figure.add_subplot(111)
        ax1.plot(self.data_1d['close'], color='b', marker='o', label='close')
        ax2 = ax1.twinx()
        ax2.plot(self.data_1d['AROONDown'], color='r', marker='*', label='AROONDown')
        ax2.plot(self.data_1d['AROONUp'], color='y', marker='*', label='AROONUp')
        ax2.plot(self.data_1d['AROONOSC'], color='g', marker='*', label='AROONOSC')
        self.data_1d['AROONOSC'] = talib.AROONOSC(self.high, self.low, timeperiod=14)
        ax1.legend(bbox_to_anchor=(0.98, 0.98))
        ax2.legend(bbox_to_anchor=(0.98, 0.96))
        print(self.data_1d)
        plt.title('AG88_2014-2016_Daily')
        plt.show()

    def atr_analysis(self):
        self.data_1d['ATR14'] = talib.ATR(self.high, self.low, self.close)
        self.ax2.plot(self.data_1d['ATR14'], color='r', marker='*', label='ATR')
        self.ax2.legend(bbox_to_anchor=(0.98, 0.96))
        plt.title('AG88_2014-2016_Daily')
        plt.show()

    def boll_analysis(self):
        self.data_1d['Boll_Up'], self.data_1d['Boll_Mid'], self.data_1d['Boll_Down'] = talib.BBANDS(self.close,
                                                                                                    timeperiod=20,
                                                                                                    nbdevup=2,
                                                                                                    nbdevdn=2,
                                                                                                    matype=0)


    def __call__(self):
        self.data_1d.to_csv('data.csv')


if __name__ == '__main__':
    indicator_analysis = IndicatorAnalysis()
    indicator_analysis.atr_analysis()
