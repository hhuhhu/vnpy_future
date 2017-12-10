# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: atr_analysis.py
@time: 2017/8/29 17:12
"""
import pandas as pd
import matplotlib.pyplot as plt
import talib

from data_handle.data_load import data_load
from core.ctaBase import *


def atr_analysis():
    start_date = '20160701'
    end_date = '20180823'
    db_name = FUTURE_1MIN
    symbol = 'RB1801'
    data = data_load(db_name=db_name, symbol=symbol, start_date=start_date, end_date=end_date)
    print('data_length: ', len(data))
    high = data['high']
    low = data['low']
    close = data['close']
    volume = data['volume']
    # volume.plot()
    # close.plot(label='close')
    # low.plot(label='low')
    # high.plot(label='high')
    # plt.bar(data=volume, height=2000000, left=10, label='volume')
    # plt.plot(volume, label='volume')
    #
    # plt.legend()
    # plt.show()
    atr_length = 10
    ma = talib.MA(close.values)
    atr = talib.ATR(high.values,low.values,close.values,20)
    # plt.bar(left=0.01, height=2,data=atr, width=0.01)
    # plt.scatter(20,20,data=atr)
    # plt.hist(atr)
    plt.plot(atr)

    print(atr)
    plt.show()

if __name__ == '__main__':
    atr_analysis()
