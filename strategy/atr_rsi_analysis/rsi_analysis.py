# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: rsi_analysis.py
@time: 2017/9/11 8:28
"""
import pandas as pd
import matplotlib.pyplot as plt
import talib

from data_handle.data_load import data_load
from core.ctaBase import *


def rsi_analysis():
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
    rsi = talib.RSI(close.values,14)
    # plt.bar(left=0.01, height=2,data=atr, width=0.01)
    # plt.scatter(20,20,data=atr)
    # plt.hist(atr)
    plt.plot(rsi)

    print(rsi)
    plt.show()

if __name__ == '__main__':
    rsi_analysis()
