# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: system_research.py
@time: 2017/8/18 16:19
"""
import matplotlib.pyplot as plt

import talib
import tushare


def indicator_plot():
    stock = '002230'
    data = tushare.get_hist_data(stock)
    data.sort_index(inplace=True)
    close = data['close'].values
    long_ma_period = 26
    short_ma_period = 13
    long_ma = talib.MA(close, long_ma_period)
    short_ma = talib.MA(close, short_ma_period)
    long_short_diff = long_ma-short_ma
    # print('long_short_diff: {}'.format(long_short_diff))
    data['close'].plot(label='close')
    plt.plot(long_ma, label='long_ma_{}'.format(long_ma_period))
    plt.plot(short_ma, label='short_ma_{}'.format(short_ma_period))
    plt.plot(long_short_diff, label='long_short_diff')
    plt.title('stock_{}'.format(stock))
    plt.ylabel('price')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    indicator_plot()
