# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: band.py
@time: 2017/7/27 16:05
"""
import talib
import tushare
import numpy as np
import pandas as pd
data = tushare.get_hist_data('000001')
# print('data: ', data)
close = data['close']
# print('average: ', pd.rolling_mean(close, 10))
print('close_length: ', len(close))
# print('close: ', close)
close = np.array(close)
band = talib.BBANDS(close, 10)
print('band_length: ', len(band))
print('upper: ', band[0])
# print('middle: ', band[1])
