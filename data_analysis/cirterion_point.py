# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: cirterion_point.py
@time: 2017/6/29 14:18
"""
import os
import sys

from data_analysis.indicator_analysis.PriceSignal import *
from data_handle.data_load import data_read

data = data_read('RB1601')
data.index = data['datetime']
fig, ax = plt.subplots()
data['open'].plot(ax=ax, legend=False)
# new_series, _ = criterion_method(reader.data['XAUUSD']['open'], 10, 0.10)
new_series, bins = piecewise_linear(data['open'], 15)
# new_series, bins = criterion_method(data['open'], 200, 0.1)
print("new_series: ", new_series)
print("bins: ", bins)
print("criterion_point_num: ", len(bins))
print(new_series["signal"].iat[bins[1]], new_series["signal"].iat[bins[-1]])
new_series["trading_point"].plot(ax=ax, legend=False, linestyle='dashed', marker='o', markerfacecolor='red',
                                 markersize=4)
# new_series["signal"].plot(ax=ax, linestyle='--', secondary_y=True, color='g')
plt.show()