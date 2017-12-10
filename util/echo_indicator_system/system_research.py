# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: system_research.py
@time: 2017/8/18 16:19
"""
import matplotlib.pyplot as plt
import tushare
data = tushare.get_hist_data('002230')
data.sort_index(inplace=True)
print(data['close'])
data['close'].plot()
plt.show()
