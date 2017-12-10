# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: dataframe_learn.py
@time: 2017/7/24 17:09
"""
import tushare as ts
import pandas as pd

stockdata = ts.get_hist_data('000001')
print(stockdata.close)