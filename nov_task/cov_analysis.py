# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: cov_analysis.py
@time: 2017/11/19 19:03
"""

import numpy as np

import os

import pandas as pd

from pandas import Series,DataFrame
import tushare
import matplotlib.pyplot as plt


all_data={}
for ticker in ['600030', '600837']:
    all_data[ticker]=tushare.get_hist_data(ticker,'2016-09-01','2017-09-01')

# price=DataFrame({tic:data['Adg Close']
#                 for tic,data in all_data.items()})
price=DataFrame({tic:data['close']
                for tic,data in all_data.items()})

# volume.tail()
returns=price.pct_change()
tail = returns.tail()
corr = returns.corr()
print('tail: ', tail)
print('corr: ', corr)