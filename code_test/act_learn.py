# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: act_learn.py
@time: 2017/10/23 17:30
"""
import os
import sys

import talib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from research.rnn.rnn import fix_data, data_frequency_transfer, get_factors, dense_to_one_hot

local_path = sys.path[1]
file_path = os.path.join(local_path, 'research/rnn', '白银88.csv')
tmp = fix_data(file_path)

# targets 1d 数据合成
tmp_1d = data_frequency_transfer(tmp, '1d')
rolling = 88
targets = tmp_1d
targets['returns'] = targets['close'].shift(-2) / targets['close'] - 1.0
targets['upper_boundary'] = targets.returns.rolling(rolling).mean() + 0.5 * targets.returns.rolling(rolling).std()
targets['lower_boundary'] = targets.returns.rolling(rolling).mean() - 0.5 * targets.returns.rolling(rolling).std()
targets.dropna(inplace=True)
targets['labels'] = 1
targets.loc[targets['returns'] >= targets['upper_boundary'], 'labels'] = 2
targets.loc[targets['returns'] <= targets['lower_boundary'], 'labels'] = 0

# factors 1d 数据合成
Index = tmp_1d.index
High = tmp_1d.high.values
Low = tmp_1d.low.values
Close = tmp_1d.close.values
Open = tmp_1d.open.values
Volume = tmp_1d.volume.values
factors = get_factors(Index, Open, Close, High, Low, Volume, rolling=26, drop=True)
factors = factors.loc[:targets.index[-1]]
tmp_factors_1 = factors.iloc[:12]
targets = targets.loc[tmp_factors_1.index[-1]:]
gather_list = np.arange(factors.shape[0])[11:]
inputs = np.array(factors).reshape(-1, 1, factors.shape[1])
targets = dense_to_one_hot(targets['labels'])
targets = np.expand_dims(targets, axis=1)
print(targets.shape)
