# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: pandas_learn.py
@time: 2017/11/6 19:11
"""
import pandas as pd
pd = pd.DataFrame()
pd['a'] = list(range(100001))
pd['b'] = list(range(100001))
df2 = pd
df2['c'] = df2['a'].shift(-1) / df2['a']-1.0
print(df2)
