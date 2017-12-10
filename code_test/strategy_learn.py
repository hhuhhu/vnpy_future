# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: strategy_learn.py
@time: 2017/10/20 11:58
"""
import pandas as pd
a = pd.DataFrame([1,2,3,4,5])
# print(a.values)
m = a.shift(-2)-a
print(m)