# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: bias.py
@time: 2017/6/25 14:29
"""
from data_handle.data_read import data_read
import matplotlib.pyplot as plt

rb1601 = data_read('RB1601')
ma = rb1601['close'].rolling(73).mean()
rb1601['ma'] = ma
rb1601['bias'] = (rb1601['ma'] - rb1601['close'])/rb1601['ma']*100
plt.plot(rb1601['bias'])
# plt.plot(rb1601['close'])
plt.show()
print(rb1601)
