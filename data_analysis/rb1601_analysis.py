# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: rb1601_analysis.py
@time: 2017/6/25 14:05
"""
import pandas as pd
import matplotlib.pyplot as plt

from data_handle.mongo import mongo_connect
from data_handle.data_insert import BarData


client = mongo_connect()
collection = client.future_1min.RB1601
dbCursor = collection.find()
data = BarData()
init_data = []
for d in dbCursor:
    init_data.append(d)
# print(init_data)
data_pd = pd.DataFrame(init_data)
print('origin_length: ', len(data_pd))
print("length: ", len(data_pd[data_pd.volume==0]))
print("theory_length:", 345*252)
plt.plot(data_pd['volume'])
plt.show()
