# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: data_read.py
@time: 2017/6/25 15:16
"""
import pandas as pd
import matplotlib.pyplot as plt

from data_handle.mongo import mongo_connect
from data_handle.data_insert import BarData


def data_read(symbol):
    client = mongo_connect()
    collection = eval('client.future_1min.{}'.format(symbol))
    dbCursor = collection.find()
    init_data = []
    for d in dbCursor:
        init_data.append(d)
    # print(init_data)
    data_pd = pd.DataFrame(init_data)
    return data_pd