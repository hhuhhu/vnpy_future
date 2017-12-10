# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: xauusd_analysis.py
@time: 2017/11/3 16:16
"""
import os
import sys

import pandas as pd

def data_get():
    dir_path = sys.path[0]
    file_path = os.path.join(dir_path, 'XAUUSD60.csv')
    # print(file_path)
    header = ['date','time','open','high','low','close', 'volume']
    data = pd.read_csv(file_path,names=header)
    start = '2004.07.01'
    end = '2016.07.01'
    data = data[data.date>=start]
    data = data[data.date < end]
    data['datetime'] = data['date'] +' ' + data['time']
    del data['date']
    del data['time']
    data.set_index(data.datetime, inplace=True)
    print("data_length: {}".format(len(data)))

    return data
