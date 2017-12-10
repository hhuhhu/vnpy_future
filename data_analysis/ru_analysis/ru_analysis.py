# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: ru_analysis.py
@time: 2017/9/28 16:46
"""
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('E://ru//ru1711.csv',encoding='gbk')
data['settlement'] = data['turover']/data['volumw']/10  # 结算价
data['high'] = data['high']/10000
data['low'] = data['low']/10000
data['close'] = data['close']/10000
data['open'] = data['open']/10000


def func_data_handle(x):
    """
    处理结算价：如果最后一位小于5则为0否则为5，对应品种橡胶
    :param x:价格数据
    :return:结算价
    """
    x = math.floor(x)
    if int(str(x)[-1]) < 5:
        num = str(x)[:-1] + '0'
    else:
        num = str(x)[:-1] + '5'
    return eval(num)
data['settlement'] = data['settlement'].apply(func_data_handle)
data_length = len(data)
settlement_value = data['settlement'].values[:-1]
temp_values = [0]
for value in settlement_value:
    temp_values.append(value)
data['settlement'] = temp_values
# data['settlement'] =
data['ammount_of_increase'] = (data['close']-data['settlement'])/data['settlement']*100

if __name__ == '__main__':
    pass




