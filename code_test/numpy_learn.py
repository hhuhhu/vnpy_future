# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: numpy_learn.py
@time: 2017/11/2 15:28
"""
import numpy as np


def np_prod_learn(data):
    return np.prod(data)

if __name__ == '__main__':
    data = [1.,2.]
    data = [[1.,2.],[2.,2.]]
    data = np.array([536870910, 536870910, 536870910, 536870910])
    data = [10000000,10000000,10000000,10000000]
    print(data)
    print(np_prod_learn(data))