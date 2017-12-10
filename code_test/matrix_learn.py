# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: matrix_learn.py
@time: 2017/11/3 11:06
"""
import numpy as np
a = [[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]
b = a*2
# print(a)
# print(b)
c = np.stack(b, 1)
a = np.arange(1,13).reshape((2,2,3))
b = np.arange(13,25).reshape((2,3,2))
c = np.matmul(a, b)
d = np.matmul(b,a)
print(a)
print(a.shape)
print(b)
print(b.shape)
print(c)
print(c.shape)
print(d)
print(d.shape)