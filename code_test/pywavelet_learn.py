# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: pywavelet_learn.py
@time: 2017/11/25 13:00
"""
import matplotlib.pyplot as plt
import pywt
x = [3,7,1,1,-2,5,4,6]
plt.plot(x)
plt.show()
cA,cD = pywt.dwt(x, 'db2')
print(cA, '\n', cD)
plt.plot(cA, cD)
plt.show()
w = pywt.Wavelet('sym3')
cA, cD = pywt.dwt(x, wavelet=w, mode='constant')