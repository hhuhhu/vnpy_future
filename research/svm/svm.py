# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: svm.py
@time: 2017/11/19 19:40
"""
from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import statsmodels.tsa.stattools as ts
from scipy import stats
import pywt
from nov_task.data_get import data_get
def wavelet():
    file_path = 'f:/NovTask/XAUUSDh4.csv'
    data = data_get(file_path)

    fig = plt.figure()
    ax1 = fig.add_subplot(711)
    ax2 = fig.add_subplot(712)
    ax3 = fig.add_subplot(713)
    ax4 = fig.add_subplot(714)
    ax5 = fig.add_subplot(715)
    ax6 = fig.add_subplot(716)
    ax7 = fig.add_subplot(717)

    optimize_date = '2016.12.20'
    test = data[data.date<=optimize_date]
    train_lenth = len(test)
    test = test['close']
    test = test.astype(float)
    test.plot(ax=ax5)
    print('train_length: ', train_lenth)
    data_length=len(data)
    total_predict_data=data_length - train_lenth

    data_close = data['close']
    close_dtw = pywt.dwt(data_close, 'db4')
    ax6.plot(close_dtw[0])
    ax7.plot(close_dtw[1])
    plt.show()
    print('close_dtw: ', close_dtw)
    print('close_dtw[0]:', close_dtw[0])
    print('type: ', type(close_dtw))

file_path = 'f:/NovTask/XAUUSDh4.csv'
data = data_get(file_path)
data['']
optimize_date = '2016.12.20'
train = data[data.date<=optimize_date]
data_length = len(data)
train_length = len(train)
test_length = data_length-train_length
clf = svm.SVC()
