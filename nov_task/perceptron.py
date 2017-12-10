# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: perceptron.py
@time: 2017/11/25 15:54
"""
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from nov_task.data_get import data_get

file_path = 'F:/NovTask/XAUUSDh4.csv'
data = data_get(file_path)
data['label'] = 1
data['returns'] = (data['open'] - data['close'].shift(160))/data['close'].shift(160)
data.loc[data['returns'] >= 0.01, 'label'] = 0 # buy
data.loc[data['returns'] <= -0.01, 'label'] = 2 # sell
data['ma'] = data['close'].rolling(62).mean()
data.dropna(inplace=True)
dataset = []
close = data['close'].values
for i in range(len(close)-62):
    dataset.append(close[i:i+62])
label = data['label'].values[62:]
# print(len(dataset),":", len(label))
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2)
clf = Perceptron()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
accurcy = accuracy_score(y_test, y_predict)
print(y_predict)
print('accurcy: ', accurcy)