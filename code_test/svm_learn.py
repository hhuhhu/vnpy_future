# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: svm_learn.py
@time: 2017/11/27 21:05
"""
from sklearn import svm
X = [[0], [1], [2]]
Y = [0, 1, 2]
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(X,Y)
a = clf.predict(0)
# print(a)
a = list(range(65))
dataset = []
for i in range(len(a)-62):
    dataset.append(a[i:i+62])
print("a: ", a)
print('dataset: ', dataset)
print("lengh: ", len(dataset))
print(dataset[0])