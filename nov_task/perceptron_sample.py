# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: perceptron_sample.py
@time: 2017/11/28 14:04
"""
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

iris = datasets.load_iris()  # http://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
X = iris.data[:, [2, 3]]
y = iris.target  # 取species列，类别

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)  # train_test_split方法分割数据集

sc = StandardScaler()  # 初始化一个对象sc去对数据集作变换
sc.fit(X_train)  # 用对象去拟合数据集X_train，并且存下来拟合参数
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron()  # y=w.x+b
ppn.fit(X_train_std, y_train)


# 验证perceptron的原理
def prelabmax(X_test_std):
    pym = []
    for i in range(X_test_std.shape[0]):
        py = np.dot(ppn.coef_, X_test_std[i, :].T) + ppn.intercept_
        pym.append(max(py))
    return pym

prelabmax(X_test_std)

def prelabindex(X_test_std, pym):
    index = []
    for i in range(X_test_std.shape[0]):
        py = np.dot(ppn.coef_, X_test_std[i, :].T) + ppn.intercept_
        pymn = pym[i]
        for j in range(3):
            if py[j] == pymn:
                index.append(j)
    return np.array(index)
pym = prelabmax(X_test_std)
prelabindex(X_test_std, pym)
prelabindex(X_test_std, pym) == ppn.predict(X_test_std)
