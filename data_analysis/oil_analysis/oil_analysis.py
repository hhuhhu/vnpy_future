# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: oil_analysis.py
@time: 2017/9/29 10:59
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel('E://XTI//XTIUSD.xlsx')
print(data)
def close_plot():
    data['close'].plot()
    plt.title('ru_close_1998-01-05_2017-09-28_daily')
    plt.savefig('E:/ru_close.png')

    plt.show()


def increase_plot():
    plt.plot(data['increase'].values)
    plt.title(u'Increase of RU-Index')
    plt.yticks(np.linspace(-7,7,15))
    plt.ylabel('Increase * 100')
    plt.savefig('E:/ru_increase.png')
    plt.show()
if __name__ == '__main__':
    close_plot()