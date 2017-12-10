# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: ru_index_analysis.py
@time: 2017/9/28 18:39
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel('E://ru//test.xlsx')

data['increase'] = data['increase'] * 100
increase = data['increase'].values
increase[0] = 0
data['increase'] = increase
increase_sort = data['increase'].sort_values()
print(increase_sort)
limit_down = pd.DataFrame(columns=['time','open', 'high', 'low', 'close', 'increase', 'amplitude', 'volume', 'turnover'])
# for row in [4708, 4794, 4625, 4589, 4673, 4691, 3196, 4750, 1267, 4639]:
#     limit_down.append(data.iloc[[row]], ignore_index=True)
#     print(data.iloc[[row]])
# print(limit_down)

for row in [2657,3350,4629,3359,4200,2033]:
    print(data.iloc[[row]])


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
    # close_plot()
    pass