# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: ag_analysis.py
@time: 2017/10/25 15:12
"""
import sys
import os
import datetime

import pandas as pd
import matplotlib

# 解决中文乱码问题
matplotlib.use('qt5agg')
# 指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.family'] = 'sans-serif'
# 解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
from matplotlib.pylab import date2num

from research.rnn.rnn import fix_data, data_frequency_transfer


def ths_data_handle():
    file_path = os.path.join(sys.path[0], 'ag88.xlsx')
    data = pd.read_excel(file_path)
    a = data['date'].values
    a = [a[i].split(',')[0] for i in range(len(a))]
    data['date'] = a
    data = data.sort_values('date')
    test = data['date'].values[0]
    print(type(test))
    data = data[data.date < '2017-06-03']
    return data


def candle_plot(data):
    data_list = []
    for dates, row in data.iterrows():
        print('dates: ', dates)
        print('dates_type: ', type(dates))
        # 将时间转换为数字
        # date_time = datetime.datetime.strptime(dates, '%Y-%m-%d')
        t = date2num(dates)
        print("t_type: ", type(t))
        print('t: ', t)
        open, high, low, close = row[0:4]
        datas = (t, open, high, low, close)
        data_list.append(datas)
    # 创建一个子图
    fig, ax = plt.subplots(facecolor=(0.5, 0.5, 0.5))
    fig.subplots_adjust(bottom=0.2)
    # 设置X轴刻度为日期时间
    ax.xaxis_date()
    # X轴刻度文字倾斜45度
    plt.xticks(rotation=45)
    plt.title("ag88_20140101_20161221")
    plt.xlabel("Date")
    plt.ylabel("Price")
    mpf.candlestick_ohlc(ax, data_list, width=1.2, colorup='r', colordown='green')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # candle_plot(ths_data_handle())
    tmp = fix_data('白银88.csv')

    # targets 1d 数据合成
    tmp_1d = data_frequency_transfer(tmp, '1d')
    candle_plot(tmp_1d)