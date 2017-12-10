# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: data_get.py
@time: 2017/11/20 16:33
"""
import os
import sys
from datetime import datetime
import csv
import chardet

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import date2num
import matplotlib.finance as mpf

from research.rnn.rnn import data_frequency_transfer


def data_get(file_path):
    header = ['date', 'open', 'high', 'low', 'close']
    data = pd.read_csv(file_path, names=header, encoding='utf-16', delimiter='\t')
    # data = pd.read_excel(file_path)
    start = '2015.11.19 00:00:00'
    end = '2017.11.20 08:00:00'
    # start = datetime.strptime(start, "%Y.%m.%d")
    data = data[data.date >= start]
    data = data[data.date <= end]
    data.set_index(data.date, inplace=True)
    data.sort_values('date', inplace=True)
    return data


def cov_analysis():
    symbols = ["XAUUSD", "XAGUSD", "BRN", "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
    df = pd.DataFrame()
    for symbol in symbols:
        file_path = 'F:/NovTask/{}h4.csv'.format(symbol)
        data = data_get(file_path)
        df[symbol] = data['close']

    df = df.pct_change()
    corr = df.corr()
    corr.to_csv('F:/NovTask/corr.csv')


def cov_analysis_v2():
    symbols = ["XAUUSD", "XAGUSD", "BRN", "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
    df = pd.DataFrame()
    for symbol in symbols:
        file_path = 'F:/NovTask/{}h4.csv'.format(symbol)
        data = data_get(file_path)
        df[symbol] = data['close']
        df[symbol] = (df[symbol] - df[symbol].mean()) / df[symbol].std()
        corr = df.corr()
        corr.to_csv('F:/NovTask/corr_v2.csv')


def close_plot():
    symbols = ["XAUUSD", "XAGUSD", "BRN", "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
    df = pd.DataFrame()
    for symbol in symbols:
        file_path = 'F:/NovTask/{}h4.csv'.format(symbol)
        data = data_get(file_path)
        df[symbol] = data['close']

        df[symbol].plot()
        plt.title('{}_2015.11.19_2015.11.20_h4'.format(symbol))
        plt.savefig('F:/NovTask/{}h4.png'.format(symbol))
        plt.close()
    #     plt.legend()
    # plt.show()


def close_plot_v2():
    symbols = ["XAUUSD", "XAGUSD", "BRN", "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
    colors = ['black', 'blue', 'green', 'gray', 'red', 'yellow','orange', 'mediumaquamarine', 'pink', 'thistle']
    df = pd.DataFrame()
    for i,symbol in enumerate(symbols):
        file_path = 'F:/NovTask/{}h4.csv'.format(symbol)
        data = data_get(file_path)
        df[symbol] = data['close']
        df[symbol] = (df[symbol] - df[symbol].mean())/df[symbol].std()
        df[symbol].plot(color=colors[i])
        plt.legend()
    plt.title('Close by Zscore from 2015.11.19 to 2017.11.20')
    plt.savefig('F:/NovTask/Close by Zscore from 2015.11.19 to 2017.11.20.png')
    plt.show()


def candle_plot():
    symbols = ["XAUUSD", "XAGUSD", "BRN", "EURUSD", "USDJPY", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
    df = pd.DataFrame()
    for symbol in symbols:
        file_path = 'F:/NovTask/{}h4.csv'.format(symbol)
        data = data_get(file_path)
        # df[symbol] = data['close']
        data_list = []
        for dates, row in data.iterrows():
            # 将时间转换为数字
            # date_time = datetime.datetime.strptime(dates, '%Y-%m-%d')
            # print(dates, row)
            t = datetime.strptime(dates,'%Y.%m.%d %H:%M:%S')
            t = date2num(t)
            open, high, low, close = row[1:5]
            datas = (t, open, high, low, close)
            data_list.append(datas)
        # 创建一个子图
        fig, ax = plt.subplots(facecolor=(0.5, 0.5, 0.5))
        fig.subplots_adjust(bottom=0.2)
        # 设置X轴刻度为日期时间
        ax.xaxis_date()
        # X轴刻度文字倾斜45度
        plt.xticks(rotation=45)
        plt.title("{}_20151119_20171120".format(symbol))
        plt.xlabel("Date")
        plt.ylabel("Price")
        mpf.candlestick_ohlc(ax, data_list, width=1.2, colorup='r', colordown='green')
        plt.grid(True)
        plt.savefig("f:/NovTask/{}_20151119_20171120_4h.png".format(symbol))
        plt.show()
        plt.close()


if __name__ == '__main__':
    # data = data_get()
    # cov_analysis()
    # close_plot()
    # close_plot_v2()
    # cov_analysis_v2()
    candle_plot()
