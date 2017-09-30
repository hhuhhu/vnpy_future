#! /user/bin/python
# -*-coding:UTF-8-*-
import os
from datetime import datetime
from itertools import islice

import numpy as np
import pandas as pd
import stats as sts
import matplotlib.pyplot as plt
import tushare as ts

from data_analysis.dtw.dtw import dtw


def medscale(s):
    '''

    :param s: 向量
    :return: 以s的中位数归一化, 全部除以序列的中位数
    '''
    medval = np.median(s)
    if medval == 0:
        medval = 0.001  # 若中位数为0，由于除数不能为0，令中位数等于一个极小值
    s_scaled = pd.Series(map(lambda x: x / medval, s))
    return s_scaled


def window(s, n):
    '''

    :param s: 序列
    :param n: 窗口长度
    :return: s以n为窗口长度, 步长为1滑动得到的排列
    '''
    z = (islice(range(len(s)), i, None) for i in range(n))  # islice(a,b,c)的用法是对a序列从b位置至c位置切片
    # result e.g: [(0,1,2,3,4),(1,2,3,4),(2,3,4),(3,4)]
    return list(zip(*z))  # result e.g: [0,1,2,3][1,2,3,4]


def core_dtwA(s1, s2):
    '''

    :param s1: df
    :param s2: df
    :param k: number, 阈值
    :return: 输出s2所有窗口期内dtw值最小的窗口编号, 并输出s2所有窗口，所有dtw值
    '''
    s1_scaled = s1.apply(medscale)  # 对s1的每一列应用medscale函数，将时间序列归一化处理
    n = len(s1)  # 设置历史时间序列的窗口长度等于待匹配序列
    s2_win = window(s2, n)  # 得到待匹配序列的滑动窗口序号
    val = []  # val值储存每一个历史序列匹配得到的dtw值
    for i in range(0, len(s2_win), 5):  # 以5为间隔滑动，加快运算速度
        s2_ch = s2.iloc[s2_win[i], :]
        s2_scaled = s2_ch.apply(medscale)  # 对s1的每一列应用medscale函数，将时间序列归一化处理
        dtw_temp, cost, acc, path = dtw(np.array(s1_scaled), np.array(s2_scaled),
                                        dist=lambda x, y: np.linalg.norm(x - y, ord=1)
                                        )
        val.append(dtw_temp)
    x = [idx for idx, n in enumerate(val) if n == min(val)]  # 找出val中最小值的所有位置
    return x, s2_win, val


def matfig(s1, s2, s1_next, s2_next, sl, i, test_date, val1):
    '''

    :param s1: 待匹配序列
    :param s2: 历史匹配序列
    :param s1_next: 待匹配序列的后续走势序列
    :param s2_next: 历史序列的后续走势序列
    :param sl: 历史序列后续走势的时间段
    :param i: 编号, 目的是区分图像
    :param test_date: 待匹配序列开始时点
    :return: 左图中红色的是待匹配序列, 蓝色是匹配得到的历史序列; 右图是后续走势对比, 横坐标为时间点
    '''
    s1_scaled = s1.apply(medscale)
    s2_scaled = s2.apply(medscale)
    s1_next_s = s1_next.apply(medscale)
    s2_next_s = s2_next.apply(medscale)  # 归一化
    val2, cost, acc, path = dtw(np.array(s1_next_s), np.array(s2_next_s), dist=lambda x, y: np.linalg.norm(x - y))

    plt.figure(i, figsize=(12, 6))
    p1 = plt.subplot(1, 2, 1)
    p1.plot(range(len(s1_scaled.iloc[:, 0])), s1_scaled.iloc[:, 0], 'r')
    p1.plot(range(len(s2_scaled.iloc[:, 0])), s2_scaled.iloc[:, 0], 'b')
    plt.title('DTW (%.4f)' % val1)
    p2 = plt.subplot(1, 2, 2)
    plt.title(test_date + 'DTW (%.4f)' % val2)
    if int(len(sl) / 4):
        _step = int(len(sl) / 4)
    else:
        _step = 1
    xticks = range(0, len(sl), _step)  # make ticks and tick labels
    xticklabels = [sl.iloc[i] for i in xticks]
    p2.plot(range(len(s1_next_s.iloc[:, 0])), s1_next_s.iloc[:, 0], 'r')
    p2.plot(range(len(s2_next_s.iloc[:, 0])), s2_next_s.iloc[:, 0], 'b')
    p2.set_xticks(xticks)  # set ticks and tick labels
    p2.set_xticklabels(xticklabels, rotation=-20)
    # save the figure
    xlabel = xticklabels[0].replace('/', '')
    xlabel = xlabel.replace(':', '')
    xlabel = xlabel.strip()
    plt.savefig(test_date + '/' + test_date + 'in' + xlabel + '.png')
    plt.close()
    return val2


def matfignd(s1, s2, s1_next, s2_next, sl, i, test_date, val1):
    s1_scaled = s1.apply(medscale)
    s2_scaled = s2.apply(medscale)
    s1_next_s = s1_next.apply(medscale)
    s2_next_s = s2_next.apply(medscale)
    val2, cost, acc, path = dtw(np.array(s1_next_s), np.array(s2_next_s), dist=lambda x, y: np.linalg.norm(x - y))

    plt.figure(i, figsize=(12, 6))

    p1 = plt.subplot(3, 2, 1)
    p1.plot(range(len(s1_scaled.iloc[:, 0])), s1_scaled.iloc[:, 0], 'r')
    p1.plot(range(len(s2_scaled.iloc[:, 0])), s2_scaled.iloc[:, 0], 'b')
    plt.title('DTW (%.4f)' % val1)

    p3 = plt.subplot(4, 2, 5)
    p3.bar(range(len(s1_scaled.iloc[:, 1])), s1_scaled.iloc[:, 1], color='r')
    p4 = plt.subplot(4, 2, 7)
    p4.bar(range(len(s2_scaled.iloc[:, 0])), s2_scaled.iloc[:, 0], color='b')

    p2 = plt.subplot(3, 2, 2)
    plt.title(test_date + 'DTW (%.4f)' % val2)
    if int(len(sl) / 4):
        _step = int(len(sl) / 4)
    else:
        _step = 1
    xticks = range(0, len(sl), _step)  # make ticks and tick labels
    xticklabels = [sl.iloc[i] for i in xticks]
    p2.plot(range(len(s1_next_s.iloc[:, 0])), s1_next_s.iloc[:, 0], 'r')
    p2.plot(range(len(s2_next_s.iloc[:, 0])), s2_next_s.iloc[:, 0], 'b')
    p2.set_xticks(xticks)  # set ticks and tick labels
    p2.set_xticklabels(xticklabels, rotation=-20)
    # save the figure
    xlabel = xticklabels[0].replace('/', '')
    xlabel = xlabel.replace(':', '')
    xlabel = xlabel.strip()

    p5 = plt.subplot(4, 2, 6)
    p5.bar(range(len(s1_next_s.iloc[:, 1])), s1_next_s.iloc[:, 1], color='r')
    p6 = plt.subplot(4, 2, 8)
    p6.bar(range(len(s2_next_s.iloc[:, 0])), s2_next_s.iloc[:, 0], color='b')

    plt.savefig(test_date + '/' + test_date + 'in' + xlabel + '.png')
    plt.close()
    return val2


def kusk(t_win, stockdata, indexes):
    '''

    :param t_win: 窗口长度
    :param stockdata: 股票序列
    :param indexes: 遍历下标
    :return: 滑动窗口得到股票序列的峰度、偏度、波动率、收益率指标
    '''
    sk = []
    ku = []
    std = []
    res = []
    for j in indexes:
        s = stockdata.iloc[-j - 2 * t_win:-j - t_win]
        print("s_length: ", len(s))
        sk.append(sts.skewness(s))
        ku.append(sts.kurtosis(s))
        std.append(np.std(s))
        res.append((s.iloc[-1] - s.iloc[0]) / s.iloc[0])

    df = pd.DataFrame(
        {"时间": pd.Series(s.date), "峰度": pd.Series(sk), "偏度": pd.Series(ku), "波动率": pd.Series(std),
         "收益率": pd.Series(res)})
    return df


def main():
    t_win = 10
    m = 2  # 预测窗口长度
    # stockdata = pd.read_csv('000001SH2000.csv')
    stockdata = ts.get_hist_data('000001')
    stockdata['date'] = stockdata.index
    columns = ['date', 'open', 'high', 'close', 'low']
    stockdata = pd.DataFrame(data=stockdata,columns=columns)
    stockdata.sort_index(inplace=True)
    print("stockdata_length: ", len(stockdata))
    # print('stock_data: ', stockdata)
    dtw_now = []
    dtw_next = []
    dtw_next_date = []
    for l in range(1, 400, 20):
        print("location: ", l)  # 显示循环位置
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))  # 显示当前时间

        s1 = stockdata.iloc[-l - 2 * t_win:-l - t_win, 1:]  # s1是待匹配序列的多维时间序列
        s2 = stockdata.iloc[:-l - 2 * t_win, 1:]  # s2应当是一个多维时间序列, 时间小于s1
        print('s1_length:{};s2_length:{}'.format(len(s1), len(s2)))
        s2_date = stockdata.iloc[:-l - 2 * t_win, 0]  # s2_date是s2的时间标签
        s1_next = stockdata.iloc[-l - t_win:-l, 1:][0:m]

        test_date = stockdata.date.iloc[-l - t_win]
        test_date = test_date.replace('/', '-')
        if not os.path.exists(test_date):
            os.mkdir(test_date)

        a, b, c = core_dtwA(s1, s2)
        print(a)

        for i in range(len(a)):
            s2_now = s2.iloc[b[a[i]], :]
            s2_next = s2.iloc[[b[a[i]][j] + t_win for j in range(len(b[a[i]]))][0:m], :]
            s2labels = s2_date.iloc[[b[a[i]][j] + t_win for j in range(len(b[a[i]]))][0:m]]
            val1 = c[i]
            if len(s2.iloc[1, :]) == 1:
                val2 = matfig(s1, s2_now, s1_next, s2_next, s2labels, i + l, test_date, val1)
            else:
                val2 = matfignd(s1, s2_now, s1_next, s2_next, s2labels, i + l, test_date, val1)
            dtw_now.append(val1)
            dtw_next.append(val2)
            dtw_next_date.append(s2labels.iloc[0])

    c1 = pd.Series(dtw_now)
    c2 = pd.Series(dtw_next)
    df_dtw_next = pd.DataFrame({"时间段": pd.Series(dtw_next_date), "匹配dtw": c1, "后续dtw": c2})
    df_dtw_next.to_csv('dtwnext.csv', index=False)

    df_kusks = kusk(t_win, stockdata.iloc[:, 1], range(1, len(stockdata), 20))
    df_kusks.to_csv('dddnow.csv', index=False)

if __name__ == '__main__':
    main()
