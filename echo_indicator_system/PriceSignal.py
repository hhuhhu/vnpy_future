# coding=utf-8
"""实现价格序列转化为高低点的理想交易信号"""
from numpy.core import ndarray
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extreme_point(t_series):
    """计算序列的极点，返回极点的位置list"""
    assert isinstance(t_series, (list, ndarray)), "输入的序列必须是一维的list或者ndarray"
    i = 2   # 从第三个数据开始遍历
    landmark = []
    while i < len(t_series):
        if (t_series[i] - t_series[i-1]) * (t_series[i-1] - t_series[i-2]) < 0:
            landmark.append(i)
        i += 1
    return landmark


def convert_trading_signal(t_series, bins):
    trading_signal = [np.nan] * len(t_series)  # 交易信号序列
    trading_signal[0], trading_signal[-1] = 0.5, 0.5
    for b in range(len(bins) - 1):
        segment = t_series[bins[b]:bins[b + 1]]
        s_num = len(segment)
        # uptrend
        if t_series[bins[b]] <= t_series[bins[b + 1]]:
            for p, s in enumerate(segment):
                if p <= (s_num / 2):
                    trading_signal[bins[b] + p] = 0.5 - p / s_num
                else:
                    trading_signal[bins[b] + p] = (p + 1) / s_num - 0.5
        # downtrend
        else:
            for p, s in enumerate(segment):
                if p <= (s_num / 2):
                    trading_signal[bins[b] + p] = 0.5 + p / s_num
                else:
                    trading_signal[bins[b] + p] = 1.5 - (p + 1) / s_num

    return trading_signal


def MDPP(t_series, distance, vib_percentage, time_unit='M', convert_trading_signals=True):
    """
    reference:
    Perng, C.S., Wang, H., Zhang, S.R., & Parker, D.S. (2000). Landmarks: a new model for similarity based pattern 
    querying in time series databases, In Proceedings of the sixteenth international conference on data engineering. 
    San Diego, USA.
    Notes:
        这种方法找到的并不是最高点，而是根据选择投资周期和风险幅度来计算，不同的值，产生的结果将会不相同

    Parameters
    ----------
    t_series:[DataFrame, Series],一列时间序列
    distance:[int]，时间间隔
    vib_percentage:[double], 风险幅度，两个极点的价格百分比小于这个幅度时，去除，可以理解成投资周期内愿意承担的风险或收益
    time_unit:[str,"M", "H", "D", "W"]，时间单位
    convert_trading_signals:[bool]，是否转化为信号
    """
    time_list = t_series.index
    t_series = t_series.values
    res = [np.nan] * len(t_series)              # 不满足条件的点均为nan
    time_trans_unit = {"M": 60, "H": 3600, "D": 3600 * 24, "W": 3600 * 24 * 7}
    # 计算序列的极点序列
    landmark = extreme_point(t_series)
    i = 1
    bins = []
    while i < len(landmark):
        pos, last_pos = landmark[i], landmark[i-1]
        # 满足条件的landmark点
        if (time_list[pos] - time_list[last_pos]).total_seconds() / time_trans_unit[time_unit] < distance and\
                (abs(t_series[pos] - t_series[last_pos]) * 2 / (t_series[pos] + t_series[last_pos]) < vib_percentage):
            pass
        else:
            res[pos] = t_series[pos]
            bins.append(pos)
        i += 1

    if convert_trading_signals:
        trading_signal = convert_trading_signal(t_series, bins)
        res = pd.DataFrame(data=res, index=time_list, columns=["trading_point"])
        res.loc[:, "signal"] = trading_signal
        res["signal"] = res["signal"].interpolate()
        return res, bins

    return pd.DataFrame(data=res, index=time_list, columns=["trading_point"]), bins


def point_line_distance(line_point_1, line_point_2, point):
    """
    计算点到两点线的距离
    Notes:本函数仅计算坐标原点出发的直线距离
    """
    k = (line_point_2[1] - line_point_1[1]) / line_point_2[0] - line_point_1[0]  # 计算斜率
    return abs((k * (point[0] - line_point_1[0]) + line_point_1[1] - point[1]) / np.sqrt(k * k + 1))


def piecewise_linear(time_series, threshold, convert_trading_signals=True):
    """
    P.C. Chang, C.Y. Fan, C.H. Liu, Integrating a piecewise linear representation method and a neural network model 
    for stock trading points prediction, IEEETrans. Syst. Man Cybern. Part C: Apply. Rev. 39 (2009) 80–92

    Parameters
    ----------
    time_series:[Series],时间序列
    threshold:[double], 识别的最小距离
    convert_trading_signals: 是否转为0到1之间的交易信号
    """
    t_series = time_series.values               # 将series转化为array减少计算时间
    res = [np.nan] * len(t_series)                  # 划分好后的价格序列点
    res[0],  res[-1] = t_series[0], t_series[-1]    # 首尾端点
    bins = [0, len(t_series) - 1]                   # 存储每个区间的左右两端下标
    i = 0
    while i < len(bins) - 1:
        segment = t_series[bins[i]:bins[i + 1]]
        # 计算期间内与直线的距离最大的位置,注意每个区间的x坐标点均从0开始
        dist = [point_line_distance((0, segment[0]), (len(segment) - 1, segment[-1]), (p, values))
                for p, values in enumerate(segment)]

        max_dist = np.max(dist)
        max_dist_arg = np.argmax(dist)
        pos = bins[i] + max_dist_arg
        if max_dist > threshold:
            bins.insert(i+1, pos)
            res[pos] = t_series[pos]
        else:
            i += 1

    if convert_trading_signals:
        trading_signal = convert_trading_signal(t_series, bins)
        res = pd.DataFrame(data=res, index=time_series.index, columns=["trading_point"])
        res.loc[:, "signal"] = trading_signal
        return res, bins

    return pd.DataFrame(data=res, index=time_series.index, columns=["trading_point"]), bins


if __name__ == '__main__':
    import tushare
    data = tushare.get_hist_data('002230')
    fig, ax = plt.subplots()
    data['close'].plot()
    # new_series, _ = MDPP(reader.data['XAUUSD']['open'], 10, 0.10)
    new_series, bins = piecewise_linear(data['close'], 3)
    print(new_series["signal"].iat[bins[1]], new_series["signal"].iat[bins[-1]])
    new_series["trading_point"].plot(ax=ax, legend=False, linestyle='dashed', marker='o', markerfacecolor='red',
                                     markersize=4)
    new_series["signal"].plot(ax=ax, linestyle='--', secondary_y=True, color='g')
    plt.show()
