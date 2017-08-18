# coding=utf-8
"""指标根据输入的参数生成的买卖信号的函数"""
import talib


def ma_cross(apply_price, N, n, a1, b1, c1, a2, b2, c2):
    """
    黄金交叉和死亡交叉
    z = MA(N) - MA(n)
    买入信号：
        若z[t] >= 0 ，找到最近的j, j<t, 如果z[j-1]<=0<z[j], Mz=max(z[j],z[j+1], ...,z[t]), 若mz > b1 * c1 且
    z[i] < min(mz/a1, c1)
    卖出信号：
        若z[t] < 0 ，找到最近的j, j<t, 如果z[j-1]>=0>z[j], Mz=max(-z[j],-z[j+1], ...,-z[i]), 若mz > b1 * c1 且
    z[i] < min(mz/a1, c1)
    Parameters
    ----------
    apply_price:[array, list], 计算的价格序列
    N:[int], ma的长周期
    n:[int], ma的短周期
    a1:[double]
    b1:[double]
    c1:[double]
    a2:[double]
    b2:[double]
    c2:[double]
    """
    long_ma = talib.MA(apply_price, timeperiod=N)
    short_ma = talib.MA(apply_price, timeperiod=n)
    z = long_ma - short_ma
    _signal_pos = []
    index = 2

    while index < len(z):
        item = z[index]
        flag = False
        # 买入点
        if item >= 0:
            i = 1
            while (index - i) >= 0:
                if z[index - i - 1] <= 0 < z[index - i]:
                    flag = True
                    break
                i += 1
            if flag:
                mz = max(z[(index - i):(index + 1)])
                if mz > b1 * c1 and item < min(mz / a1, c1):
                    _signal_pos.append((index, -1))
        # 卖出点
        else:
            i = 1
            while (index - i) >= 0:
                if z[index - i - 1] >= 0 > z[index - i]:
                    flag = True
                    break
                i += 1
            if flag:
                mz = max(map(lambda x: -x, z[(index - i):(index + 1)]))
                if mz > b2 * c2 and item < min(mz / a2, c2):
                    _signal_pos.append((index, 1))
        index += 1
    return _signal_pos


def envelope(apply_price, period, uBandWith, lBandWith):
    """

    Parameters
    ----------
    apply_price:价格序列
    period:ma的周期
    uBandWith:maEnvelope上轨宽度
    lBandWith:maEnvelope下轨宽度
    """
    ma = talib.MA(apply_price, timeperiod=period)
    _signal_pos = []
    p1 = ma + uBandWith
    p2 = ma - lBandWith
    for i in range(len(ma)):
        # 买入信号
        if apply_price[i] < p2[i]:
            _signal_pos.append((i, -1))
            continue
        # 卖出信号
        if apply_price[i] > p1[i]:
            _signal_pos.append((i, 1))
            continue
    return _signal_pos


def rsi(apply_price, period, over_buy, over_sell, slope_p, slope_q):
    """
    Parameters
    ----------
    apply_price:价格序列
    period:RSI的周期
    over_buy:超过此指认为超买，将下跌
    over_sell:超过此值认为炒卖，将上涨
    slope_p:rsi超过炒买值后，从超买值点开始画一条斜线，该线的斜率为slope_p
    slope_q:rsi超过炒卖值后，从超买值点开始画一条斜线，该线的斜率为slope_q
    """
    rsi = talib.RSI(apply_price, timeperiod=period)
    _signal_pos = []
    for index, item in enumerate(rsi):
        # 卖出点：当指标达到超买线，从超买值点开始画一条斜线的斜率小于slope_p
        if item > over_buy:
            zero1 = index
            value1 = item + slope_p * (index - zero1)
            if item < value1:
                _signal_pos.append((index, 1))
                continue
        # 买入点：当指标达到超卖线，从超买值点开始画一条斜线的斜率大于slope_q
        if item < over_sell:
            zero2 = index
            value2 = item - slope_q * (index - zero2)
            if item > value2:
                _signal_pos.append((index, -1))
                continue
    return _signal_pos


def roc(apply_price, long_period, short_period, u_border, l_border, u_band, l_band):
    """
    Parameters
    ----------
    apply_price:价格序列
    long_period:长期roc的周期
    short_period:短期roc的周期
    u_border:长期roc的上界
    l_border:长期roc的下界
    u_band:均衡线上轨的宽度
    l_band:均衡线下轨的宽度
    """
    long_roc = talib.ROC(apply_price, timeperiod=long_period)
    short_roc = talib.ROC(apply_price, timeperiod=short_period)
    _signal_pos = []
    u_equilibrium = 100 + u_band
    l_equilibrium = 100 - l_band

    for index in range(len(apply_price)):
        # 卖出点:当长期ROC大于上界且短期ROC在均衡线附近
        if long_roc[index] > u_border and (l_equilibrium <= short_roc[index] <= u_equilibrium):
            _signal_pos.append((index, 1))
            continue
        # 买入点:当长期ROC小于下界且短期ROC在均衡线附近
        if long_roc[index] < l_border and (l_equilibrium <= short_roc[index] <= u_equilibrium):
            _signal_pos.append((index, -1))
            continue
    return _signal_pos


def stochastic_oscillator(time_series, a, b, c, d):
    """随机震荡指标"""
    _signal_pos = []
    _k, _d = talib.STOCH(time_series["close"].values, time_series["high"].values,
                         time_series["low"].values)
    for index in range(len(time_series)):
        if _k[index] > _d[index] and (_k[index] < a and _k[index] - _d[index] < b):
            _signal_pos.append((index, -1))
            continue
        if _k[index] < _d[index] and (_k[index] > c and _d[index] - _k[index] < d):
            _signal_pos.append((index, 1))
            continue
    return _signal_pos


def hammer(time_series, a):
     hammmer = talib.CDLHAMMER(time_series["open"].values, time_series["high"].values,
                              time_series["low"].values, time_series["close"].values)


