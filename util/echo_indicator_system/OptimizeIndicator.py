# coding=utf-8
"""
重现和修改了 Xiaowei Lin, Zehong Yang, Yixu Song,2011,Intelligent stock trading system based on improved technical analysis and
Echo State Network 关于遗传算法优化指标的方式
"""
from __future__ import division
from deap import creator, base, tools, algorithms
from data_analysis.indicator_analysis.IndicatorAnalyst import *
from data_analysis.indicator_analysis.IndicatorSignal import *
import random
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import multiprocessing

# 创建适应度类为求最小值
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# 创建具有适应度属性的个体
creator.create("Individual", list, fitness=creator.FitnessMin)


def indicator_fitness(p_signal, signal_bins, price, ind_func, parameters):
    """
    计算指标参数的适应度
    Parameters
    ----------
    p_signal:[DataFrame], 价格反转点，非反转点为NaN
    signal_bins:[list]，存储了价格反转点的位置坐标
    price:[array,list]，价格序列，用于计算指标
    ind_func:[func],IndicatorSignal中的指标信号产生函数
    parameters:[list]，指标的参数
    """
    t = signal_bins
    t_pos, s_pos = 1, 0  # 期望买卖点list的下标，指标信号的下标
    expect_signal = p_signal['trading_point'].values
    ind_signal = ind_func(price, *tuple(parameters))
    fitness = []
    _close = price
    while t_pos < len(t) - 2:
        # 期望买点
        if expect_signal[t[t_pos]] - expect_signal[t[t_pos - 1]] < 0:
            s_count = 0
            while s_pos < (len(ind_signal) - 1) and ind_signal[s_pos][0] < t[t_pos + 1]:
                # 建议买点
                if ind_signal[s_pos][1] == -1:
                    fitness.append(_close[ind_signal[s_pos][0]] - _close[t[t_pos]])
                    s_count += 1
                # 建议买点
                else:
                    fitness.append(2 * (_close[t[t_pos]] - min(_close[slice(t[t_pos - 1], t[t_pos + 1])])))
                    s_count += 1
                s_pos += 1
            # 不存在建议的信号点
            if s_count == 0:
                fitness.append(max(_close[slice(t[t_pos - 1], t[t_pos + 1])]) - _close[t[t_pos]])
        # 期望卖点
        else:
            s_count = 0
            while s_pos < (len(ind_signal) - 1) and ind_signal[s_pos][0] < t[t_pos + 1]:
                # 建议买点
                if ind_signal[s_pos][1] == -1:
                    fitness.append(2 * (_close[t[t_pos]] - min(_close[slice(t[t_pos - 1], t[t_pos + 1])])))
                    s_count += 1
                # 建议卖点
                else:
                    fitness.append(_close[t[t_pos]] - _close[ind_signal[s_pos][0]])
                    s_count += 1
                s_pos += 1
            # 不存在建议的信号点
            if s_count == 0:
                fitness.append(_close[t[t_pos]] - max(_close[slice(t[t_pos - 1], t[t_pos + 1])]))
        t_pos += 1
    return sum(fitness),


class IndicatorOptimize(object):
    """指标参数使用遗传算法优化"""
    def __init__(self, time_series, apply_price, process_num):
        self._time_series = time_series
        self._apply_price = time_series[apply_price].values
        self.signal, self.signal_bin = piecewise_linear(self._time_series[apply_price], 10, convert_trading_signals=False)
        self.pool = multiprocessing.Pool(process_num)
        self.toolbox = base.Toolbox()
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)

    def to_signal(self, _signal_pos):
        """转化为信号价格序列"""
        signal = [np.nan] * len(self._apply_price)
        for pos in _signal_pos:
            signal[pos[0]] = self._apply_price[pos[0]]
        return pd.DataFrame(data=signal, index=self._time_series.index, columns=['signal'])

    def plot_indicator_signal(self, apply_price, ind_func, *args):
        """
        绘制价格序列和指标的信号点，便于分析
        Parameters
        ----------
        apply_price:采用的价格
        ind_func:指标函数
        args:指标函数的参数
        """
        plt.close('all')
        fig, axe = plt.subplots()
        self._time_series[apply_price].plot(ax=axe)
        signal_pos = ind_func(*args)
        self.to_signal(signal_pos).plot(ax=axe, linestyle='dashed', marker='o', markerfacecolor='red', markersize=4)
        plt.show()

    def ma_gen_parameter(self):
        """ma交叉的参数产生器，即一个个体"""
        res = []
        N = random.randint(10, 400)
        n = N - random.randint(1, N - 1)
        z = talib.MA(self._apply_price, timeperiod=N) - talib.MA(self._apply_price, timeperiod=n)
        res.append(N)
        res.append(n)
        res.extend([random.uniform(np.nanmin(z), np.nanmax(z)) for _ in range(6)])
        return res

    def init_evolution(self, mu, lambda_, cxpb, mutpb, ngen):
        """初始化遗传算法，设置"""
        # 一套参数是一个个体，产生个体的编码
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.ma_gen_parameter)
        # 创建种群
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        # 计算适应度的函数
        self.toolbox.register("evaluate", indicator_fitness, self.signal, self.signal_bin, self._apply_price, ma_cross)
        # 交叉、变异和选择
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.5, indpb=0.5)
        self.toolbox.register("select", tools.selNSGA2)
        pop = self.toolbox .population(n=10)
        self.toolbox.register("map", self.pool.map)
        hof = tools.ParetoFront()  # 非占优最优集

        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)

        pop, logbook = algorithms.eaMuPlusLambda(pop, self.toolbox, mu, lambda_, cxpb, mutpb, ngen, stats=self.stats,
                                                 halloffame=hof, verbose=True)
        self.pool.close()
        return pop


if __name__ == '__main__':
    import tushare
    data = tushare.get_hist_data('002230')
    data.sort_index(inplace=True)
    opt = IndicatorOptimize(data, 'close', 7)
    mu, lambda_, cxpb, mutpb, ngen = 5, 20, 0.5, 0.5, 10
    opt.init_evolution(mu, lambda_, cxpb, mutpb, ngen)
    opt.plot_indicator_signal('close', ma_cross, *tuple(opt.ma_gen_parameter()))

