# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: bias_optimize.py
@time: 2017/6/25 17:59
"""
import random
import multiprocessing

import numpy as np
from deap import creator, base, tools, algorithms

from strategy.bias import BiasStrategy
from core.ctaBacktesting import OptimizationSetting, BacktestingEngine
from core.ctaBase import FUTURE_1MIN

creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)


def test():
    engine = BacktestingEngine()

    # 设置引擎的回测模式为K线
    engine.setBacktestingMode(engine.BAR_MODE)

    # 设置回测用的数据起始日期
    engine.setStartDate('20150101')
    engine.setEndDate('20160120')
    # 设置产品相关参数
    engine.setSlippage(1)  # 股指1跳
    engine.setRate(1.1 / 10000)  # 万0.3
    engine.setSize(10)  # 股指合约大小
    engine.setPriceTick(1)  # 股指最小价格变动

    # 设置使用的历史数据库
    engine.setDatabase(FUTURE_1MIN, 'RB1601')
    setting = OptimizationSetting()  # 新建一个优化任务设置对象
    setting.setOptimizeTarget('capital')  # 设置优化排序的目标是策略净盈利始11，结束12，步进1

    # 运行单进程优化函数，自动输出结果，耗时：359秒
    engine.runOptimization(BiasStrategy, setting)

    # 多进程优化，耗时：89秒
    # engine.runParallelOptimization(AtrRsiStrategy, setting)


def parameter_generate():
    """

    :param period: Bias周期，为二元列表，例如：[4，524]
    :param bias: Bias区间，为二元列表，例如：[-0.5，0.6]
    :param stop_profit_time: 止盈出场时间，例如[200，500]s
    :return: 某个外汇品种的策略参数
    """
    parameter_list = []
    p1 = random.randint(24, 524)
    p2 = random.uniform(-1.0, 1.0)
    p3 = random.uniform(-1.0, 1.0)
    p4 = random.uniform(-1.0, 1.0)
    p5 = random.uniform(-1.0, 1.0)
    # p6 = random.randint(stop_profit_time[0], stop_profit_time[1])

    parameter_list.append(p1)
    parameter_list.append(p2)
    parameter_list.append(p3)
    parameter_list.append(p4)
    parameter_list.append(p5)
    # parameter_list.append(p6)

    return parameter_list


def object_func(strategy_avg):
    """
    本函数为优化目标函数，返回优化参考值
    :param test_data: 回测数据
    :param strategy_avg: 回测参数
    :return: capital，资金净值；profit_factor,盈亏比
    """
    engine = BacktestingEngine()

    # 设置引擎的回测模式为K线
    engine.setBacktestingMode(engine.BAR_MODE)

    # 设置回测用的数据起始日期
    engine.setStartDate('20150101')
    engine.setEndDate('20160120')
    # 设置产品相关参数
    engine.setSlippage(1)  # 股指1跳
    engine.setRate(1.1 / 10000)  # 万0.3
    engine.setSize(10)  # 股指合约大小
    engine.setPriceTick(1)  # 股指最小价格变动

    # 设置使用的历史数据库
    engine.setDatabase(FUTURE_1MIN, 'RB1601')

    setting = {'bias_period': strategy_avg[0], 'bias_buy': strategy_avg[1], 'bias_short': strategy_avg[2],
               'bias_sell': strategy_avg[3], 'bias_cover': strategy_avg[4]}
    engine.initStrategy(BiasStrategy, setting)

    # 开始跑回测
    engine.runBacktesting()
    backresult = engine.calculateBacktestingResult()
    capital = backresult['capital']
    profit_loss_ratio = backresult['profitLossRatio']
    order_num = backresult['totalResult']
    return capital, profit_loss_ratio, order_num


def mut_flip_bit(individual, indpb):
    """
    变异
    :param individual: 个体，实际为策略参数
    :param indpb: 变异概率
    :return: 变异后的个体
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            if i % 6 == 0:
                individual[i] = random.randint(24, 524)
            # elif i % 6 == 5:
            #     individual[i] = random.randint(300, 1300)
            else:
                individual[i] = random.uniform(-1.0, 1.0)

    return individual,


def optimize():
    """
    优化函数，返回优化后保存的参数结果
    :param data: 优化数据集
    :return: 优化后保存的参数结果
    """

    toolbox = base.Toolbox()
    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     parameter_generate)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", object_func)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mut_flip_bit, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)
    random.seed(64)
    # pool = mp.ProcessingPool(config.processes)
    pool = multiprocessing.Pool(processes=7)
    toolbox.register("map", pool.map)
    MU = 20  # 每一代选择的个体数
    LAMBDA = 100  # 每一代产生的子女数
    pop = toolbox.population(80)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 300  # 分别为交叉概率、变异概率、产生种群代数
    hof = tools.ParetoFront()  # 非占优最优集
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof)
    pool.close()
    return pop

if __name__ == '__main__':
    a = optimize()
    print(a)