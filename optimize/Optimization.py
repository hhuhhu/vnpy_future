# -*- coding: utf-8 -*-
from __future__ import division

import copy
import math
import multiprocessing
import os
import random
from compiler.ast import flatten
from datetime import datetime

import numpy as np
import pandas as pd
from deap import creator, base, tools, algorithms

import Validation
import config
from Function import cal_backtest_result, read_mt4_data, save_backresult
from TradeTest import TradeTest, account_plot
from util import parameter_match, strategy_evaluate


def ahp():
    """
    层次分析法，待实现
    :return:
    """
    return


def paramater_save(symbols, test_period, avg, months):
    """
    用来保存优化后的参数
    :param symbols: 品种列表，type为list，例如：['XAUUSD','GBPUSD']
    :param test_period: 参数优化区间，type为String,例如：‘20160804-20160904’
    :param avg: 保存的参数，类型为list
    :param months: 回测周期，以月为单位,类型为int
    :return:
    """
    symbols_length = len(symbols)
    array_split = int(len(avg)/symbols_length)
    avg = zip(*[iter(avg)]*array_split)
    columns = ['BiasPeriod', 'Buy', 'Sell', 'Short', 'Cover', 'StopProfitTime']
    df = pd.DataFrame(data=avg, columns=columns)
    df['Symbol'] = symbols
    df['TestPeriod'] = test_period
    df['Months'] = months
    return df


def date_handle(date, num_month):
    """
    用于计算测试数据（每次循环）的开始日期
   :param date: 输入日期
    :param num_month: 月数
    :return: 第i次的开始日期
    """
    date_temp = date[0:4] + '-' + date[5:7]
    date_temp = np.datetime64(date_temp) + np.timedelta64(num_month, 'M')
    _start_date = str(date_temp)[0:4] + '.' + str(date_temp)[5:7] + '.' + date[8:10]

    return _start_date


class Data(object):
    """
    构建的一个数据类，为了解决函数多参问题
    """
    def __init__(self):
        self.data = []


def capital(test_data, strategy_avg):
    """
    本函数为优化目标函数，返回优化参考值
    :param test_data: 回测数据
    :param strategy_avg: 回测参数
    :return: capital，资金净值；profit_factor,盈亏比
    """
    gross_loss = 0.0
    gross_profit = 0.0
    profit_factor = 0.0
    symbols = config.test_symbol
    parameter = parameter_match(symbols, strategy_avg)
    test = TradeTest(parameter)
    test.init_data = test_data.data

    _order_list = test.on_test()
    _pnl_list = [t[1] for t in _order_list]
    _capital = test.account.balance
    for pnl in _pnl_list:
        if pnl >= 0:
            gross_profit += pnl
        else:
            gross_loss += pnl
        if gross_loss != 0:
            profit_factor = abs(gross_profit / gross_loss)
    return _capital, profit_factor


def mul_parameter():
    """
    生成多品种遗传算法参数
    :return:遗传算法个体
    """
    symbols = config.test_symbol
    parameters = []
    [parameters.append(copy.deepcopy(single_parameter(symbol))) for symbol in symbols]
    parameters = flatten(parameters)
    return parameters


def single_parameter(symbol):
    """

    :param symbol: 外汇品种：必须为 'USDJPY, GBPUSD, OILUSD, USDCHF, XAUUSD' 其中之一
    :return:遗传算法个体
    """
    assert isinstance(symbol, str), "symbol needs to be a string"
    if symbol.upper() == "XAUUSD":
        parameters = parameter_generate(period=[24, 524], bias=[-0.5, 0.6], stop_profit_time=[300, 1300])
    elif symbol.upper() == "OILUSD":
        parameters = parameter_generate(period=[24, 524], bias=[-2, 2], stop_profit_time=[300, 1300])
    elif symbol.upper() == "GBPUSD":
        parameters = parameter_generate(period=[24, 524], bias=[-0.7, 0.7], stop_profit_time=[300, 1300])
    elif symbol.upper() == "USDCHF":
        parameters = parameter_generate(period=[24, 524], bias=[-0.7, 0.7], stop_profit_time=[300, 1300])
    elif symbol.upper == "USDJPY":
        parameters = parameter_generate(period=[24, 524], bias=[-0.7, 0.7], stop_profit_time=[300, 1300])
    else:
        raise "You got a wrong forex, It needs to be one of the 'USDJPY, GBPUSD, USDOIL, USDCHF, XAUUSD'"

    return parameters


def parameter_generate(period, bias, stop_profit_time):
    """

    :param period: Bias周期，为二元列表，例如：[4，524]
    :param bias: Bias区间，为二元列表，例如：[-0.5，0.6]
    :param stop_profit_time: 止盈出场时间，例如[200，500]s
    :return: 某个外汇品种的策略参数
    """
    parameter_list = []
    assert isinstance(period, list), "period需要是一个元组"
    assert isinstance(bias, list), "bias需要是一个元组"
    assert isinstance(stop_profit_time, list), "stop_profit_time需要是一个元组"
    p1 = random.randint(period[0], period[1])
    p2 = random.uniform(bias[0], bias[1])
    p3 = random.uniform(bias[0], bias[1])
    p4 = random.uniform(bias[0], bias[1])
    p5 = random.uniform(bias[0], bias[1])
    p6 = random.randint(stop_profit_time[0], stop_profit_time[1])

    parameter_list.append(p1)
    parameter_list.append(p2)
    parameter_list.append(p3)
    parameter_list.append(p4)
    parameter_list.append(p5)
    parameter_list.append(p6)

    return parameter_list


def mut_flip_bit(individual, indpb):
    """
    变异
    :param individual: 个体，实际为策略参数
    :param indpb: 变异概率
    :return: 变异后的个体
    """
    for i in xrange(len(individual)):
        if random.random() < indpb:
            if i % 6 == 0:
                individual[i] = random.randint(24, 524)
            elif i % 6 == 5:
                individual[i] = random.randint(300, 1300)
            else:
                individual[i] = random.uniform(-0.5, 0.6)

    return individual,


def optimize(data):
    """
    优化函数，返回优化后保存的参数结果
    :param data: 优化数据集
    :return: 优化后保存的参数结果
    """
    creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     mul_parameter)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", capital, data)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mut_flip_bit, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)
    random.seed(64)
    # pool = mp.ProcessingPool(config.processes)
    pool = multiprocessing.Pool(processes=config.processes)
    toolbox.register("map", pool.map)
    MU = 20  # 每一代选择的个体数
    LAMBDA = 100  # 每一代产生的子女数
    pop = toolbox.population(config.population_num)
    CXPB, MUTPB, NGEN = 0.5, 0.2, config.ngen_num  # 分别为交叉概率、变异概率、产生种群代数
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


def fig_title_generate(symbols):
    """
    生成图例名字
    :param symbols: 外汇品种列表
    :return: 图例名字
    """

    num_symbol = len(symbols)
    fig_title = ''
    for i in xrange(num_symbol):
        if i == 0:
            fig_title = symbols[i]
        else:
            fig_title = fig_title + '+' + symbols[i]
    return fig_title


def test_period_generate(start_date, end_date):
    """
    生成数据测试区间
    :param start_date: 开始日期
    :param end_date: 结束日期
    :return: 数据测试区间
    """
    test_period = start_date.replace('.', '') + '-' + end_date.replace('.', '')
    return test_period


def num_month_get(start_date, end_date):
    """
    计算测试区间的月数
    :param start_date:开始测试日期
    :param end_date:结束测试日期
    :return:测试区间的月数
    """
    start_date = datetime.strptime(start_date, "%Y.%m.%d")
    end_date = datetime.strptime(end_date, "%Y.%m.%d")
    num_month = end_date - start_date
    num_month = int(round(num_month.days/30))
    return num_month


def end_date_compare(end_date, test_end):
    """
    判断两个时间的大小
    :param end_date:
    :param test_end:
    :return:
    """
    end_date = datetime.strptime(end_date, "%Y.%m.%d")
    test_end = datetime.strptime(test_end, "%Y.%m.%d")
    compare_result = (test_end - end_date).seconds
    if compare_result > 0:
        return True
    else:
        return False


def num_iteration_get(total_month, opt_month, val_month, test_month):

    iteration = math.ceil((total_month - opt_month - val_month)/test_month)
    return int(iteration)


def opt_result_save(symbols, parameter, data, dir_path):
    fig_title = fig_title_generate(symbols)
    parameters = parameter_match(symbols, parameter)
    test = TradeTest(parameters)
    test.init_data = data
    order_list = test.on_test()
    # 计算回测结果并保存订单
    backresult = cal_backtest_result(order_list, fig_title, plot_date=False, dir_path=dir_path)
    account_plot(test.account, test.init_data, dir_path=dir_path, plot_show=False)
    save_backresult(backresult, dir_path)


def test_main(symbols, start_date, end_date, opt_month, val_month, test_month):
    fig_title = fig_title_generate(symbols)  # 用于生成图表的标题
    order_list = []  # 存放回测结果
    parameters = pd.DataFrame()
    # 数据读取
    file_dir = "Data//TestData//"
    result_dir = "Data//Backresult//"
    file_paths = [file_dir+forex+'1.csv' for forex in symbols]
    data = read_mt4_data(file_paths, start=start_date, end=end_date)
    total_month = num_month_get(start_date, end_date)  # 总回测月数
    iteration = num_iteration_get(total_month, opt_month, val_month, test_month)
    evaluation_factor = 0
    for i in range(iteration):

        _start = date_handle(start_date, i*opt_month)
        _end = date_handle(_start, opt_month)
        opt_period = test_period_generate(_start, _end)
        my_data = Data()
        my_data.data = [dt.ix[_start:_end] for dt in data]

        # pop 存放优化后的策略参数

        pop = optimize(data=my_data)
        pop = np.array(pop)  # 把pop类型由individual转化为array类型，方便处理
        a = np.array(pd.DataFrame(pop).drop_duplicates())  # 去除pop里边重复的参数
        # 参数验证
        num_avg = len(a)  # 参数个数
        val = Validation.StrategyValidation(symbols)  # 生成参数验证实例
        if val_month == 0:
            pass
        else:
            _start = _end  # 验证开始时间
            _end = date_handle(_start, val_month)  # 验证结束时间
        val.data = [dt.ix[_start:_end] for dt in data] # 验证数据
        val_dict = {}  # 保存策略参数，类型为字典

        # 参数遍历,返回优化后的参数
        for j in xrange(num_avg):
            _avg = a[j]
            immediate = val.capital(_avg)
            val_dict[immediate] = _avg
        immediate_keys = val_dict.keys()
        immediate_keys.sort()
        avg = val_dict[immediate_keys[-1]]
        # 保存最优参数
        opt_path = result_dir + 'opt_result' + "//" + opt_period + "//"
        if os.path.exists(opt_path):
            pass
        else:
            os.makedirs(opt_path)
        opt_result_save(symbols, avg, my_data.data, opt_path)
        parameter = paramater_save(symbols, opt_period, avg, opt_month)
        parameters = parameters.append(copy.deepcopy(parameter))
        _start = _end  # 实测开始时间
        _end = date_handle(_start, test_month)  # 实测结束时间

        if end_date_compare(end_date, _end):
            _end = end_date
        test_data = [dt.ix[_start:_end] for dt in data]
        val.data = test_data
        alt_parameter, alt_factor = val.parameter_exchange()
        if (evaluation_factor < 0 and alt_factor >0):
            avg = alt_parameter
        bt = TradeTest(parameter_match(symbols, avg))
        bt.init_data = test_data
        temp = bt.on_test()
        evaluation_factor = strategy_evaluate(temp)
        test_period = test_period_generate(_start, _end)
        test_path = result_dir + 'test_result' + "//" + test_period + "//"
        if os.path.exists(test_path):
            pass
        else:
            os.makedirs(test_path)
        backresult = cal_backtest_result(temp, fig_title, plot_date=False, dir_path=test_path)
        account_plot(bt.account, bt.init_data, test_path, plot_show=False)
        save_backresult(backresult, test_path)
        if temp:
            order_list.extend(temp)
    parameters.to_csv("Data//Backresult//parameters.csv")

    # 计算回测结果
    backresult = cal_backtest_result(order_list, fig_title, plot_date=False)
    save_backresult(backresult)

if __name__ == "__main__":
    symbols = config.test_symbol
    start_date = '2016.01.04'  # 数据回测起始日期
    end_date = '2016.08.04'  # 数据回测结束日期
    opt_month = 2
    val_month = 0
    test_month = 2
    test_main(symbols, start_date, end_date, opt_month, val_month, test_month)



