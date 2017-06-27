# -*- coding: utf-8 -*-
"""使用历史优化参数，组合出最优的参数集合"""
from __future__ import division
import multiprocessing
import os
import config
import pandas as pd
from Function import cal_backtest_result, read_mt4_data, datetime_offset_by_month
from TradeTest import TradeTest, Symbol
from datetime import datetime


def read_args_set(filepath, child_dir_num=11):
    """
    读取策略参数

    Parameters
    ----------
    filepath : 文件夹路径
    child_dir_num : 子文件夹数目
    """
    args_dict = {}
    for month in range(child_dir_num):
        path = os.path.join(filepath, u"Backresult_{}个月".format(month+1), "parameters.csv")
        if os.path.exists(path):
            args_dict["period_{}".format(month+1)] = pd.read_csv(path, usecols=range(1, 10))
        else:
            raise IOError(u'Warning: {}该文件不存在!'.format(path))
    return args_dict


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


class PeriodOrderPool(object):
    """每个周期内备选的订单池"""
    def __init__(self):
        self.__profit_dict = multiprocessing.Manager().dict()
        self.__pool_dict = multiprocessing.Manager().dict()

    def put(self, avg_id, _order_list):
        """添加订单列表"""
        self.__pool_dict[avg_id] = _order_list
        self.__profit_dict[avg_id] = sum(order[1] for order in _order_list)

    def get_best_strategy(self):
        """获得最优策略"""
        profit_dict, pool_dict = dict(self.__profit_dict), dict(self.__pool_dict)  # 共享的dict需要转化为通常的dict
        best_avg_id = max(profit_dict, key=lambda key: profit_dict[key])
        return best_avg_id, profit_dict[best_avg_id], pool_dict[best_avg_id]


class OptimizePortfolio(object):
    """
    优化参数的动态规划问题：先选择一段样本期内的优化参数作为备选参数池，随后开始按照不同的period周期进行迭代择优。假设
    迭代的周期是1个月，且样本区间为2011年1月20日到2012年1月20号，具体操作步骤如下：
    1）共有12个最优参数初始存入可选参数池中；
    2）从2012年1月20日开始将参数池中的参数全部应用于2012年1月20日到2012年2月20日的回测中；
    3）选择其中表现最好的结果存入实盘账户中；
    4）并将总体参数集中当月的最优参数加入到可选参数池中；
    5）随后迭代到下个月，重复2），3），4），5）的操作
    """
    def __init__(self, args_set, start_date, end_date, real_begin_date="2012.01.20"):
        self.__args = args_set                  # 所有参数集合
        self.__current_args = None              # 当前周期所有的最优集合
        self.__pool = []                        # 备选的参数集合
        self.__real_order_list = []             # 实盘订单列表
        self.__start_date = start_date          # 测试数据开始时间，包含样本期和实测期
        self.__end_date = end_date              # 测试数据结束时间
        self.__order_pool = PeriodOrderPool()   # 每个周期内备选的订单池
        self.data_set = None                    # 数据集
        self.sample_month = 12                  # 样本期间的月份数
        self.real_begin_date = datetime.strptime(real_begin_date, "%Y.%m.%d")   # 实测期开始日期

    def prepare_pool(self, period, start_position):
        """
        初始时的可选参数池

        Parameters
        ----------
        period : int, 采用的周期,最大为11
        start_position : int, 实测期间开始的位置
        """
        assert 0 < period < 12, u"采用的周期区间为（0,12）"
        dict_key = "period_{}".format(period)
        self.__current_args = self.__args[dict_key].values         # 获得当前周期参数的值，不使用pandas，处理速度更快
        self.__pool.extend([self.parameter_match(avg) for avg in self.__current_args[:start_position]])

    def pool_add_avg(self, iter_time):
        """周期结束时，向策略备选池中添加已保存的最优参数

        Parameters
        ----------
        iter_time : 迭代的次数，从0开始的整数
        """
        offset_id = len(self.__pool) + iter_time
        try:
            self.__pool.append(self.parameter_match(self.__current_args[offset_id]))
        except IndexError:
            print ("Please check the iter_time is correct!")

    @staticmethod
    def parameter_match(avg):
        """将参数匹配成TradeTest参数输入形式 parameter=[{}],可以根据需求重载"""
        _parameter = [{'symbol': Symbol(avg[6], config.xauusd_base_info_low[1]),
                         'strategy_id': 20170316, 'bias_period': int(avg[0]),
                         'bias_buy': avg[1], 'bias_short': avg[2],
                         'bias_sell': avg[3], 'bias_cover': avg[4],  'is_check_outTime': True,
                         'outTime': avg[5], 'limit_profit': config.stop_profit}]

        return _parameter

    def _period_on_test(self, _avg_id, _parameters, _init_data):
        """运行策略"""
        test = TradeTest(_parameters)
        test.init_data = _init_data
        self.__order_pool.put(_avg_id, test.on_test())

    def get_period_date(self, period, iter_time):
        """计算从实测开始的第(iter_time+1)个周期的左右时间区间,目前只能实现按月变动"""
        _start = datetime_offset_by_month(self.real_begin_date,  period * iter_time)
        _end = datetime_offset_by_month(self.real_begin_date, period * (iter_time+1))
        return str(_start).replace("-", "."), str(_end).replace("-", ".")

    def period_back_test(self, period):
        """周期内的历史回测"""
        total_month = num_month_get(self.__start_date, self.__end_date)              # 样本期和实测期间的总月数
        iter_num = int((total_month - self.sample_month - 1) / period)                      # 实测期间的总迭代次数

        self.prepare_pool(period, int(self.sample_month/period))

        for i in range(iter_num):
            start, end = self.get_period_date(period, i)
            test_data = [_data[(_data.date >= start) & (_data.date <= end)] for _data in self.data_set]
            """
            采用多进程计算备选参数集中的所有参数，其中_period_on_test（）为主执行函数，并将结果保存在以参数序号为key的
            订单池（dict）中,每个order_list是一个二维列表，每一行表示一个订单信息流。
            """
            process_list = []
            for avg_id, parameters in enumerate(self.__pool):
                p = multiprocessing.Process(target=self._period_on_test, args=(avg_id, parameters, test_data))
                process_list.append(p)
                p.daemon = True
                p.start()

            [p.join() for p in process_list]
            # 将当前周期最优的参数加入到真实账户中，其中的元素为tuple,包含最优策略的id, 期间内总盈利， 期间内的订单
            self.__real_order_list.append(self.__order_pool.get_best_strategy())
            # 将总体参数集中当月的最优参数加入到可选参数池中
            self.pool_add_avg(i)
            print(self.__real_order_list[-1][1])

        return self.__real_order_list


if __name__ == '__main__':
    args_sets = read_args_set(u"F://共享文件//高学义//Backresult//xauusd")
    data_start, data_end = "2011.01.20", "2017.01.20"
    init_data = read_mt4_data(["..//Data//TestData//XAUUSD1.csv"], start=data_start, end=data_end)
    optimizer = OptimizePortfolio(args_sets, data_start, data_end)
    optimizer.data_set = init_data
    real_order_list = optimizer.period_back_test(1)
    order_list = []
    for r in real_order_list:
        order_list.extend(r[2])
    cal_backtest_result(order_list, "best_XAUUSD", is_show=True)
