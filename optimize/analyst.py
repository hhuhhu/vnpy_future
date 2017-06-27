# coding=utf-8
"""分析参数结果"""
import pandas as pd
import os
import matplotlib.pylab as plt
from pylab import mpl

plt.style.use("ggplot")
mpl.rcParams['font.sans-serif'] = ['SimHei']  # matplotlib中文指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['figure.autolayout'] = True
# 参数类型，详见judge_avg函数的doc
LONG_TREND_SHORT_TREND = 0
LONG_INVERSE_SHORT_INVERSE = 1
LONG_INVERSE_SHORT_TREND = 2
LONG_TREND_SHORT_INVERSE = 3
OTHER_HUMAN_IDENTIFY = 4


class Analyst(object):
    """优化结果统计分析者"""
    def __init__(self, args_set_path, best_args_path=None):
        self.__args = self.read_args_set(args_set_path)
        self.__best_args = {}
        self.__order = {}
        self.__total_best_args = []
        self.__class_set = pd.DataFrame()
        self.__default_path = best_args_path
        self.read_best_args()

    @property
    def args(self):
        """全体优化参数集合"""
        return self.__args

    @property
    def best_args(self):
        """最优参数集合"""
        return self.__best_args

    @property
    def class_set(self):
        # return type:pd.DataFrame
        """分类后的所有周期的参数集合"""
        return self.__class_set

    @property
    def total_best_args(self):
        """分类后的最优参数集合"""
        return self.__total_best_args

    @staticmethod
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
            path = os.path.join(filepath, u"Backresult_{}个月".format(month + 1), "parameters.csv")
            if os.path.exists(path):
                args_dict["period_{}".format(month + 1)] = pd.read_csv(path, usecols=range(1, 10))
            else:
                raise IOError(u'Warning: {}该文件不存在!'.format(path))
        return args_dict

    @staticmethod
    def judge_avg(args):
        """
        将所有参数分成四类：
        LONG_TREND_SHORT_TREND ：多头趋势，空头趋势
        LONG_INVERSE_SHORT_INVERSE ：多头反转，空头反转
        LONG_INVERSE_SHORT_TREND ：多头反转，空头趋势
        LONG_TREND_SHORT_INVERSE ：多头趋势，空头反转
        """
        # 多头趋势、空头趋势
        if abs(args.Buy) < abs(args.Sell) and abs(args.Short) < abs(args.Cover):
            return LONG_TREND_SHORT_TREND

        # 多头反转、空头反转
        if abs(args.Buy) >= abs(args.Sell) and abs(args.Short) >= abs(args.Cover):
            return LONG_INVERSE_SHORT_INVERSE

        # 多头反转、空头趋势
        if abs(args.Buy) >= abs(args.Sell) and abs(args.Short) < abs(args.Cover):
            return LONG_INVERSE_SHORT_TREND

        # 多头趋势、空头反转
        if abs(args.Buy) < abs(args.Sell) and abs(args.Short) >= abs(args.Cover):
            return LONG_TREND_SHORT_INVERSE

    @staticmethod
    def judge_avg_0(args):
        """
        将所有参数分成五类：
        LONG_TREND_SHORT_TREND ：多头趋势，空头趋势
        LONG_INVERSE_SHORT_INVERSE ：多头反转，空头反转
        LONG_INVERSE_SHORT_TREND ：多头反转，空头趋势
        LONG_TREND_SHORT_INVERSE ：多头趋势，空头反转
        OTHER_HUMAN_IDENTIFY : 其他的情况
        """
        # 多头趋势、空头趋势
        if args.Sell >= args.Buy > 0 > args.Cover <= args.Short:
            return LONG_TREND_SHORT_TREND

        # 多头反转、空头反转
        if 0 < args.Buy <= args.Sell and args.Short > 0 and args.Short > args.Cover:
            return LONG_INVERSE_SHORT_INVERSE

        # 多头反转、空头趋势
        if args.Buy < 0 and args.Buy <= args.Sell and 0 > args.Short > args.Cover:
            return LONG_INVERSE_SHORT_TREND

        # 多头趋势、空头反转
        if 0 < args.Buy <= args.Sell and args.Short > 0 and args.Short > args.Cover:
            return LONG_TREND_SHORT_INVERSE

        return OTHER_HUMAN_IDENTIFY

    def read_best_args(self, is_read_order=False, child_dir_num=11):
        """读取选出的最优参数"""
        if self.__default_path is None:
            self.__default_path = "..//Data//Backresult//combine_3"

        for month in range(child_dir_num):
            path = os.path.join(self.__default_path, u"{}_month".format(month + 1))
            if is_read_order:
                self.__order[month] = pd.read_csv(os.path.join(path, u'order_list.csv'))
            self.__best_args[month] = pd.read_csv(os.path.join(path, u'best_parameters.csv'))

    def args_classify(self, save_figure=True):
        """
        为整个参数集合分类
        其中前11个子图是每个周期每一类，出现的频数图。最后一个图是所有周期每一类出现的频数图
        """
        class_set = []
        f1, ax1 = plt.subplots(4, 3)
        for month, key in enumerate(self.__args):
            args = self.__args[key]
            position = divmod(month, 3)
            type_list = [self.judge_avg(a_) for a_ in args.itertuples()]
            args.loc[:, "type"] = type_list
            args.groupby("type").size().plot.bar(ax=ax1[position], title=u"周期-{}".format(month + 1))
            ax1[position].set_ylabel(u"频数")
            class_set.append(args)

        self.__class_set = pd.concat(class_set)
        self.__class_set.groupby("type").size().plot.bar(ax=ax1[3, 2], title=u"总体每一类的频数")
        ax1[3, 2].set_ylabel(u"频数")
        if save_figure:
            f1.savefig(os.path.join(self.__default_path, 'all_avg_result.png'))

    def statistics_best_args(self, args_period=1, save_figure=True):
        """统计最优参数"""
        assert len(self.__best_args) == 11, u"默认为共11个周期，否则需要重构该函数"
        f_best_id, ax_id = plt.subplots(4, 3)
        f_best_profit, ax_profit = plt.subplots(4, 3)
        args_list = []
        args_pool = self.__args["period_{}".format(args_period)]

        for month in self.__best_args:
            _args = self.__best_args[month]
            class_list = [self.judge_avg(args_pool.iloc[a.best_avg_id, :]) for a in _args.itertuples()]
            _args.loc[:, "months"] = month
            _args.loc[:, 'type'] = class_list
            position = divmod(month, 3)

            args_by_id = _args.groupby("best_avg_id")
            args_by_id.size().plot.bar(ax=ax_id[position], title=u"周期-{}:最优策略ID频数图".format(month + 1))
            ax_id[position].set_ylabel(u"频数")
            args_by_id.mean()["best_profit"].plot.bar(ax=ax_profit[position],
                                                      title=u"周期-{}:最优策略平均收益频数图".format(month + 1))
            ax_profit[position].set_ylabel(u"频数")
            args_list.append(_args)

        self.__total_best_args = pd.concat(args_list)
        total_group = self.__total_best_args.groupby("best_avg_id")
        total_group.size().plot.bar(ax=ax_id[3, 2], title=u"总体最优策略ID频数图")
        total_group.sum()["best_profit"].plot.bar(ax=ax_profit[3, 2], title=u"最优策略收益频数图")

        f_id_profit, ax_id_profit = plt.subplots(2, 2)
        grouped = self.__total_best_args.groupby("best_avg_id")
        grouped.sum()["best_profit"].plot.bar(ax=ax_id_profit[0, 0], title=u"最优策略对应的总盈利")
        grouped.mean()["best_profit"].plot.bar(ax=ax_id_profit[0, 1], title=u"最优策略对应的平均盈利")

        grouped_by_type = self.__total_best_args.groupby("type")
        grouped_by_type.size().plot.bar(ax=ax_id_profit[1, 0], title=u"每一类策略被选中的频数")
        grouped_by_type.sum()["best_profit"].plot.bar(ax=ax_id_profit[1, 1], title=u"每一类策略的总盈利")

        if save_figure:
            f_best_id.savefig(os.path.join(self.__default_path, 'best_avg_result.png'))
            f_best_profit.savefig(os.path.join(self.__default_path, 'best_profit_avg_result.png'))
            f_id_profit.savefig(os.path.join(self.__default_path, 'total_avg_result.png'))
        plt.show()


if __name__ == '__main__':
    analyst = Analyst(u"F://共享文件//高学义//Backresult//xauusd", best_args_path="..//Data//Backresult//combine_4")
    analyst.args_classify()
    analyst.statistics_best_args()

