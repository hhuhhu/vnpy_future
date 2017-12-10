# coding=utf-8
"""用于各类指标的分析和统计"""
from pandas.core.groupby import DataFrameGroupBy
from pandas.core.api import DataFrame, Series
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pylab import mpl
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import os


plt.style.use("ggplot")
mpl.rcParams['font.sans-serif'] = ['SimHei']  # matplotlib中文指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['figure.autolayout'] = True


class IndicatorAnalyst(object):
    """
    指标的统计分析类：
    1）原始数据情况分析
    2）添加不同指标后，原始数据被分成了长度为window的数据集，一种是不重复地分组，另一组是移动分组
    3）对组内的数据进行描述统计分析和整个品种的描述统计分析
    """
    def __init__(self, data_set, indicator=None):
        """
        Args:
            data_set: dict(symbol=DataFrame)或DataFrame, 待分析的数据集是一个以品种名为key,value是DataFrame或者是一个DataFrame
            indicator: Series，指标序列，默认是一个空的Series，可以通过直接设置indicator属性设置，或者在类内编写指标获得
        Notes:
            数据集的长度应当与indicator长度相同，否则会报错
        """
        self.__identify = None       # 识别标签函数对象，目前主要是_group_identify 和 _rolling_identify
        self.__indicator = None      # 当前处理的指标对象
        self.__data = None           # 当前处理的数据集对象
        self.__group = None          # 当前处理的分组对象
        self.__symbol = None         # 当前品种对象
        self.__profit = None         # 当前品种的盈亏序列
        self._data_set = data_set.copy()     # 总体数据集
        self._indicator = Series() if indicator is None else indicator
        self._ind_len = 0             # 当前处理的指标数据行数
        self._group = None
        self._profit_func = {FOREX_TYPE: self._forex_profit,
                             STOCK_TYPE: self._stock_profit,
                             FUTURE_TYPE: self._future_profit}

    @property
    def data_set(self):
        return self._data_set

    @property
    def group(self):
        """
        按照条件后分组的对象集合,若输入的数据集是dict，则返回dict,若是DataFrame则返回DataFrame
        """
        return self._group

    @property
    def indicator(self):
        """指标序列"""
        return self._indicator

    @indicator.setter
    def indicator(self, ind):
        """设置指标序列"""
        self._indicator = ind

    def interval_analyst(self, condition, symbol, window=200, rolling=False, profit_mode=True, direction=1,
                         group_plot=False, applied_price="open", fig_save_path=None, ):
        """
         分析指标满足条件下，在之后的窗口内价格的统计信息
        Args:
            condition: [func]返回, True或False的函数对象；
            symbol:[dict or Symbol], 统计的品种对象；
            window:[int, -1, default 200] 观察窗口的大小，默认是200个bar,当rolling为True时，window取-1表示将满足条件开始的
                点直到最后一个数据归为一组；当rolling为False,window取-1时，表示每一次满足条件区间内的数据分为一组，这种情况下
                每一组的长度不相等。
            rolling: [True, False],窗口是采用滚动模式还是截断分组，默认是每组数据重叠的截断分组；
            profit_mode: [True, False],计算盈利模式
            direction: [1, -1],计算盈利时多空的方向
            group_plot: [bool, default False], 绘制每一组数据的价格，当组数很大时将会绘制的很密集
            applied_price: ["open", "low", "high", "close", default "open"],分析采用的价格
            fig_save_path: [list, str, path] ,保存图片的路径，默认存储在
        Returns:

        """
        if isinstance(self._data_set, dict):
            self._group = {}
            for key in self._data_set:
                self.__data = self._data_set[key]
                self.__indicator = self._indicator[key]
                print(u"{}的{}指标描述性统计:".format(key, self.__indicator.name))
                print(self.__indicator.describe())
                self.__group = self._interval_analyst(condition, window, rolling)
                self.__symbol = symbol[key]
                self.group_analyst(profit_mode,
                                   direction=direction,
                                   fig_save_path=self.check_fig_path(fig_save_path, key),
                                   group_plot=group_plot,
                                   applied_price=applied_price)
                self._group[key] = self.__group

        elif isinstance(self._data_set, DataFrame):
            self.__data = self._data_set
            key = self.__data["symbol"].iat[0]
            self.__indicator = self._indicator
            print(u"{}的{}指标描述性统计:".format(self.__data.iat[0, 5], self.__indicator.name))
            print(self.__indicator.describe())
            self.__group = self._interval_analyst(condition, window, rolling)
            self.__symbol = symbol
            self.group_analyst(profit_mode,
                               direction=direction,
                               fig_save_path=self.check_fig_path(fig_save_path, key),
                               group_plot=group_plot,
                               applied_price=applied_price)
            self._group = self.__group

    def _interval_analyst(self, condition, window, rolling):
        """
        分析指标满足条件下，在之后的窗口内价格的统计信息
        Args:
            condition: 返回True或False的函数对象
            window: 观察窗口的大小，默认是200个bar
            rolling: 窗口是采用滚动模式还是截断分组，默认是每组数据重叠的截断分组
        """
        if rolling:
            self.__identify = self._roll_identify
        else:
            self.__identify = self._group_identify

        if isinstance(self.__indicator, Series):
            self._ind_len = len(self.__indicator)
            assert self._ind_len == len(self.__data), u"指标的长度应当与数据集长度相同"
        else:
            raise ValueError(u"指标类型输入错误！")
        return self.__identify(condition, window=window)

    def _roll_identify(self, condition, window):
        """将满足条件的行及随后的window个数据识别成一类，并将其下标存储在groups中"""
        groups = {}
        count = 0
        on_state = False
        for i, ind in enumerate(self.__indicator):
            if on_state:
                if condition(ind):
                    continue
                else:
                    on_state = False

            if condition(ind):
                count += 1
                on_state = True
                # 当窗口为无限长，既到数据末尾
                if window == -1:
                    groups[count] = np.arange(i, self._ind_len)
                else:
                    if i + window < self._ind_len:
                        groups[count] = np.arange(i, i + window)
                    else:
                        groups[count] = np.arange(i, self._ind_len)
        return groups

    def _group_identify(self, condition, window, less_drop_num=10):
        """
        识别满足指标条件的行，并按照1到n的标志分组,原始数据添加一列name为指标name的标志数据,当window=-1时，为取满足
        区间内的数据分为一组，且组内数据小于less_drop_num的剔除
        Args:
            condition: 指标的条件
            window: 窗口的大小
            less_drop_num: 窗口内数据太小需要剔除的临界值

        Returns:

        """
        flag_list = [np.nan] * self._ind_len
        flag = 0
        count = 0
        last_position = 0
        on_state = False
        if isinstance(self.__data, DataFrame):
            for i, _ind in enumerate(self.__indicator.values):
                if on_state:

                    # 当窗口为符合条件的区间时
                    if window == -1:
                        if not condition(_ind):
                            # 当这组数据小于less_drop_num，不考虑这个样本
                            if i - last_position < less_drop_num:
                                flag_list[last_position:i] = [np.nan] * (i - last_position)
                                flag -= 1
                            on_state = False
                        else:
                            flag_list[i] = flag
                    else:
                        if count < window:
                            count += 1
                        else:
                            count = 0
                            on_state = False
                    continue

                if condition(_ind):
                    on_state = True
                    flag += 1
                    # 当窗口为符合条件的区间时
                    if window == -1:
                        last_position = i
                        flag_list[i] = flag
                    else:
                        if (i + window) < self._ind_len:
                            flag_list[i:(i + window)] = [flag] * window

        else:
            raise ValueError("数据集的结构必须是DataFrame")
        self.__data.loc[:, self.__indicator.name] = self.__indicator
        self.__data.loc[:, "group_flag"] = flag_list
        return self.__data.groupby("group_flag", as_index=False)

    def group_analyst(self, profit_mode, direction=1, fig_save_path=None, group_plot=False, applied_price="open"):
        """
        数据分组分析，默认分析的是开盘价
        Args:
            profit_mode: [bool， default False],分析盈亏或价格
            direction: [1, -1],计算盈利时多空的方向
            fig_save_path:[list, str, path] ,保存图片的路径
            group_plot: [bool, default False], 绘制每一组数据的价格，当组数很大时将会绘制的很密集
            applied_price: ["open", "low", "high", "close", default "open"],分析采用的价格
        """
        # 开始分析绘制图表
        print(u"划分的区间数为{}".format(len(self.__group)))
        group_analyst = None
        fig, axe = plt.subplots(2, 2)

        g_fig, g_axe = None, None
        if group_plot:
            g_fig, g_axe = plt.subplots()

        fig3 = None
        if isinstance(self.__group, DataFrameGroupBy):
            group_analyst = self._frame_group_analyst(profit_mode, direction, group_plot, applied_price, g_axe)
            fig3, fig4 = self.group_density()
        elif isinstance(self.__group, dict):
            group_analyst = self._dict_group_analyst(profit_mode, direction, group_plot, applied_price, g_axe)
            fig3, fig4 = self.group_density()
        group_analyst["max"].plot.hist(ax=axe[(0, 0)], title=u"最大值分布", bins=60, legend=False)
        group_analyst["min"].plot.hist(ax=axe[(0, 1)], title=u"最小值分布", bins=60, legend=False)
        group_analyst["mean"].plot.hist(ax=axe[(1, 0)], title=u"平均值分布", bins=60, legend=False)
        group_analyst["std"].plot.hist(ax=axe[(1, 1)], title=u"标准差分布", bins=60, legend=False)

        if profit_mode:
            fig1, axe1 = plt.subplots(3, 2)
            group_analyst["max"].cumsum().plot(ax=axe1[0, 0], title=u"潜在的最大盈利变动", legend=False)
            group_analyst["max"].plot(ax=axe1[0, 1], title=u"每个样本的最大盈利", legend=False)
            group_analyst["min"].cumsum().plot(ax=axe1[1, 0], title=u"潜在的最大亏损变动", legend=False)
            group_analyst["min"].plot(ax=axe1[1, 1], title=u"每个样本的最大亏损", legend=False)
            group_analyst["mean"].cumsum().plot(ax=axe1[2, 0], title=u"潜在的平均盈亏变动", legend=False)
            group_analyst["mean"].plot(ax=axe1[2, 1], title=u"每个样本的平均盈亏", legend=False)
            fig1.savefig(os.path.join(fig_save_path, u"潜在盈亏分析图.png"))

        fig2, axe2 = plt.subplots(2)
        print(u"达到最大值的所需分钟数的描述统计")
        print(group_analyst["max_arg"].describe())
        group_analyst["max_arg"].plot.hist(ax=axe2[0], title=u"达到最大值的所需时间（minute）的分布",
                                           bins=60, legend=False)
        print(u"达到最小值的所需分钟数的描述统计")
        print(group_analyst["min_arg"].describe())
        group_analyst["min_arg"].plot.hist(ax=axe2[1], title=u"达到最小值的所需时间（minute）的分布",
                                           bins=60, legend=False)

        self.save_figure(fig_obj=[fig, fig2, fig3, fig4], save_path=fig_save_path, fig_name=[u"每一组数据的统计分布.png",
                                                                                             u"达到极值所需时间分布.png",
                                                                                             u"概率分布随时间的演化.png",
                                                                                             u"统计特征随时间的演化"])
        if g_fig is not None:
            g_fig.savefig(os.path.join(fig_save_path, u"窗口盈亏变动图.png"))
        plt.show()

    def _group_apply_func(self, x, _direction=1, my_func=None, arg_func=None, _profit_mode=True, apply_price="open",
                          in_position=1, symbol=None):
        """
        DataFrameGroupBy的具体的apply函数
        Args:
            x: [Series]，每一组数据
            _direction: [1， -1]，方向
            my_func: [func],Series自带一些统计函数
            arg_func: [func],numpy中的函数
            _profit_mode: [bool, default True],选择分析盈亏还是价格
            apply_price: ["open", "low", "high", "close", default "open"],分析采用的价格
            in_position: [int],计算盈亏时，进场点的位置
            symbol: [Symbol], 品种对象
        Returns:
           返回一个Series
        """
        assert _direction in (1, -1), u"direction只能取1和-1"
        assert len(x) > in_position, u"每组的长度不能为{}".format(in_position)
        # 分析的数据的选择
        if _profit_mode:
            group_data = self._profit_func[symbol.symbol_type](x, symbol, _direction, apply_price, in_position)
        else:
            group_data = x[apply_price]

        if arg_func is None and my_func is not None:
            return Series(my_func(group_data))
        elif arg_func is not None:
            return Series((arg_func(group_data) - x[apply_price].index[in_position]).seconds / 60)
        else:
            return group_data

    @staticmethod
    def _forex_profit(x, symbol, _direction, apply_price, in_position):
        """外汇盈亏的计算"""
        if symbol.exchange_kind is USD_CURRENCY:
            return symbol.size_value * (_direction * (x[apply_price] - x[apply_price][in_position]) - symbol.slippage)
        elif symbol.exchange_kind is NO_USD_CURRENCY:
            return symbol.size_value * (_direction * (x[apply_price] - x[apply_price][in_position]) - symbol.slippage
                                       ) / x[apply_price]
        else:
            # 暂时不支持交叉货币的计算
            return None

    @staticmethod
    def _future_profit(x, symbol, _direction, apply_price, in_position):
        """期货盈亏计算"""
        open_cost, close_cost = 0.0, []
        if symbol.open_cost_rate != 0.0:
            open_cost = symbol.open_cost_rate * x[apply_price][in_position] * symbol.size_value
        if symbol.close_cost_rate != 0.0:
            close_cost = [symbol.close_cost_rate * price * symbol.size_value for price in x[apply_price]]
        return symbol.size_value * (_direction * (x[apply_price] - x[apply_price][in_position]) - symbol.slippage
                                   ) - close_cost - open_cost

    @staticmethod
    def _stock_profit(x, symbol, _direction, apply_price, in_position):
        close_cost = []
        if symbol.close_cost_rate != 0.0:
            close_cost = [symbol.close_cost_rate * price * symbol.size_value for price in x[apply_price]]
        return symbol.size_value * (_direction * (x[apply_price] - x[apply_price][in_position]) - symbol.slippage
                                   ) - close_cost

    def _frame_group_analyst(self, _profit_mode,  _direction, _group_plot, _applied_price, _axe):
        """使用DataFrameGroupBy类的分组分析"""
        if _profit_mode:
            max_plot = self.__group.apply(self._group_apply_func, symbol=self.__symbol, my_func=np.max,
                                          _direction=_direction, apply_price=_applied_price)
            max_arg = self.__group.apply(self._group_apply_func, symbol=self.__symbol, arg_func=np.argmax,
                                         _direction=_direction, apply_price=_applied_price)
            min_plot = self.__group.apply(self._group_apply_func, symbol=self.__symbol, my_func=np.min,
                                          _direction=_direction, apply_price=_applied_price)
            min_arg = self.__group.apply(self._group_apply_func, symbol=self.__symbol, arg_func=np.argmin,
                                         _direction=_direction, apply_price=_applied_price)
            mean_plot = self.__group.apply(self._group_apply_func, symbol=self.__symbol, my_func=np.mean,
                                           _direction=_direction, apply_price=_applied_price)
            std_plot = self.__group.apply(self._group_apply_func, symbol=self.__symbol, my_func=np.std,
                                          _direction=_direction, apply_price=_applied_price)

            # 每组数据绘制图片
            if _group_plot:
                self.__profit = self.__group.apply(self._group_apply_func, symbol=self.__symbol, _direction=_direction,
                                                   apply_price=_applied_price)
                for g in self.__profit.index.levels[0]:
                    _axe.plot(self.__profit[g].values)
        else:
            max_plot = self.__group.max()[_applied_price]
            max_arg = self.__group.apply(self._group_apply_func, arg_func=np.argmax, _profit_mode=False,
                                         _direction=_direction, apply_price=_applied_price)
            min_plot = self.__group.min()[_applied_price]
            min_arg = self.__group.apply(self._group_apply_func, arg_func=np.argmin, _profit_mode=False,
                                         _direction=_direction, apply_price=_applied_price)
            mean_plot = self.__group.mean()[_applied_price]
            std_plot = self.__group.std()[_applied_price]

        return pd.concat([max_plot, max_arg, min_plot, min_arg, mean_plot, std_plot],
                         axis=1,
                         keys=["max", "max_arg", "min", "min_arg", "mean", "std"])

    def _dict_group_analyst(self, _profit_mode, _direction, _group_plot, _applied_price, _axe):
        """字典形式的分组分析"""
        max_list, max_arg_list, min_list, min_arg_list, mean_list, std_list = [], [], [], [], [], []
        index_map = {"open": 0, "high": 1, "low": 2, "close": 3}
        index = index_map[_applied_price]
        in_position = 1
        profit_list = []
        top_index = []
        bottom_index = []
        for key in self.__group:
            data_ = self.__data.iloc[self.__group[key], index]
            if _profit_mode:
                profit = self._profit_func[self.__symbol.symbol_type](data_, self.__symbol, _direction, _applied_price,
                                                                      in_position)
                profit_list.extend(profit.values)
                top_index.extend([key] * len(profit.index))
                bottom_index.extend(profit.index.values)
                max_list.append(profit.max())
                max_arg_list.append((profit.argmax() - profit.index[in_position]).total_seconds() / 60)
                min_list.append(profit.min())
                min_arg_list.append((profit.argmin() - profit.index[in_position]).total_seconds() / 60)
                mean_list.append(profit.mean())
                std_list.append(profit.std())
                if _group_plot:
                    _axe.plot(profit.values)
            else:
                max_list.append(data_.max())
                max_arg_list.append((data_.argmax() - data_.index[in_position]).total_seconds() / 60)
                min_list.append(data_.min())
                min_arg_list.append((data_.argmax() - data_.index[in_position]).total_seconds() / 60)
                mean_list.append(data_.mean())
                std_list.append(data_.std())
        index = pd.MultiIndex.from_arrays([top_index, bottom_index], names=[None, 'date'])
        self.__profit = Series(profit_list, index=index)
        return DataFrame({"max": max_list, "max_arg": max_arg_list, "min": min_list, "min_arg": min_arg_list,
                          "mean": mean_list, "std": std_list})

    def group_density(self, bin_num=40, window=200, plot_surface=True):
        """绘制每组数据的概率密度随时间的变化图"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fig_1, ax_1 = plt.subplots(3)
        max_profit = self.__profit.max()
        min_profit = self.__profit.min()
        g_profit = self.__profit.groupby(level=0)
        max_list, min_list, mean_list = [], [], []
        xs = []
        ys = []
        zs = []
        for i in range(window):
            max_list.append(g_profit.nth(i).max())
            min_list.append(g_profit.nth(i).min())
            mean_list.append(g_profit.nth(i).mean())
            hist, bins = np.histogram(g_profit.nth(i).values, bins=np.linspace(min_profit, max_profit, bin_num),
                                      density=True)
            xs.append(bins[:-1])
            ys.append(i * np.ones(bin_num - 1))
            zs.append(hist * np.diff(bins))
            if not plot_surface:
                ax.plot(xs[-1], ys[-1], zs=zs[-1])

        ax_1[0].plot(max_list)
        ax_1[0].set_title(u"最大值随时间的演化")
        ax_1[1].plot(min_list)
        ax_1[1].set_title(u"最小值随时间的演化")
        ax_1[2].plot(mean_list)
        ax_1[2].set_title(u"平均值随时间的演化")

        if plot_surface:
            surf = ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.set_zlim(0, 1)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('% .02f'))
            fig.colorbar(surf, shrink=0.5, aspect=5)
        return fig, fig_1

    def save_group_data(self, file_path, file_patch):
        """保存分组数据"""
        for k in self._data_set:
            self._data_set[k].to_csv(os.path.join(file_path, k.lower() + file_patch))

    @staticmethod
    def save_figure(fig_obj, save_path, fig_name):
        """保存图片"""
        for f, n in zip(fig_obj, fig_name):
            f.savefig(os.path.join(save_path, n))

    @staticmethod
    def check_fig_path(path, dir_name):
        """检查存储路径是否合法"""
        if path is None:
            path = os.path.join(os.getcwd(), "analyst_result", dir_name)
        else:
            path = os.path.join(path, "analyst_result", dir_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path


def read_forex_data():
    # 读取数据并分析数据
    reader = ReadForexData()
    reader.read_csv(["../Data/TestData/XAUUSD1.csv"], start='2014.01.01', end='2016.01.01', skiprows=2000000,
                    symbol_name=["XAUUSD"])
    # 近三个月的实盘数据
    # reader.read_csv(["../Data/TestData/R_XAUUSD1.csv", "../Data/TestData/R_USDJPY1.csv"])
    return reader.data


def read_future_data():
    reader = ReadFutureData()
    reader.read_csv(["../Data/TestData/i1601-DCE-Tick.csv", "../Data/TestData/rb1601-SHF-Tick.csv"],
                    symbol_name=["i1601", "rb1601"])
    return reader.data


def read_stock_data():
    reader = ReadStockData()
    reader.read_csv(["../Data/TestData/600597.SH.csv", "../Data/TestData/600605.SH.csv"],
                    symbol_name=['600597', '600605'])
    return reader.data


def analyst_data(data):
    # 分析数据质量
    data_manager = DataManager()
    data_manager.data_analyst(data)

if __name__ == '__main__':
    # data = read_future_data()
    my_data = read_stock_data()
    # 计算指标
    analyst = IndicatorAnalyst(my_data)
    # analyst.indicator = Indicator().mt4_rsi(data, 14, 'close')
    analyst.indicator = Indicator(data_set=my_data, applied_price='close').i_bias(24)
    # 分组统计分析
    # analyst.interval_analyst(lambda x: True if x < 0.35 else False, symbol={'XAUUSD': Symbol("XAUUSD", 0.55)},
    #                          group_plot=True)
    # analyst.interval_analyst(lambda x: True if x < 0.35 else False,
    #                          symbol={"i1601": Symbol("i1601", 0.5, size_value=100, symbol_type=FUTURE_TYPE,
    #                                                  open_cost_rate=0.0001, close_cost_rate=0.0001),
    #                                  "rb1601": Symbol("i1601", 0.5, size_value=10, symbol_type=FUTURE_TYPE,
    #                                                   open_cost_rate=0.0001, close_cost_rate=0.0001)},
    #                          profit_mode=True,
    #                          group_plot=True)
    analyst.interval_analyst(lambda x: True if x < 0.35 else False,
                             symbol={"600597": Symbol("600597", 0.0, size_value=100, symbol_type=STOCK_TYPE,
                                                      close_cost_rate=0.0012),
                                     '600605': Symbol('600605', 0.0, size_value=100, symbol_type=STOCK_TYPE,
                                                      close_cost_rate=0.0012)},
                             profit_mode=True,
                             group_plot=True)
    # analyst.save_group_data(u"E:/EA研究/技术指标系统研究/指标统计分析/repeat_group/", "_repeated_group_rsi_30.csv")
