# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: bias.py.py
@time: 2017/6/23 11:34
"""
from collections import deque

import numpy as np

from core.ctaBase import *
from core.ctaTemplate import CtaTemplate


class BiasStrategy(CtaTemplate):
    """结合ATR和RSI指标的一个分钟线交易策略"""
    className = 'BiasStrategy'
    author = u'Daniel'

    # 策略参数
    bias_period = 24
    bias_buy = -0.35
    bias_short = 0.50
    bias_sell = 0.19
    bias_cover = -0.03
    is_check_outTime = True
    outTime = 200
    limit_profit = 10.0
    fixed_size = 1
    initDays = 0
    # 策略变量
    bar = None  # K线对象
    barMinute = EMPTY_STRING  # K线当前的分钟

    orderList = []  # 保存委托代码的列表

    # 参数列表，保存了参数的名称
    paramList = ['name',
                 'className',
                 'author',
                 'vtSymbol',
                 'bias_period',
                 'bias_bug',
                 'bias_sell',
                 'bias_short',
                 'bias_cover']

    # 变量列表，保存了变量的名称
    varList = ['inited',
               'trading',
               'pos',
               'atrValue',
               'atrMa',
               'rsiValue',
               'rsiBuy',
               'rsiSell']

    # ----------------------------------------------------------------------
    def __init__(self, ctaEngine, setting):
        """Constructor"""
        super(BiasStrategy, self).__init__(ctaEngine, setting)

        buffer_size = self.bias_period
        self.close_buff = deque([np.nan] * buffer_size, maxlen=buffer_size)
        self.ma_buffer = deque([np.nan] * buffer_size, maxlen=buffer_size)
        self.bias_buffer = deque([np.nan] * buffer_size, maxlen=buffer_size)
        self.ITER_COUNT = 0
        # 注意策略类中的可变对象属性（通常是list和dict等），在策略初始化时需要重新创建，
        # 否则会出现多个策略实例之间数据共享的情况，有可能导致潜在的策略逻辑错误风险，
        # 策略类中的这些可变对象属性可以选择不写，全都放在__init__下面，写主要是为了阅读
        # 策略时方便（更多是个编程习惯的选择）

    # ----------------------------------------------------------------------
    def onInit(self):
        """初始化策略（必须由用户继承实现）"""
        self.writeCtaLog(u'%s策略初始化' % self.name)

        # 载入历史数据，并采用回放计算的方式初始化策略数值
        # initData = self.loadBar(self.initDays)
        # for bar in initData:
        #     self.onBar(bar)

        self.putEvent()

    # ----------------------------------------------------------------------
    def onStart(self):
        """启动策略（必须由用户继承实现）"""
        self.writeCtaLog(u'%s策略启动' % self.name)
        self.putEvent()

    # ----------------------------------------------------------------------
    def onStop(self):
        """停止策略（必须由用户继承实现）"""
        self.writeCtaLog(u'%s策略停止' % self.name)
        self.putEvent()

    # ----------------------------------------------------------------------
    def onTick(self, tick):
        """收到行情TICK推送（必须由用户继承实现）"""
        # 计算K线
        tickMinute = tick.datetime.minute

        if tickMinute != self.barMinute:
            if self.bar:
                self.onBar(self.bar)

            bar = CtaBarData()
            bar.vtSymbol = tick.vtSymbol
            bar.symbol = tick.symbol
            bar.exchange = tick.exchange

            bar.open = tick.lastPrice
            bar.high = tick.lastPrice
            bar.low = tick.lastPrice
            bar.close = tick.lastPrice

            bar.date = tick.date
            bar.time = tick.time
            bar.datetime = tick.datetime  # K线的时间设为第一个Tick的时间

            self.bar = bar  # 这种写法为了减少一层访问，加快速度
            self.barMinute = tickMinute  # 更新当前的分钟
        else:  # 否则继续累加新的K线
            bar = self.bar  # 写法同样为了加快速度

            bar.high = max(bar.high, tick.lastPrice)
            bar.low = min(bar.low, tick.lastPrice)
            bar.close = tick.lastPrice

    # ----------------------------------------------------------------------
    def onBar(self, bar):
        """收到Bar推送（必须由用户继承实现）"""
        # 撤销之前发出的尚未成交的委托（包括限价单和停止单）
        for orderID in self.orderList:
            self.cancelOrder(orderID)
        self.orderList = []

        # 计算指标数值
        self.close_buff.append(bar.close)
        # ma和bias指标的计算
        ma = np.mean(list(self.close_buff)[:])
        self.ma_buffer.append(ma)
        self.bias_buffer.append((bar.close - ma) * 100 / ma)

        self.ITER_COUNT += 1
        if self.ITER_COUNT < self.bias_period:
            return

        # 当前无仓位
        if self.pos == 0:

            if self.bias_buffer[-1] <= self.bias_buy:
                    # 这里为了保证成交，选择超价5个整指数点下单
                self.buy(bar.open + 0.5, self.fixed_size)

            elif self.bias_buffer[-1] >= self.bias_short:
                self.short(bar.open - 0.5, self.fixed_size)

        # 持有多头仓位
        elif self.pos > 0:
            ma = self.ma_buffer[-1]
            pre_ma = self.ma_buffer[0]

            # 发出本地止损委托，并且把委托号记录下来，用于后续撤单
            if self.bias_buffer[-1] >= self.bias_sell and (ma - pre_ma) < 0.0:
                self.sell(bar.open, abs(self.pos), stop=False)

        # 持有空头仓位
        elif self.pos < 0:
            ma = self.ma_buffer[-1]
            pre_ma = self.ma_buffer[0]
            if self.bias_buffer[-1] <= self.bias_cover and (ma - pre_ma) > 0.0:
                self.cover(bar.open, abs(self.pos), stop=False)

        # 发出状态更新事件
        self.putEvent()

    # ----------------------------------------------------------------------
    def onOrder(self, order):
        """收到委托变化推送（必须由用户继承实现）"""
        pass

    # ----------------------------------------------------------------------
    def onTrade(self, trade):
        # 发出状态更新事件
        self.putEvent()


if __name__ == '__main__':
    # 提供直接双击回测的功能
    # 导入PyQt4的包是为了保证matplotlib使用PyQt4而不是PySide，防止初始化出错
    from core.ctaBacktesting import *

    # from PyQt4 import QtCore, QtGui

    # 创建回测引擎
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

    # 在引擎中创建策略对象
    # strategy_avg = [73, -0.9086709986332435, 0.566766406031368, -0.6270363358246269, -0.9821013758300181]
    # strategy_avg = [24, -0.5376071711564816, 0.014560780473585888, -0.8786230780066739, -0.9808172242153463]
    # strategy_avg = [24, -0.5376071711564816, 0.014560780473585888, -0.8786230780066739, -0.9808172242153463]
    strategy_avg = [24, -0.5376071711564816, 0.288118583358942, 0.025341378002950554, -0.67]
    setting = {'bias_period': strategy_avg[0], 'bias_buy': strategy_avg[1], 'bias_short': strategy_avg[2],
               'bias_sell': strategy_avg[3], 'bias_cover': strategy_avg[4]}
    engine.initStrategy(BiasStrategy, setting)

    # 开始跑回测
    engine.runBacktesting()

    # 显示回测结果
    engine.showBacktestingResult()

