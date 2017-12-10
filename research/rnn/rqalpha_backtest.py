# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: rqalpha_backtest.py
@time: 2017/10/13 14:18
"""
import os
import sys

import numpy as np
import pandas as pd
import rqalpha
import rqalpha.api as rqa
from rqalpha import run_func

from research.rnn.rnn import data_frequency_transfer, fix_data, get_factors, Classifier_PonderDNC_BasicLSTM_L3, \
    dense_to_one_hot

tmp = fix_data('白银88.csv')

# targets 1d 数据合成
tmp_1d = data_frequency_transfer(tmp, '1d')
rolling = 88
targets = tmp_1d
targets['returns'] = targets['close'].shift(-2) / targets['close'] - 1.0
targets['upper_boundary'] = targets.returns.rolling(rolling).mean() + 0.5 * targets.returns.rolling(rolling).std()
targets['lower_boundary'] = targets.returns.rolling(rolling).mean() - 0.5 * targets.returns.rolling(rolling).std()
targets.dropna(inplace=True)
targets['labels'] = 1
targets.loc[targets['returns'] >= targets['upper_boundary'], 'labels'] = 2
targets.loc[targets['returns'] <= targets['lower_boundary'], 'labels'] = 0

# factors 1d 数据合成
tmp_1d = data_frequency_transfer(tmp, '1d')
Index = tmp_1d.index
High = tmp_1d.high.values
Low = tmp_1d.low.values
Close = tmp_1d.close.values
Open = tmp_1d.open.values
Volume = tmp_1d.volume.values
factors = get_factors(Index, Open, Close, High, Low, Volume, rolling=26, drop=True)

factors = factors.loc[:targets.index[-1]]

tmp_factors_1 = factors.iloc[:12]
targets = targets.loc[tmp_factors_1.index[-1]:]

gather_list = np.arange(factors.shape[0])[11:]
inputs = np.array(factors).reshape(-1, 1, factors.shape[1])
targets = dense_to_one_hot(targets['labels'])
targets = np.expand_dims(targets, axis=1)
op7 = Classifier_PonderDNC_BasicLSTM_L3(
    inputs=inputs,
    targets=targets,
    gather_list=gather_list,
    hidden_size=50,
    memory_size=50,
    pondering_coefficient=1e-1,
    learning_rate=1e-4)


# op7.fit(training_iters=100,
#         display_step=10,
#         save_path=os.path.join(sys.path[0], "saver/ResidualPonderDNC_4.ckpt"),
#         restore_path=os.path.join(sys.path[0], "saver/ResidualPonderDNC_3.ckpt"))


def init(context):
    context.contract = 'AG88'
    context.BarSpan = 200
    context.TransactionRate = '1d'
    context.DataFields = ['datetime', 'open', 'close', 'high', 'low', 'volume']
    context.DefineQuantity = 5
    context.func_get_factors = get_factors
    context.model_classifier = op7


def handle_bar(context, bar_dict):
    # 合约池代码
    contract = context.contract
    # rqa.logger.info('------------------------------------')
    # timepoint = rqa.history_bars(contract, 1, '1d', 'datetime')[0]
    # timepoint = pd.to_datetime(str(timepoint))
    # timepoint = rqa.get_next_trading_date(timepoint)
    # rqa.logger.info (timepoint)

    # 获取合约报价
    Quotes = rqa.history_bars(
        order_book_id=contract,
        bar_count=context.BarSpan,
        frequency=context.TransactionRate,
        fields=context.DataFields)
    Quotes = pd.DataFrame(Quotes)
    print('Quotes: ', Quotes)

    # 计算技术分析指标
    tmp_factors = context.func_get_factors(
        index=pd.to_datetime(Quotes['datetime']),
        Open=Quotes['open'].values,
        Close=Quotes['close'].values,
        High=Quotes['high'].values,
        Low=Quotes['low'].values,
        Volume=Quotes['volume'].values,
        drop=True)
    inputs = np.expand_dims(np.array(tmp_factors), axis=1)
    print('inputs: ', inputs.shape)
    # 模型预测
    probability, classification = context.model_classifier.pred(inputs, restore_path=os.path.join(sys.path[0],
                                                                                                  "version1.0/ResidualPonderDNC_3.ckpt"))
    flag = classification[-1][0]
    rqa.logger.info(str(flag))
    # print (flag)

    # 绘制估计概率
    rqa.plot("估空概率", probability[-1][0][0])
    rqa.plot("振荡概率", probability[-1][0][1])
    rqa.plot("估多概率", probability[-1][0][2])

    # 获取仓位
    print(context.portfolio)
    cur_position = context.portfolio.future_account.positions

    tmp_buy_quantity = 0
    tmp_sell_quantity = 0
    if cur_position:
        tmp_buy_quantity = cur_position[contract].buy_quantity
        tmp_sell_quantity = cur_position[contract].sell_quantity

    # 沽空
    if flag == 0:
        rqa.logger.info('沽空')

        if tmp_buy_quantity > 0:
            rqa.sell_close(contract, tmp_buy_quantity)
            rqa.sell_open(contract, context.DefineQuantity)
            rqa.logger.info('平多单  开空单')

        elif tmp_sell_quantity > 0:
            rqa.logger.info('持有空头，不调仓')
        else:
            rqa.sell_open(contract, context.DefineQuantity)
            rqa.logger.info('开空单')

    # 沽多
    if flag == 2:
        rqa.logger.info('沽多')
        if tmp_sell_quantity > 0:
            rqa.buy_close(contract, tmp_sell_quantity)
            rqa.buy_open(contract, context.DefineQuantity)
            rqa.logger.info('平空单  开多单')

        elif tmp_buy_quantity > 0:
            rqa.logger.info('持有多头，不调仓')
            pass
        else:
            rqa.logger.info('开多单')
            rqa.buy_open(contract, context.DefineQuantity)

    if flag == 1:
        rqa.logger.info('振荡区间')
        if tmp_sell_quantity > 0:
            rqa.buy_close(contract, tmp_sell_quantity)
            rqa.logger.info('平空单')
        if tmp_buy_quantity > 0:
            rqa.sell_close(contract, tmp_buy_quantity)
            rqa.logger.info('平多单')
        else:
            rqa.logger.info('空仓规避')


start_date = '2017-01-01'
end_date = '2017-09-01'
accounts = {'future': 1e5}

config = {
    'base': {'start_date': start_date, 'end_date': end_date, 'accounts': accounts, 'future_starting_cash': 1000000,
             'stock_starting_cash': 10000, 'securities': ['future']},
    'extra': {'log_level': 'info'},
    'mod': {'sys_analyser': {'enabled': True, 'plot': True}}
}

results = run_func(init=init, handle_bar=handle_bar, config=config)
keys = ['future_account', 'plots', 'trades', 'future_positions', 'portfolio']
for key in keys:
    results['sys_analyser'][key].to_csv("{}.csv".format(key), encoding='utf-8')
print('results: ', results)
