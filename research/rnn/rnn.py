# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: rnn.py
@time: 2017/10/12 9:33
"""
import os
import sys
import copy
import time
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import talib
import tensorflow as tf

from research.rnn.ponder_dnc import DNCore_L3
from research.rnn.ponder_dnc import ResidualACTCore as ACTCore

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 1000)
sns.set_style('whitegrid')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Hide messy TensorFlow warnings


def fix_data(path):
    data = pd.read_csv(path, encoding="gbk", engine='python')
    data.rename(columns={'Unnamed: 0': 'trading_time'}, inplace=True)
    data['trading_point'] = pd.to_datetime(data.trading_time)
    del data['trading_time']
    data.set_index(data.trading_point, inplace=True)
    return data


def data_frequency_transfer(tmp, freq):
    """处理从RiceQuant下载的分钟线数据，
    从分钟线数据合成低频数据
    2017-08-11    
    """
    # 分别处理bar数据
    tmp_open = tmp['open'].resample(freq).ohlc()
    tmp_open = tmp_open['open'].dropna()

    tmp_high = tmp['high'].resample(freq).ohlc()
    tmp_high = tmp_high['high'].dropna()

    tmp_low = tmp['low'].resample(freq).ohlc()
    tmp_low = tmp_low['low'].dropna()

    tmp_close = tmp['close'].resample(freq).ohlc()
    tmp_close = tmp_close['close'].dropna()

    tmp_price = pd.concat([tmp_open, tmp_high, tmp_low, tmp_close], axis=1)

    # 处理成交量
    tmp_volume = tmp['volume'].resample(freq).sum()
    tmp_volume.dropna(inplace=True)

    return pd.concat([tmp_price, tmp_volume], axis=1)


def get_factors(index,
                Open,
                Close,
                High,
                Low,
                Volume,
                rolling=26,
                drop=False,
                normalization=True):
    tmp = pd.DataFrame()
    tmp['tradeTime'] = index

    # 累积/派发线（Accumulation / Distribution Line，该指标将每日的成交量通过价格加权累计，
    # 用以计算成交量的动量。属于趋势型因子
    # tmp['AD'] = talib.AD(High, Low, Close, Volume)

    # 佳庆指标（Chaikin Oscillator），该指标基于AD曲线的指数移动均线而计算得到。属于趋势型因子
    # tmp['ADOSC'] = talib.ADOSC(High, Low, Close, Volume, fastperiod=3, slowperiod=10)

    # 平均动向指数，DMI因子的构成部分。属于趋势型因子
    tmp['ADX'] = talib.ADX(High, Low, Close, timeperiod=14)

    # 相对平均动向指数，DMI因子的构成部分。属于趋势型因子
    tmp['ADXR'] = talib.ADXR(High, Low, Close, timeperiod=14)

    # 绝对价格振荡指数
    tmp['APO'] = talib.APO(Close, fastperiod=12, slowperiod=26)

    # Aroon通过计算自价格达到近期最高值和最低值以来所经过的期间数，帮助投资者预测证券价格从趋势到区域区域或反转的变化，
    # Aroon指标分为Aroon、AroonUp和AroonDown3个具体指标。属于趋势型因子
    tmp['AROONDown'], tmp['AROONUp'] = talib.AROON(High, Low, timeperiod=14)
    tmp['AROONOSC'] = talib.AROONOSC(High, Low, timeperiod=14)

    # 均幅指标（Average TRUE Ranger），取一定时间周期内的股价波动幅度的移动平均值，
    # 是显示市场变化率的指标，主要用于研判买卖时机。属于超买超卖型因子。
    tmp['ATR14'] = talib.ATR(High, Low, Close, timeperiod=14)
    tmp['ATR6'] = talib.ATR(High, Low, Close, timeperiod=6)

    # 布林带
    tmp['Boll_Up'], tmp['Boll_Mid'], tmp['Boll_Down'] = talib.BBANDS(Close, timeperiod=20, nbdevup=2, nbdevdn=2,
                                                                     matype=0)

    # 均势指标
    tmp['BOP'] = talib.BOP(Open, High, Low, Close)

    # 5日顺势指标（Commodity Channel Index），专门测量股价是否已超出常态分布范围。属于超买超卖型因子。
    tmp['CCI5'] = talib.CCI(High, Low, Close, timeperiod=5)
    tmp['CCI10'] = talib.CCI(High, Low, Close, timeperiod=10)
    tmp['CCI20'] = talib.CCI(High, Low, Close, timeperiod=20)
    tmp['CCI88'] = talib.CCI(High, Low, Close, timeperiod=88)

    # 钱德动量摆动指标（Chande Momentum Osciliator），与其他动量指标摆动指标如相对强弱指标（RSI）和随机指标（KDJ）不同，
    # 钱德动量指标在计算公式的分子中采用上涨日和下跌日的数据。属于超买超卖型因子
    tmp['CMO_Close'] = talib.CMO(Close, timeperiod=14)
    tmp['CMO_Open'] = talib.CMO(Close, timeperiod=14)

    # DEMA双指数移动平均线
    tmp['DEMA6'] = talib.DEMA(Close, timeperiod=6)
    tmp['DEMA12'] = talib.DEMA(Close, timeperiod=12)
    tmp['DEMA26'] = talib.DEMA(Close, timeperiod=26)

    # DX 动向指数
    tmp['DX'] = talib.DX(High, Low, Close, timeperiod=14)

    # EMA 指数移动平均线
    tmp['EMA6'] = talib.EMA(Close, timeperiod=6)
    tmp['EMA12'] = talib.EMA(Close, timeperiod=12)
    tmp['EMA26'] = talib.EMA(Close, timeperiod=26)

    # KAMA 适应性移动平均线
    tmp['KAMA'] = talib.KAMA(Close, timeperiod=30)

    # MACD
    tmp['MACD_DIF'], tmp['MACD_DEA'], tmp['MACD_bar'] = talib.MACD(Close, fastperiod=12, slowperiod=24, signalperiod=9)

    # 中位数价格 不知道是什么意思
    tmp['MEDPRICE'] = talib.MEDPRICE(High, Low)

    # 负向指标 负向运动
    tmp['MiNUS_DI'] = talib.MINUS_DI(High, Low, Close, timeperiod=14)
    tmp['MiNUS_DM'] = talib.MINUS_DM(High, Low, timeperiod=14)

    # 动量指标（Momentom Index），动量指数以分析股价波动的速度为目的，研究股价在波动过程中各种加速，
    # 减速，惯性作用以及股价由静到动或由动转静的现象。属于趋势型因子
    tmp['MOM'] = talib.MOM(Close, timeperiod=10)

    # 归一化平均值范围
    tmp['NATR'] = talib.NATR(High, Low, Close, timeperiod=14)

    # OBV 	能量潮指标（On Balance Volume，OBV），以股市的成交量变化来衡量股市的推动力，
    # 从而研判股价的走势。属于成交量型因子
    # tmp['OBV'] = talib.OBV(Close, Volume)

    # PLUS_DI 更向指示器
    tmp['PLUS_DI'] = talib.PLUS_DI(High, Low, Close, timeperiod=14)
    tmp['PLUS_DM'] = talib.PLUS_DM(High, Low, timeperiod=14)

    # PPO 价格振荡百分比
    tmp['PPO'] = talib.PPO(Close, fastperiod=6, slowperiod=26, matype=0)

    # ROC 6日变动速率（Price Rate of Change），以当日的收盘价和N天前的收盘价比较，
    # 通过计算股价某一段时间内收盘价变动的比例，应用价格的移动比较来测量价位动量。属于超买超卖型因子。
    tmp['ROC6'] = talib.ROC(Close, timeperiod=6)
    tmp['ROC20'] = talib.ROC(Close, timeperiod=20)
    # 12日量变动速率指标（Volume Rate of Change），以今天的成交量和N天前的成交量比较，
    # 通过计算某一段时间内成交量变动的幅度，应用成交量的移动比较来测量成交量运动趋向，
    # 达到事先探测成交量供需的强弱，进而分析成交量的发展趋势及其将来是否有转势的意愿，
    # 属于成交量的反趋向指标。属于成交量型因子
    # tmp['VROC6'] = talib.ROC(Volume, timeperiod=6)
    # tmp['VROC20'] = talib.ROC(Volume, timeperiod=20)

    # ROC 6日变动速率（Price Rate of Change），以当日的收盘价和N天前的收盘价比较，
    # 通过计算股价某一段时间内收盘价变动的比例，应用价格的移动比较来测量价位动量。属于超买超卖型因子。
    tmp['ROCP6'] = talib.ROCP(Close, timeperiod=6)
    tmp['ROCP20'] = talib.ROCP(Close, timeperiod=20)
    # 12日量变动速率指标（Volume Rate of Change），以今天的成交量和N天前的成交量比较，
    # 通过计算某一段时间内成交量变动的幅度，应用成交量的移动比较来测量成交量运动趋向，
    # 达到事先探测成交量供需的强弱，进而分析成交量的发展趋势及其将来是否有转势的意愿，
    # 属于成交量的反趋向指标。属于成交量型因子
    # tmp['VROCP6'] = talib.ROCP(Volume, timeperiod=6)
    # tmp['VROCP20'] = talib.ROCP(Volume, timeperiod=20)

    # RSI
    tmp['RSI'] = talib.RSI(Close, timeperiod=14)

    # SAR 抛物线转向
    tmp['SAR'] = talib.SAR(High, Low, acceleration=0.02, maximum=0.2)

    # TEMA
    tmp['TEMA6'] = talib.TEMA(Close, timeperiod=6)
    tmp['TEMA12'] = talib.TEMA(Close, timeperiod=12)
    tmp['TEMA26'] = talib.TEMA(Close, timeperiod=26)

    # TRANGE 真实范围
    tmp['TRANGE'] = talib.TRANGE(High, Low, Close)

    # TYPPRICE 典型价格
    tmp['TYPPRICE'] = talib.TYPPRICE(High, Low, Close)

    # TSF 时间序列预测
    tmp['TSF'] = talib.TSF(Close, timeperiod=14)

    # ULTOSC 极限振子
    tmp['ULTOSC'] = talib.ULTOSC(High, Low, Close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

    # 威廉指标
    tmp['WILLR'] = talib.WILLR(High, Low, Close, timeperiod=14)

    # 标准化
    if normalization:
        factors_list = tmp.columns.tolist()[1:]

        if rolling >= 26:
            for i in factors_list:
                tmp[i] = (tmp[i] - tmp[i].rolling(window=rolling, center=False).mean()) \
                         / tmp[i].rolling(window=rolling, center=False).std()
        elif rolling < 26 & rolling > 0:
            print('Recommended rolling range greater than 26')
        elif rolling <= 0:
            for i in factors_list:
                tmp[i] = (tmp[i] - tmp[i].mean()) / tmp[i].std()

    if drop:
        tmp.dropna(inplace=True)

    tmp.set_index('tradeTime', inplace=True)

    return tmp


def dense_to_one_hot(labels_dense):
    """标签 转换one hot 编码
    输入labels_dense 必须为非负数
    2016-11-21
    """
    num_classes = len(np.unique(labels_dense))  # np.unique 去掉重复函数
    raws_labels = labels_dense.shape[0]
    index_offset = np.arange(raws_labels) * num_classes
    labels_one_hot = np.zeros((raws_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class Classifier_PonderDNC_BasicLSTM_L3(object):
    def __init__(self,
                 inputs,
                 targets,
                 gather_list=None,
                 mini_batch_size=1,
                 hidden_size=10,
                 memory_size=10,
                 threshold=0.99,
                 pondering_coefficient=1e-2,
                 num_reads=3,
                 num_writes=1,
                 learning_rate=1e-4,
                 optimizer_epsilon=1e-10,
                 max_gard_norm=50):

        self._tmp_inputs = inputs
        self._tmp_targets = targets
        self._in_length = None
        self._in_width = inputs.shape[2]
        self._out_length = None
        self._out_width = targets.shape[2]
        self._mini_batch_size = mini_batch_size
        self._batch_size = inputs.shape[1]

        # 声明计算会话
        self._sess = tf.InteractiveSession()

        self._inputs = tf.placeholder(dtype=tf.float32,
                                      shape=[self._in_length, self._batch_size, self._in_width],
                                      name='inputs')
        self._targets = tf.placeholder(dtype=tf.float32,
                                       shape=[self._out_length, self._batch_size, self._out_width],
                                       name='targets')

        act_core = DNCore_L3(hidden_size=hidden_size,
                             memory_size=memory_size,
                             word_size=self._in_width,
                             num_read_heads=num_reads,
                             num_write_heads=num_writes)
        self._InferenceCell = ACTCore(core=act_core,
                                      output_size=self._out_width,
                                      threshold=threshold,
                                      get_state_for_halting=self._get_hidden_state)

        self._initial_state = self._InferenceCell.initial_state(self._batch_size)

        tmp, act_final_cumul_state = \
            tf.nn.dynamic_rnn(cell=self._InferenceCell,
                              inputs=self._inputs,
                              initial_state=self._initial_state,
                              time_major=True)
        act_output, (act_final_iteration, act_final_remainder) = tmp

        # 测试
        self._final_iteration = tf.reduce_mean(act_final_iteration)

        self._act_output = act_output
        if gather_list is not None:
            out_sequences = tf.gather(act_output, gather_list)

        else:
            out_sequences = act_core

        # 设置损失函数
        pondering_cost = (act_final_iteration + act_final_remainder) * pondering_coefficient
        rnn_cost = tf.nn.softmax_cross_entropy_with_logits(
            labels=self._targets, logits=out_sequences)
        self._pondering_cost = tf.reduce_mean(pondering_cost)
        self._rnn_cost = tf.reduce_mean(rnn_cost)
        self._cost = self._pondering_cost + self._rnn_cost
        self._pred = tf.nn.softmax(out_sequences, dim=2)
        correct_pred = tf.equal(tf.argmax(self._pred, 2), tf.argmax(self._targets, 2))
        self._accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # 设置优化器
        # Set up optimizer with global norm clipping.
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self._cost, trainable_variables), max_gard_norm)
        global_step = tf.get_variable(
            name="global_step",
            shape=[],
            dtype=tf.int64,
            initializer=tf.zeros_initializer(),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate, epsilon=optimizer_epsilon)
        self._train_step = optimizer.apply_gradients(
            zip(grads, trainable_variables), global_step=global_step)


    def _get_hidden_state(self, state):
        controller_state, access_state, read_vectors = state
        layer_1, layer_2, layer_3 = controller_state
        L1_next_state, L1_next_cell = layer_1
        L2_next_state, L2_next_cell = layer_2
        L3_next_state, L3_next_cell = layer_3
        return tf.concat([L1_next_state, L2_next_state, L3_next_state], axis=-1)

    def fit(self,
            training_iters=1e2,
            display_step=5,
            save_path=None,
            restore_path=None):

        self._sess.run(tf.global_variables_initializer())
        # 保存和恢复
        self._variables_saver = tf.train.Saver()
        if restore_path is not None:
            self._variables_saver.restore(self._sess, restore_path)

        if self._batch_size == self._mini_batch_size:
            for scope in range(np.int(training_iters)):
                _, loss, acc, tp1, tp2, tp3 = \
                    self._sess.run([self._train_step,
                                    self._cost,
                                    self._accuracy,
                                    self._pondering_cost,
                                    self._rnn_cost,
                                    self._final_iteration],
                                   feed_dict={self._inputs: self._tmp_inputs, self._targets: self._tmp_targets})
                # 显示优化进程
                if scope % display_step == 0:
                    print(scope,
                          '  loss--', loss,
                          '  acc--', acc,
                          '  pondering_cost--', tp1,
                          '  rnn_cost--', tp2,
                          '  final_iteration', tp3)
                    # 保存模型可训练变量
                    if save_path is not None:
                        self._variables_saver.save(self._sess, save_path)

            print("Optimization Finished!")
        else:
            print('未完待续')

    def close(self):
        self._sess.close()
        print('结束进程，清理tensorflow内存/显存占用')

    def pred(self, inputs, gather_list=None, restore_path=None):
        if restore_path is not None:
            self._sess.run(tf.global_variables_initializer())
            self._variables_saver = tf.train.Saver()
            self._variables_saver.restore(self._sess, restore_path)

        output_pred = self._act_output
        if gather_list is not None:
            output_pred = tf.gather(output_pred, gather_list)
        probability = tf.nn.softmax(output_pred)
        classification = tf.argmax(probability, axis=-1)

        return self._sess.run([probability, classification], feed_dict={self._inputs: inputs})


def test1():
    op1 = Classifier_PonderDNC_BasicLSTM_L3(
        inputs=inputs,
        targets=targets,
        gather_list=gather_list,
        hidden_size=50,
        memory_size=50,
        pondering_coefficient=1e-2,
        learning_rate=1e-3)

    op1.fit(training_iters=10000,
            display_step=10,
            save_path=os.path.join(sys.path[0], "version1.0/ResidualPonderDNC_3.ckpt"),
            restore_path=os.path.join(sys.path[0], "version1.0/ResidualPonderDNC_3.ckpt"))
            # restore_path=None)
    op1.close()


if __name__ == '__main__':
    from research.rnn.xauusd_analysis import data_get
    # tmp = fix_data('白银88.csv')
    #
    # # targets 1d 数据合成
    # tmp_1d = data_frequency_transfer(tmp, '1d')
    rolling = 88
    # targets = tmp_1d
    tmp_1d = data_get()
    targets = tmp_1d
    targets['returns'] = targets['close'].shift(-1) / targets['close'] - 1.0
    targets['upper_boundary'] = targets.returns.rolling(rolling).mean() + 0.5 * targets.returns.rolling(rolling).std()
    targets['lower_boundary'] = targets.returns.rolling(rolling).mean() - 0.5 * targets.returns.rolling(rolling).std()
    targets.dropna(inplace=True)
    targets['labels'] = 1
    targets.loc[targets['returns'] >= targets['upper_boundary'], 'labels'] = 2
    targets.loc[targets['returns'] <= targets['lower_boundary'], 'labels'] = 0
    # factors 1d 数据合成
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
    # tmp['close'].plot()
    # plt.show()
    inputs = np.array(factors).reshape(-1, 1, factors.shape[1])
    targets = dense_to_one_hot(targets['labels'])
    targets = np.expand_dims(targets, axis=1)
    test1()


