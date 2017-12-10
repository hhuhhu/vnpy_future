# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: echo_indicator_system.py
@time: 2017/7/12 15:39
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from pre_code.echo.esn_cell import ESNCell


class EchoIndicatorSystem:
    def __init__(self, data, units, connectivity, scale):
        esn = ESNCell(units, connectivity, scale, activation=math_ops.sigmoid)
        outputs, final_state = tf.nn.dynamic_rnn(esn, data, dtype=tf.float32)
    with tf.Session() as S:
        S.run(tf.global_variables_initializer())

        print("Computing embeddings...")
        res = S.run(washed)

        print("Computing direct solution...")
        state = np.array(res)
        tr_state = np.mat(state[:tr_size])
        ts_state = np.mat(state[tr_size:])
        wout = np.transpose(
            np.mat(data[washout_size + 1:tr_size + washout_size + 1]) * np.transpose(np.linalg.pinv(tr_state)))

        print("Testing performance...")
        ts_out = np.mat((np.transpose(ts_state * wout).tolist())[0][:-1])
        ts_y = np.mat(data[washout_size + tr_size + 1:])
        print("ts_y", ts_y)

        ts_mse = np.mean(np.square(ts_y - ts_out))

if __name__ == '__main__':
    data = [[1, 0], [0, 1]]
    eis = EchoIndicatorSystem(data, units=100)