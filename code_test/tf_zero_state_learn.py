# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: tf_zero_state_learn.py
@time: 2017/7/20 14:18
"""
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl

batch_size = tf.placeholder(tf.int32, 5)
cell = rnn_cell_impl.BasicLSTMCell(128)
initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)