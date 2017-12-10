# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: ops.py
@time: 2017/12/9 15:12
"""
import math
import functools

import numpy as np
import tensorflow as tf


def conv1d(input,
           output_dim,
           conv_w=9,
           conv_s=2,
           padding="SAME",
           name="conv1d",
           stddev=0.02,
           bias=False):
    with tf.variable_scope(name):
        w = tf.