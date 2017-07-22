# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: tf_slice_learn.py
@time: 2017/7/12 15:25
"""
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Hide messy TensorFlow warnings

input_tensor = [[[1, 1, 1], [2, 2, 2]],
              [[3, 3, 3], [4, 4, 4]],
              [[5, 5, 5], [6, 6, 6]]]
test = tf.slice(input_tensor, [0, 1, 0], [3, -1, 1])
sess = tf.Session()
print(sess.run(test))