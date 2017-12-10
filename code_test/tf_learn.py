# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: tf_learn.py
@time: 2017/10/26 9:57
"""
import os

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Hide messy TensorFlow warnings


graph = tf.Graph()
with graph.as_default():
    a = tf.placeholder(tf.int32, shape=None, name='input')
    b = tf.invert_permutation(a)
    c = tf.reduce_sum(a*a,axis=2,keep_dims=True)

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    input = [[[2]]]
    output = sess.run(c, feed_dict={a: input})
    print(output)

tf.nn.softmax()


