# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: tensorflow_learn.py
@time: 2017/10/23 18:25
"""
import numpy as np
import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    i = tf.placeholder(tf.float32, shape=(None, 1), name="i")
    a = tf.Variable(tf.constant(10.0, shape=[1]), name="a")
    b = tf.Variable(tf.constant(10.0, shape=[2]), name="b")
    n1 = tf.add(i, a, name="n1")
    n2 = tf.add(i, b, name="n2")
    n3 = tf.add(n1, n2, name="n3")

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    input_data = np.array([2]).reshape(-1, 1)
    output = sess.run(n3, feed_dict={i: input_data})
    print(input_data)
    print(output)
