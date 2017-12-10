# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: tf_softmax_learn.py
@time: 2017/10/26 14:16
"""
import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Hide messy TensorFlow warnings

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(X,W)+b)

cross_entropy = -tf.reduce_sum(Y*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(100000):
        batch_trx, batch_try = mnist.train.next_batch(256)
        sess.run(train_step, feed_dict={X: batch_trx, Y: batch_try})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

