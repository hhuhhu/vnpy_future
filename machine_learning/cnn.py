# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: cnn.py
@time: 2017/12/9 15:57
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

from machine_learning.data_load import read_data_sets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Hide messy TensorFlow warnings
ops.reset_default_graph()

sess = tf.Session()

data_dir = 'data/mnist'
mnist = read_data_sets(data_dir)

print(mnist.train.images)
x_train = np.array([np.reshape(x, (28,28)) for x in mnist.train.images])
x_test = np.array([np.reshape(x, (28,28)) for x in mnist.test.images])

train_labels = mnist.train.labels
test_labels = mnist.test.labels

batch_size = 100
learning_rate = 0.005
evaluation_size = 500
image_width = x_train[0].shape[0]
image_height = x_train.shape[1]

target_size = max(train_labels) + 1
num_channels = 1
generations = 500
eval_every = 5
conv1_features = 25
conv2_features = 50
max_pool_size1 = 2
max_pool_size2 = 2
fully_connected_size1 = 100

x_input_shape = (batch_size, image_width, image_height, num_channels)
x_input = tf.placeholder(tf.float32, shape=x_input_shape)
y_target = tf.placeholder(tf.int32, shape=(batch_size))
eval_input_shape = (evaluation_size, image_width, image_height, num_channels)
eval_input = tf.placeholder(tf.float32, shape=eval_input_shape)
eval_target = tf.placeholder(tf.int32, shape=(evaluation_size))
conv1_weight = tf.Variable(tf.truncated_normal([4,4,num_channels, conv1_features],stddev=0.1, dtype=tf.float32))
conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))
conv2_weight = tf.Variable(tf.truncated_normal([4,4,conv1_features,conv2_features],stddev=0.1,dtype=tf.float32))
conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))

resulting_width = image_width//(max_pool_size1*max_pool_size2)
resulting_height = image_height//(max_pool_size1*max_pool_size2)
full1_input_size = resulting_width * resulting_height*conv2_features
full1_weight = tf.Variable(tf.truncated_normal([full1_input_size, fully_connected_size1],stddev=0.1,dtype=tf.float32))
full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))
full2_weight = tf.Variable(tf.truncated_normal([fully_connected_size1, target_size], stddev=0.1, dtype=tf.float32))
full2_bias = tf.Varialbe(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))

def my_conv_net(input_data):
    conv1 = tf.nn.conv2d(input_data, conv1_weight, strides=[1,1,1,1],padding='SAME')
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    max_pool1 = tf.nn.max_pool(relu1, ksize=[1,max_pool_size1,max_pool_size1,1],strides=[1,max_pool_size1,max_pool_size1,1],
                               padding='SAME')
    conv2 = tf.nn.conv2d(max_pool1, conv1_weight, strides=[1,1,1,1], padding='SAME')
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_bias))
    max_pool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size2, max_pool_size2, 1],
                               strides=[1, max_pool_size2, max_pool_size2, 1], padding='SAME')
    final_conv_shape = max_pool2.get_shape().as_list()
    final_shape = final_conv_shape[1]*final_conv_shape[2]*final_conv_shape[3]
    flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])
    fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))
    final_model_output = tf.add(tf.matmul(fully_connected1, full2_weight), full2_bias)
    return (final_model_output)

model_output = my_conv_net(x_input)
test_model_output = my_conv_net(eval_input)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target))
prediction = tf.nn.softmax(model_output)
test_prediction = tf.nn.softmax(test_model_output)

def get_accuracy(logits, targets):
    batch_predictions = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_predictions, targets))
    return (100.*num_correct/batch_predictions.shape[0])
my_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_step = my_optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess.run(init)
train_loww = []
train_acc = []
test_acc = []
for i in range(generations):
    rand_index = np.random.choice(len(x_train), size=batch_size)
    rand_x = x_train[rand_index]
    rand_x = np.expand_dims(rand_x, 3)
    rand_y = train_labels[rand_index]
    train_dict = {x_input: rand_x, y_target:rand_y}

    sess.run(train_step, feed_dict=train_dict)
    temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict=train_dict)
    temp_train_acc = get_accuracy(temp_train_preds, rand_y)

    if(i+1)%eval_every ==0:
        eval_index = np.random.choice(len(x_test), size=evaluation_size)
        eval_x = x_test[eval_index]
        eval_x = np.expand_dims(eval_x, 3)
        eval_y = test_labels




