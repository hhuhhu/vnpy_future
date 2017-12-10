# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: access.py
@time: 2017/11/19 17:27
"""
import tensorflow as tf
from research.reforcement_learning.actor_critic import ActorNet, CriticNet

LEARNING_RATE = 1e-3
DECAY_RATE = 0.99


class Access:
    def __init__(self, batch_size, state_size, action_size):
        with tf.variable_scope('Access'):
            self.inputs = tf.placeholder(tf.float32, [None, batch_size, state_size], name='inputs')
            self.actor = ActorNet()
            self.critic = CriticNet()
            self.policy = tf.nn.softmax(self.actor(action_size, self.inputs))
            self.value = self.critic(self.inputs)

        self.optimizer_actor = tf.train.RMSPropOptimizer(LEARNING_RATE, DECAY_RATE, name='optimizer_actor')
        self.optimizer_critic = tf.train.RMSPropOptimizer(LEARNING_RATE, DECAY_RATE, name='optimizer_critic')

        var_list = self.get_trainable()
        var_list = list(var_list[0] + var_list[1])
        self.saver = tf.train.Saver(var_list=var_list)

    def get_trainable(self):
        return [self.actor.get_variables(), self.critic.get_variables()]

    def save(self, sess, path):
        self.saver.save(sess, path)

    def restore(self, sess, path):
        var_list = list(self.get_trainable()[0] + self.get_trainable()[1])
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, path)
