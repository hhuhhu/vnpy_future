# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: util.py
@time: 2017/10/26 9:49
"""
import numpy as np
import tensorflow as tf


def batch_invert_permutation(permutations):
    with tf.name_scope('batch_invert_permutation', values=[permutations]):
        unpacked = tf.unstack(permutations)
        inverses = [tf.invert_permutation(permutation) for permutation in unpacked]
        return tf.stack(inverses)


def batch_gather(values, indices):
    with tf.name_scope("batch_gather", vlaues=[values, indices]):
        unpacked = zip((tf.unstack(values), tf.unstack(indices)))
        result = [tf.gather(value, index) for value, index in unpacked]
        return tf.stack(result)


def one_hot(length, index):
    result = np.zeros(length)
    result[index] = 1
    return result
