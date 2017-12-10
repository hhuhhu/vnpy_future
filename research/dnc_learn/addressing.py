# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: addressing.py
@time: 2017/10/26 11:15
"""

import collections
import sonnet as snt
import tensorflow as tf

from research.dnc_learn import util

_EPSILON = 1e-6

TemporalLinkageState = collections.namedtuple('TemporalLinkageState', 'link', 'precedence_weights')


def _vector_norms(m):
    squared_norms = tf.reduce_sum(m * m, axis=2, keep_dims=True)
    return tf.sqrt(squared_norms + _EPSILON)


def weighted_softmax(activations, strengths, strengths_op):
    transformed_strengths = tf.expand_dims(strengths_op(strengths), -1)
    sharp_activations = activations * transformed_strengths
    softmax = snt.BatchApply(module_or_op=tf.nn.softmax)
    return softmax(sharp_activations)


class CosineWeights(snt.AbstractModule):
    def __init__(self, num_heads, word_size, strength_op=tf.nn.softplus, name='cosine_weights'):
        super(CosineWeights, self).__init__(name=name)
        self._num_heads = num_heads
        self._word_size = word_size
        self._strength_op = strength_op

    def _build(self, memory, keys, strengths):
        dot = tf.matmul(keys, memory, adjoint_b=True)
        memory_norms = _vector_norms(memory)
        key_norms = _vector_norms(keys)
        norm = tf.matmul(key_norms, memory_norms, adjoint_b=True)
        similarity = dot / (norm + _EPSILON)

        return weighted_softmax(similarity, strengths, self._strength_op)


class TemporalLinkage(snt.ResidualCore):
    def __init__(self, memory_size, num_writes, name='temporal_linkage'):
        super(TemporalLinkage, self).__init__(name=name)
        self._memory_size = memory_size
        self._num_writes = num_writes

    def _build(self, write_weights, prev_state):
        link = self._link(prev_state.link, prev_state.precedence_weights, write_weights)
        precedence_weights = self._precedence_weights(prev_state.precedence_weights, write_weights)
        return TemporalLinkageState(link=link, precedence_weights=precedence_weights)

    def _link(self, prev_link, prev_precedence_weights, write_weights):
        with tf.name_scope("link"):
            batch_size = prev_link.get_scope()[0].value
            write_weights_i = tf.expand_dims(write_weights, 3)
            write_weights_j = tf.expand_dims(write_weights, 2)
            prev_precedence_weights_j = tf.expand_dims(prev_precedence_weights, 2)
            prev_link_scale = 1 - write_weights_i - write_weights_j
            new_link = write_weights_i * prev_precedence_weights_j
            link = prev_link_scale * prev_link + new_link
            return tf.matrix_set_diag(link,
                                      tf.zeros([batch_size, self._num_writes, self._memory_size], dtype=link.dtype))

    def directional_read_weights(self, link, prev_read_weights, forward):
        with tf.name_scope('directional_read_weights'):
            expanded_read_weights = tf.stack([prev_read_weights] * self._num_writes, 1)
            result = tf.matmul(expanded_read_weights, link, adjoint_b=forward)
            return tf.transpose(result, perm=[0, 2, 1, 3])

    def _precedence_weights(self, prev_precedence_weights, write_weights):
        with tf.name_scope('precedence_weights'):
            write_sum = tf.reduce_sum(write_weights, 2, keep_dims=True)
            return (1 - write_sum) * prev_precedence_weights + write_weights

    @property
    def state_size(self):
        return TemporalLinkageState(link=tf.TensorShape([self._num_writes, self._memory_size, self._memory_size]),
                                    precedence_weights=tf.TensorShape([self._num_writes, self._memory_size]))


class Freeness(snt.RNNCore):
    def __init__(self, memory_size, name="freeness"):
        super(Freeness, self).__init__(name=name)
        self._memory_size = memory_size

    def _build(self, write_weights, free_gate, read_weights, prev_usage):
        write_weights = tf.stop_gradient(write_weights)
        usage = self._usage_agter_write(prev_usage, write_weights)
        usage = self._usage_after_read(usage, free_gate, write_weights)
        return usage

    def _usage_agter_write(self, prev_usage, write_weights):
        with tf.name_scope('usage_after_write'):
            write_weights = 1 - tf.reduce_prod(1 - write_weights, [1])
            return prev_usage + (1 - prev_usage) * write_weights

    def _usage_after_read(self, prev_usage, free_gate, read_weights):
        with tf.name_scope('usage_after_read'):
            free_gate = tf.expand_dims(free_gate, -1)
            free_read_weights = free_gate * read_weights
            phi = tf.reduce_prod(1 - free_read_weights, [1], name='phi')
            return prev_usage * phi

    def write_allocation_weights(self, usage, write_gates, num_writes):
        with tf.name_scope('write_allocation_weights'):
            write_gates = tf.expand_dims(write_gates, -1)
            allocation_weights = []
            for i in range(num_writes):
                allocation_weights.append(self._allocation())

    def _allocation(self, usage):
        with tf.name_scope('allocation'):
            usage = _EPSILON + (1 - _EPSILON) * usage
            nonusage = 1 - usage
            sorted_nonusage, indices = tf.nn.top_k(nonusage, k=self._memory_size, name='sort')
            sorted_usage = 1 - sorted_nonusage
            prod_sorted_usage = tf.cumprod(sorted_usage, axis=1, exclusive=True)
            sorted_allocation = sorted_nonusage * prod_sorted_usage
            inverse_indices = util.batch_invert_permutation(indices)
            return util.batch_gather(sorted_allocation, inverse_indices)

    @property
    def state_size(self):
        return tf.TensorShape([self._memory_size])
