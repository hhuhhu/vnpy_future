# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: access.py
@time: 2017/10/26 9:47
"""
import collections
import sonnet as snt
import tensorflow as tf

from research.dnc_learn import addressing
from research.dnc_learn import util

AccessState = collections.namedtuple('AccessState', ('memory', 'read_weights', 'write_weights', 'linkage', 'usage'))


def _erase_and_write(memory, address, reset_weights, values):
    with tf.name_scope("erase_memory", values=[memory, address, reset_weights]):
        expand_address = tf.expand_dims(address, 3)
        reset_weights = tf.expand_dims(reset_weights, 2)
        weighted_resets = expand_address * reset_weights
        reset_gate = tf.reduce_prod(1 - weighted_resets, [1])
        memory *= reset_gate

    with tf.name_scope('additive_write', values=[memory, address, values]):
        add_matrix = tf.matmul(address, values, adjoint_a=True)
        memory += add_matrix
    return memory


class MemoryAccess(snt.RNNCore):
    def __init__(self, memory_size=128, word_size=20, num_reads=1, num_writes=1, name="memory_access"):
        super(MemoryAccess, self).__init__(name=name)
        self._memory_size = memory_size
        self._word_size = word_size
        self._num_reads = num_reads
        self._num_writes = num_writes
        self._write_content_weights_mod = addressing.CosineWeights(num_writes, word_size, name='write_content_weights')
        self._read_content_weights_mod = addressing.CosineWeights(num_reads, word_size, name='read_content_weights')
        self._linkage = addressing.TemporalLinkage(memory_size, num_writes)
        self._freeness = addressing.Freeness(memory_size)

    def _build(self, inputs, prev_state):
        inputs = self._read_inputs(inputs)
        usage = self._freeness(
            write_weights=prev_state.write_weights,
            free_gate=inputs['free_gate'],
            read_weights=prev_state.read_weights,
            prev_usage=prev_state.usage
        )
        write_weights = self._write_weights(inputs, prev_state.memory, usage)
        memory = _erase_and_write(
            prev_state.memory,
            address=write_weights,
            reset_weights=inputs['erase_vectors'],
            values=inputs['write_vectors']
        )
        linkage_state = self._linkage(write_weights, prev_state.linkage)
        read_weights = self._read_weights(inputs, memory=memory,
                                          prev_read_weights=prev_state.read_weights,
                                          link=linkage_state.link)
        read_words = tf.matmul(read_weights, memory)
        return (read_words, AccessState(
            memory=memory,
            read_weights=read_weights,
            write_weights=write_weights,
            linkage=linkage_state,
            usage=usage
        ))

    def _read_inputs(self, inputs):
        def _linear(first_dim, second_dim, name, activation=None):
            linear = snt.Linear(first_dim * second_dim, name, activation=None)
            if activation is not None:
                linear = activation(linear, name='{}_activation'.format(name))
            return tf.reshape(linear, [-1, first_dim, second_dim])

        write_vectors = _linear(self._num_writes, self._word_size, 'write_vectors')
        erase_vectors = _linear(self._num_writes, self._word_size, 'erase_vectors', tf.sigmoid)
        free_gate = tf.sigmoid(snt.Linear(self._num_reads, name='free_gate')(inputs))
        allocation_gate = tf.sigmoid(snt.Linear(self._num_writes, name='allocation_gate')(inputs))
        write_gate = tf.sigmoid(snt.Linear(self._num_writes, name='write_gate')(inputs))
        num_read_modes = 1 + 2 * self._num_writes
        read_mod = snt.BatchApply(tf.nn.softmax)(_linear(self._num_reads, num_read_modes, name='read_mode'))
        write_keys = _linear(self._num_reads, self._word_size, 'write_keys')
        write_strengths = snt.Linear(self._num_writes, name='write_strengths')(inputs)
        read_keys = _linear(self._num_reads, self._word_size, 'read_keys')
        read_strengths = snt.Linear(self._num_reads, name='read_strengths')(inputs)

        result = {
            'read_content_keys': read_keys,
            'read_content_strengths': read_strengths,
            'write_content_keys': write_keys,
            'write_content_strengths': write_strengths,
            'wirte_vectors': write_vectors,
            'erase_vectors': erase_vectors,
            'free_gate': free_gate,
            'allocation_gate': allocation_gate,
            'write_gate': write_gate,
            'read_mode': read_mod,
        }
        return result

    def _write_weights(self, inputs, memory, usage):
        with tf.name_scope('write_weights', values=[inputs, memory, usage]):
            write_content_weights = self._write_content_weights_mod(memory, inputs['write_content_keys'],
                                                                    inputs['write_content_strengths'])
            write_allocation_weights = self._freeness.write_allocation_weights(
                usage=usage,
                write_gates=(inputs['allocation_gate'] * inputs['write_gate']),
                num_writes=self._num_writes
            )
            allocation_gate = tf.expand_dims(inputs['allocation_gate'], -1)
            write_gate = tf.expand_dims(inputs['write_gate'], -1)
            return write_gate * (
                allocation_gate * write_allocation_weights + (1 - allocation_gate) * write_content_weights)

    def _read_weights(self, inputs, memory, prev_read_weights, link):
        with tf.name_scope('read_weights', values=[inputs, memory, prev_read_weights, link]):
            content_weights = self._read_content_weights_mod(memory, inputs['read_content_keys'],
                                                             inputs['read_content_strengths'])
            forward_weights = self._linkage.directional_read_weights(link, prev_read_weights, forward=True)
            backward_weights = self._linkage.directional_read_weights(link, prev_read_weights, forward=False)
            backward_mode = inputs['read_mode'][:, :, :self._num_writes]
            forward_mode = (inputs['read_mode'][:, :, self._num_writes:2 * self._num_writes])
            content_mode = inputs['read_mode'][:, :, 2 * self._num_writes]
            read_weigths = (tf.expand_dims(content_mode, 2) * content_weights + tf.reduce_sum(
                tf.expand_dims(forward_mode, 3) * forward_weights, 2)
                            + tf.reduce_sum(tf.expand_dims(backward_mode, 3) * backward_weights, 2))
            return read_weigths

    @property
    def state_size(self):
        return AccessState(memory=tf.TensorShape([self._memory_size, self._word_size]),
                           read_weights=tf.TensorShape([self._num_reads, self._memory_size]),
                           write_weights=tf.TensorShape([self._num_writes, self._memory_size]),
                           linkage=self._linkage.state_size,
                           usage=self._freeness.state_size)

    @property
    def output_size(self):
        return tf.TensorShape([self._num_reads, self._word_size])
