# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: repeat_copy.py
@time: 2017/10/31 11:40
"""
import collections

import numpy as np
import tensorflow as tf
import sonnet as snt

DatasetTensors = collections.namedtuple('DatasetTensors', ('0bservations', 'target', 'mask'))

def masked_sigmoid_cross_entropy(logits, target, mask, time_average=False, log_prob_in_bits=False):
    xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits)
    loss_time_batch = tf.reduce_sum(xent, axis=2)
    loss_batch = tf.reduce_sum(loss_time_batch * mask, axis=0)
    batch_size = tf.cast(tf.shape(logits)[1], dtype=loss_time_batch.dtype)

    if time_average:
        mask_count = tf.reduce_sum(mask, axis=0)
        loss_batch/= (mask_count + np.finfo(np.float32).eps)
    loss = tf.reduce_sum(loss_batch) / batch_size
    if log_prob_in_bits:
        loss/=tf.log(2.)
    return loss
def bitstring_readable(data, batch_size, model_output=None, whole_batch=False):
    def _readable(datum):
        return '+' + ' '.join(['-' if x==0 else '%d' % x for x in datum]) + '+'
    obs_batch = data.observations
    targ_batch = data.target
    iterate_over = range(batch_size) if whole_batch else range(1)
    batch_strings = []
    for batch_index in iterate_over:
        obs = obs_batch[:, batch_index, :]
        targ = targ_batch[:, batch_index, :]
        obs_channels = range(obs.shape[1])
        targ_channels = range(targ.shape[1])
        obs_channel_strings = [_readable(obs[:, i]) for i in obs_channels]
        targ_channel_strings = [_readable(targ[:, i]) for i in targ_channels]

        readable_obs = 'Observations:\n' + '\n'.join(obs_channel_strings)
        readable_targ = 'Targets:\n' + '\n'.join(targ_channel_strings)
        strings = [readable_obs, readable_targ]

        if model_output is not None:
            output = model_output[:, batch_index, :]
            output_strings = [_readable(output[:, i]) for i in targ_channels]
            strings.append('Model Output:\n' + '\n'.join(output_strings))

        batch_strings.append('Model Output:\n' + '\n'.join(output_strings))
    return '\n' +'\n\n\n\n'.join(batch_strings)
class RepeatCopy(snt.AbstractModule):
    def __init__(self,
                 num_bits=6,
                 batch_size=1,
                 min_length=1,
                 max_length=1,
                 min_repeats=1,
                 max_repeats=2,
                 norm_max=10,
                 log_prob_in_bits=False,
                 time_average_cost=False,
                 name='repeat_copy',):
        super(RepeatCopy, self).__init__(name=name)

        self._batch_size = batch_size
        self._num_bits = num_bits
        self._min_length = min_length
        self._max_length = max_length
        self._min_repeats = min_repeats
        self._max_repeats = max_repeats
        self._norm_max = norm_max
        self._log_prob_in_bits = log_prob_in_bits
        self._time_average_cost = time_average_cost

    def _normalise(self, val):
        return val/self._norm_max
    def _unnormalise(self, val):
        return val * self._norm_max

    @property
    def time_average_cost(self):
        return self._time_average_cost

    @property
    def log_prob_in_bits(self):
        return self._log_prob_in_bits












