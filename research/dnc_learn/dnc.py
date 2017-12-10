# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: dnc.py
@time: 2017/10/26 9:45
"""
import collections
import numpy as np
import sonnet as snt
import tensorflow as tf

from research.dnc_learn import access

DNCStore = collections.namedtuple('DNCCore', ('access_output', 'access_state', 'controller_state'))


class DNC(snt.RNNCore):
    def __init__(self, access_config, controller_config, output_size, clip_value=None, name='dnc'):
        super(DNC, self).__init__(name=name)
        with self._enter_variable_scope():
            self._controller = snt.LSTM(**controller_config)
            self._access = access.MemoryAccess(**access_config)
        self._access_output_size = np.prod(self._access.output_size.as_list())
        self._output_size = output_size
        self._clip_value = clip_value or 0
        self._output_size = tf.TensorShape([output_size])
        self._state_size = DNCStore(access_output=self._access_output_size,
                                    access_state=self._access.state_size,
                                    controller_state=self._controller.state_size)

    def _clip_if_enabled(self, x):
        if self._clip_value > 0:
            return tf.clip_by_value(x, -self._clip_value, self._clip_value)
        else:
            return x

    def _build(self, inputs, prev_state):
        prev_access_output = prev_state.access_output
        prev_access_state = prev_state.access_state
        prev_controller_state = prev_state.controller_state

        batch_flatten = snt.BatchFlatten()
        controller_input = tf.concat([batch_flatten(inputs), batch_flatten(prev_access_output)], 1)

        controller_output, controller_state = self._controller(controller_input, prev_controller_state)
        controller_output = self._clip_if_enabled(controller_output)
        controller_state = snt.nest.map(self._clip_if_enabled, controller_state)
        access_output, access_state = self._access(controller_output, prev_access_state)
        output = tf.concat([controller_output, batch_flatten(access_output)], 1)
        output = snt.Linear(output_size=self._output_size.as_list()[0], name='output_linear')(output)
        output = self._clip_if_enabled(output)
        return output, DNCStore(access_output=access_output,
                                access_state=access_state,
                                controller_state=controller_state
                                )

    def initial_state(self, batch_size, dtype=tf.float32):
        return DNCStore(
            controller_state=self._controller.initial_state(batch_size, dtype),
            access_state=self._access.initial_state(batch_size, dtype),
            access_output=tf.zeros([batch_size] + self._access.output_size.as_list(), dtype)
        )

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size
