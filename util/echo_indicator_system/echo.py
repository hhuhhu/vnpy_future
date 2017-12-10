# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: echo.py
@time: 2017/8/4 8:52
"""
import os
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework.ops import convert_to_tensor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Hide messy TensorFlow warnings

class ECHO(rnn_cell_impl.RNNCell):
    def __init__(self, num_units, wr2_scale=0.7, connectivity=0.1, leaky=0.3, activation=math_ops.sigmoid,
                 win_init=init_ops.random_normal_initializer(),
                 wr_init=init_ops.random_normal_initializer(),
                 bias_init=init_ops.random_normal_initializer()):
        in_state = activation()
        self.in_weight = 0.0
        self.h_wight = 0.0
        self.out_weith = 0.0


    def internal_state(self):
        internal_state = self.in_weight*self.input + self.h_wight*self.internal_state + self.out_weith*output
        return internal_state
    def output(self):
        output =0
        return

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """ Run one step of ESN Cell

            Args:
              inputs: `2-D Tensor` with shape `[batch_size x input_size]`.
              state: `2-D Tensor` with shape `[batch_size x self.state_size]`.
              scope: VariableScope for the created subgraph; defaults to class `ESNCell`.

            Returns:
              A tuple `(output, new_state)`, computed as
              `output = new_state = (1 - leaky) * state + leaky * activation(Win * input + Wr * state + B)`.

            Raises:
              ValueError: if `inputs` or `state` tensor size mismatch the previously provided dimension.
              """

        inputs = convert_to_tensor(inputs)
        input_size = inputs.get_shape().as_list()[1]
        dtype = inputs.dtype

        with vs.variable_scope(scope or type(self).__name__):  # "ESNCell"

            win = vs.get_variable("InputMatrix", [input_size, self._num_units], dtype=dtype,
                                  trainable=False, initializer=self._win_initializer)
            wr = vs.get_variable("ReservoirMatrix", [self._num_units, self._num_units], dtype=dtype,
                                 trainable=False, initializer=self._wr_initializer)
            b = vs.get_variable("Bias", [self._num_units], dtype=dtype, trainable=False,
                                initializer=self._bias_initializer)

            in_mat = array_ops.concat([inputs, state], axis=1)
            weights_mat = array_ops.concat([win, wr], axis=0)

            output = (1 - self._leaky) * state + self._leaky * self._activation(
                math_ops.matmul(in_mat, weights_mat) + b)

        return output, output

