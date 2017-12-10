# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: main.py
@time: 2017/10/20 10:12
"""

import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell, static_rnn
from tensorflow.contrib.legacy_seq2seq import sequence_loss_by_example
from tensorflow.python.ops import array_ops
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.rnn import static_rnn
from tensorflow.python.ops import variable_scope as vs


class ACTCell(RNNCell):
    """
    A RNN cell implementing Graves' Adaptive Computation Time algorithm
    """
    def __init__(self, num_units, cell, epsilon,
                 max_computation, batch_size, sigmoid_output=False):

        self.batch_size = batch_size
        self.one_minus_eps = tf.fill([self.batch_size], tf.constant(1.0 - epsilon, dtype=tf.float32))
        self._num_units = num_units
        self.cell = cell
        self.max_computation = max_computation
        self.ACT_remainder = []
        self.ACT_iterations = []
        self.sigmoid_output = sigmoid_output

        if hasattr(self.cell, "_state_is_tuple"):
            self._state_is_tuple = self.cell._state_is_tuple
        else:
            self._state_is_tuple = False

    @property
    def input_size(self):
        return self._num_units
    @property
    def output_size(self):
        return self._num_units
    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, timestep=0, scope=None):

        if self._state_is_tuple:
            state = tf.concat(state, 1)

        with vs.variable_scope(scope or type(self).__name__):
            # define within cell constants/ counters used to control while loop for ACTStep
            prob = tf.fill([self.batch_size], tf.constant(0.0, dtype=tf.float32), "prob")
            prob_compare = tf.zeros_like(prob, tf.float32, name="prob_compare")
            counter = tf.zeros_like(prob, tf.float32, name="counter")
            acc_outputs = tf.fill([self.batch_size, self.output_size], 0.0, name='output_accumulator')
            acc_states = tf.zeros_like(state, tf.float32, name="state_accumulator")
            batch_mask = tf.fill([self.batch_size], True, name="batch_mask")


            # While loop stops when this predicate is FALSE.
            # Ie all (probability < 1-eps AND counter < N) are false.
            def halting_predicate(batch_mask, prob_compare, prob,
                          counter, state, input, acc_output, acc_state):
                return tf.reduce_any(tf.logical_and(
                        tf.less(prob_compare,self.one_minus_eps),
                        tf.less(counter, self.max_computation)))

            # Do while loop iterations until predicate above is false.
            _,_,remainders,iterations,_,_,output,next_state = \
                tf.while_loop(halting_predicate, self.act_step,
                              loop_vars=[batch_mask, prob_compare, prob,
                                         counter, state, inputs, acc_outputs, acc_states])

        #accumulate remainder  and N values
        self.ACT_remainder.append(tf.reduce_mean(1 - remainders))
        self.ACT_iterations.append(tf.reduce_mean(iterations))

        if self.sigmoid_output:
            output = tf.sigmoid(tf.contrib.rnn.BasicRNNCell._linear(output,self.batch_size,0.0))

        if self._state_is_tuple:
            next_c, next_h = tf.split(next_state, 2, 1)
            next_state = tf.contrib.rnn.LSTMStateTuple(next_c, next_h)

        return output, next_state

    def calculate_ponder_cost(self, time_penalty):
        '''returns tensor of shape [1] which is the total ponder cost'''
        return time_penalty * tf.reduce_sum(
            tf.add_n(self.ACT_remainder)/len(self.ACT_remainder) +
            tf.to_float(tf.add_n(self.ACT_iterations)/len(self.ACT_iterations)))

    def act_step(self,batch_mask,prob_compare,prob,counter,state,input,acc_outputs,acc_states):
        '''
        General idea: generate halting probabilites and accumulate them. Stop when the accumulated probs
        reach a halting value, 1-eps. At each timestep, multiply the prob with the rnn output/state.
        There is a subtlety here regarding the batch_size, as clearly we will have examples halting
        at different points in the batch. This is dealt with using logical masks to protect accumulated
        probabilities, states and outputs from a timestep t's contribution if they have already reached
        1 - es at a timstep s < t. On the last timestep for each element in the batch the remainder is
        multiplied with the state/output, having been accumulated over the timesteps, as this takes
        into account the epsilon value.
        '''

        # If all the probs are zero, we are seeing a new input => binary flag := 1, else 0.
        binary_flag = tf.cond(tf.reduce_all(tf.equal(prob, 0.0)),
                              lambda: tf.ones([self.batch_size, 1], dtype=tf.float32),
                              lambda: tf.zeros([self.batch_size, 1], tf.float32))

        input_with_flags = tf.concat([binary_flag, input], 1)

        if self._state_is_tuple:
            (c, h) = tf.split(state, 2, 1)
            state = tf.contrib.rnn.LSTMStateTuple(c, h)

        output, new_state = static_rnn(cell=self.cell, inputs=[input_with_flags], initial_state=state, scope=type(self.cell).__name__)

        if self._state_is_tuple:
            new_state = tf.concat(new_state, 1)

        with tf.variable_scope('sigmoid_activation_for_pondering'):
            p = tf.squeeze(tf.layers.dense(new_state, 1, activation=tf.sigmoid, use_bias=True), squeeze_dims=1)

        # Multiply by the previous mask as if we stopped before, we don't want to start again
        # if we generate a p less than p_t-1 for a given example.
        new_batch_mask = tf.logical_and(tf.less(prob + p, self.one_minus_eps), batch_mask)
        new_float_mask = tf.cast(new_batch_mask, tf.float32)

        # Only increase the prob accumulator for the examples
        # which haven't already passed the threshold. This
        # means that we can just use the final prob value per
        # example to determine the remainder.
        prob += p * new_float_mask

        # This accumulator is used solely in the While loop condition.
        # we multiply by the PREVIOUS batch mask, to capture probabilities
        # that have gone over 1-eps THIS iteration.
        prob_compare += p * tf.cast(batch_mask, tf.float32)

        # Only increase the counter for those probabilities that
        # did not go over 1-eps in this iteration.
        counter += new_float_mask

        # Halting condition (halts, and uses the remainder when this is FALSE):
        # If any batch element still has both a prob < 1 - epsilon AND counter < N we
        # continue, using the outputed probability p.
        counter_condition = tf.less(counter, self.max_computation)

        final_iteration_condition = tf.logical_and(new_batch_mask, counter_condition)
        use_remainder = tf.expand_dims(1.0 - prob, -1)
        use_probability = tf.expand_dims(p, -1)
        update_weight = tf.where(final_iteration_condition, use_probability, use_remainder)
        float_mask = tf.expand_dims(tf.cast(batch_mask, tf.float32), -1)

        acc_state = (new_state * update_weight * float_mask) + acc_states
        acc_output = (output[0] * update_weight * float_mask) + acc_outputs

        return [new_batch_mask, prob_compare, prob, counter, new_state, input, acc_output, acc_state]

class ACTModel(object):

    def __init__(self, config, is_training=False):
        self.config = config
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.hidden_size = hidden_size = config.hidden_size
        self.num_layers = 1
        vocab_size = config.vocab_size
        self.max_grad_norm = config.max_grad_norm
        self.use_lstm = config.use_lstm

        # Placeholders for inputs.
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.initial_state = array_ops.zeros(tf.stack([self.batch_size, self.num_steps]),
                 dtype=tf.float32).set_shape([None, self.num_steps])

        embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.hidden_size])

        # Set up ACT cell and inner rnn-type cell for use inside the ACT cell.
        with tf.variable_scope("rnn"):
            if self.use_lstm:
                inner_cell = BasicLSTMCell(self.config.hidden_size)
            else:
                inner_cell = GRUCell(self.config.hidden_size)

        with tf.variable_scope("ACT"):

            act = ACTCell(self.config.hidden_size, inner_cell, config.epsilon,
                          max_computation=config.max_computation, batch_size=self.batch_size)

        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        inputs = [tf.squeeze(single_input, [1]) for single_input in tf.split(inputs, self.config.num_steps, 1)]

        self.outputs, final_state = static_rnn(act, inputs, dtype = tf.float32)

        # Softmax to get probability distribution over vocab.
        output = tf.reshape(tf.concat(self.outputs, 1), [-1, hidden_size])
        softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        self.logits = tf.matmul(output, softmax_w) + softmax_b   # dim (numsteps*batchsize, vocabsize)

        loss = sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([batch_size * num_steps])],
                vocab_size)

        # Add up loss and retrieve batch-normalised ponder cost: sum N + sum Remainder.
        ponder_cost = act.calculate_ponder_cost(time_penalty=self.config.ponder_time_penalty)
        self.cost = (tf.reduce_sum(loss) / batch_size) + ponder_cost
        self.final_state = self.outputs[-1]

        if is_training:
            self.lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))