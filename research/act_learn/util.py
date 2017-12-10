# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: util.py
@time: 2017/10/20 10:19
"""

import pickle
import collections
import os
import time

import numpy as np
import tensorflow as tf

import research.act_learn.config as cf
"""Utilities for parsing PTB text files."""


def _read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(),
                         key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data]


def ptb_raw_data(data_path, train, valid, test):
    """Load PTB raw data from data directory "data_path".
  
    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.
  
    The PTB dataset comes from Tomas Mikolov's webpage:
  
    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  
    Args:
      data_path: string path to the directory where simple-examples.tgz has
        been extracted.
  
    Returns:
      tuple (train_data, valid_data, test_data, vocabulary)
      where each of the data objects can be passed to PTBIterator.
    """
    train_path = os.path.join(data_path, train)
    valid_path = os.path.join(data_path, valid)
    test_path = os.path.join(data_path, test)

    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary, word_to_id


def ptb_iterator(raw_data, batch_size, num_steps):
    """Iterate on the raw PTB data.
  
    This generates batch_size pointers into the raw PTB data, and allows
    minibatch iteration along these pointers.
  
    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
  
    Yields:
      Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
      The second element of the tuple is the same data time-shifted to the
      right by one.
  
    Raises:
      ValueError: if batch_size or num_steps are too high.
    """
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)


def run_epoch(session, m, data, eval_op, max_steps=None, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    num_batch_steps_completed = 0

    for step, (x, y) in enumerate(ptb_iterator(data, m.batch_size, m.num_steps)):
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y})
        costs += cost
        iters += m.num_steps
        num_batch_steps_completed += 1

        #if verbose and step % (epoch_size // 10) == 10:
        print("%.3f perplexity: %.3f speed: %.0f wps" %
              (step * 1.0 / epoch_size, np.exp(costs / iters),
               iters * m.batch_size / (time.time() - start_time)))

        if iters > max_steps:
            break

    return (costs / iters)


def get_config(conf):
    if conf == "small":
        return cf.SmallConfig
    elif conf == "medium":
        return cf.MediumConfig
    elif conf == "large":
        return cf.LargeConfig
    elif conf == "titanx":
        return cf.TitanXConfig
    else:
        raise ValueError('did not enter acceptable model config:', conf)


def save_load(save_path, sess):
    if not os.path.exists(save_path):
        with open(save_path, "wb") as file:

            variables = tf.trainable_variables()
            values = sess.run(variables)
            pickle.dump({var.name: val for var, val in zip(variables, values)}, file)
    else:
        v_dic = {v.name: v for v in tf.trainable_variables()}

        for key, value in pickle.load(open(save_path, "rb")).items():
            sess.run(tf.assign(v_dic[key], value))


def load_np(save_path):
    if not os.path.exists(save_path):
        raise Exception("No saved weights at that location")
    else:
        v_dict = pickle.load(open(save_path, "wb"))
        for key in v_dict.keys():
            print("Key name: " + key)

    return v_dict


if __name__ == '__main__':
    from sys import argv

    exit(save_load(argv))
