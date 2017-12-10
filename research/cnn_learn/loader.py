# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: loader.py
@time: 2017/12/9 15:12
"""
import os
import functools
import math
import csv

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.contrib.learn.python.learn.datasets import base

np.set_printoptions(threshold=np.nan)


class Dataset:
    def __init__(self,
                 images,
                 labels,
                 dtype=dtypes.float32,
                 seed=None):
        self.check_data(images, labels)
        seed1, seed2 = random_seed.get_seed(seed)
        self.__images = images
        self.__labels = labels
        self.__epochs_completed = 0
        self.__index_in_epoch = 0
        self.__total_batches = images.shape[0]

    def check_data(self, images, labels):
        assert images.shapes[0] == labels.shape[0], 'images.shape: {};labels.shape: {}'.format(images.shape,
                                                                                               labels.shape)

    def next_batch(self, batch_size, shuffle=True):
        start = self.__index_in_epoch


