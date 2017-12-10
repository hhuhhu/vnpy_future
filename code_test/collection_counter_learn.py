# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: collection_counter_learn.py
@time: 2017/7/20 9:16
"""
import collections
a = ['a', 'a', 'b']
words = collections.Counter(a)
m = words.keys()
n = {key:(ix+1) for ix, key in enumerate(m)}
print(words)
print(m)
print(n)