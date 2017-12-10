# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: order_dict_learn.py
@time: 2017/5/27 15:06
"""
from collections import OrderedDict
a = OrderedDict()
a = {"a": "1", "b": 2, "c": 3, "d": 5}
print(a)
a.items()
print(a.items())
print(type(a))
for key, value in a.iterms():
    print(key, value)