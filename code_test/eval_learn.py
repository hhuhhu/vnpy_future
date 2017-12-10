# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: eval_learn.py
@time: 2017/8/27 15:25
"""
a = ['ab', 'ac']
d = {}
for ca in a:
    d.update({ca, ca.capitalize()})
print(d)