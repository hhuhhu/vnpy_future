# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: dict_learn.py
@time: 2017/9/7 11:54
"""
class Symbol:
    def __init__(self, a ,b):
        self.a=a
        self.b=b

    def __str__(self):
        return str(self.__dict__)

symbol={'slippage': 1, 'size_value': 10, 'symbol_type': '',
                                     'open_cost_rate': 0.0001, 'close_cost_rate': 0.0001}
a = Symbol(a=1,b=2)
# print(a.a)

if __name__ == '__main__':
    a = {"a":1,"b":2}
    print(a["a"])