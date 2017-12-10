# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: i1601_analysis.py
@time: 2017/6/23 12:47
"""
from datetime import datetime

from data_handle.mongo import mongo_connect

price = 431
priceTick = 0.5

newPrice = round(price/priceTick, 0) * priceTick
print(newPrice)
