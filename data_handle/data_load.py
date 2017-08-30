# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: data_load.py
@time: 2017/6/25 15:16
"""
import pandas as pd
import pymongo

from core.vtFunction import loadMongoSetting
from core.ctaBase import *


def data_load(db_name, symbol, start_date, end_date):
    """载入历史数据"""
    host, port, logging = loadMongoSetting()

    db_client = pymongo.MongoClient(host, port)
    collection = db_client[db_name][symbol]

    # 载入初始化需要用的数据
    flt = {'date': {'$gte': start_date,
                        '$lte': end_date}}
    db_cursor = collection.find(flt)
    df = pd.DataFrame(list(db_cursor))
    del df['_id']
    return df

if __name__ == '__main__':
    start_date = '20160701'
    end_date = '20180823'
    db_name = FUTURE_1MIN
    symbol = 'RB1801'
    data = data_load(db_name=db_name, symbol=symbol, start_date=start_date, end_date=end_date)
    print('data: ', data)