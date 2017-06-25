# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: data_insert.py
@time: 2017/6/23 11:42
"""
import os
import json
from datetime import datetime
import csv

import pandas as pd
import numpy as np
import pymongo

from data_handle.mongo import mongo_connect
from core.ctaBase import CtaBarData
from core.vtConstant import EMPTY_INT


class BarData(CtaBarData):

    def __init__(self):
        super(BarData, self).__init__()
        self.turnover = EMPTY_INT   # 成交金额
        self.matchItem = EMPTY_INT  # 成交笔数


def data_insert():
    db = mongo_connect()

    future_db = db.future_1min
    par_path = 'D:/FutureData/bar'
    file_name = 'rb1601-SHF.csv'
    file_path = os.path.join(par_path, file_name)
    data = pd.read_csv(file_path)
    data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume', 'Close': 'close',
                         'Time': 'time', 'Interest': 'openInterest', 'Turover': 'turnover', 'MatchItem': 'matchItem',
                         'WindCode': 'windCode', 'Date': 'date'}, inplace=True)
    date_to_str = lambda x: str(x)[0:4] + '-' + str(x)[4:6] + '-' + str(x)[6:8]
    len_to_nine = lambda x: str(x).zfill(9)
    time_to_str = lambda x: str(x)[0:2] + ':' + str(x)[2:4] + ':' + str(x)[4:6]
    str_to_date = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    data['date'] = data['date'].apply(date_to_str)
    data['time'] = data['time'].apply(len_to_nine)
    data['time'] = data['time'].apply(time_to_str)
    data['datetime'] = data['date'] + ' ' + data['time']
    data['datetime'] = data['datetime'].apply(str_to_date)
    a = pd.to_datetime(data['datetime'])
    # data['datetime'] = data['datetime'].apply(date64_datetime)
    # print(data['datetime'])
    # a = data['datetime'][0]
    data['symbol'] = 'I1601'
    data['vtSymbol'] = 'I1601'
    data.drop(labels=['windCode', 'date'], inplace=True, axis=1)
    # future_db.I1601.insert(json.loads(data.to_json(orient='records', date_format='iso')))
    reader = csv.DictReader(data)
    for content in reader:
        print(content)
    print(reader)


def data_insert2(file_name, symbol):

    db = mongo_connect()
    future_db = db.future_1min
    collection = future_db.RB1601
    collection.ensure_index([('datetime', pymongo.ASCENDING)], unique=True)
    par_path = 'D:/FutureData/bar'
    file_path = os.path.join(par_path, file_name)
    with open(file_path) as file:

        reader = csv.DictReader(file)
        for d in reader:
            bar = BarData()
            bar.vtSymbol = symbol
            bar.symbol = symbol
            bar.open = float(d['Open'])
            bar.high = float(d['High'])
            bar.low = float(d['Low'])
            bar.close = float(d['Close'])
            bar.date = str(d['Date'])[0:4] + '-' + str(d['Date'])[4:6] + '-' + str(d['Date'])[6:8]
            time = str(d['Time']).zfill(9)

            bar.time = str(time)[0:2] + ':' + str(time)[2:4] + ':' + str(time)[4:6]
            bar.datetime = datetime.strptime(bar.date + ' ' + bar.time, '%Y-%m-%d %H:%M:%S')
            bar.volume = d['Volume']
            bar.openInterest = d['Interest']
            bar.turnover = d['Turover']
            bar.matchItem = d['MatchItem']
            flt = {'datetime': bar.datetime}

            collection.update_one(flt, {'$set': bar.__dict__}, upsert=True)
if __name__ == '__main__':
    file_name = 'rb1601-SHF.csv'
    symbol = 'RB1601'
    data_insert2(file_name, symbol)
    print('done')

