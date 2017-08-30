# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: mongo.py
@time: 2017/6/9 10:20
"""
import os
import json

from pymongo import MongoClient


def mongo_connect():

    file_name = 'config.json'
    path = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(path, file_name)
    with open(filename) as f:
        setting = json.load(f)
    host = setting['mongoHost']
    port = setting['mongoPort']
    client = MongoClient(host, port)
    return client
