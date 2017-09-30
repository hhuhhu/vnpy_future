# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: data_analysis.py
@time: 2017/8/23 17:04
"""
import os
import copy
import multiprocessing
import threading

import pandas as pd
import pymongo

from data_handle.data_insert import data_insert


class DataGenerate:
    def __init__(self, path):
        self.wind_path = 'KLine//SHF'
        self.path = os.path.join(path, self.wind_path)
        self.target_path = 'E://FutureData'

    def data_get(self):
        sub_paths = os.listdir(self.path)
        file_list = os.listdir(os.path.join(self.path, sub_paths[0]))
        symbols = [file_name.split('.')[0] for file_name in file_list]
        temp_data = []
        temp_dict = {}
        data = {}
        for symbol in symbols:
            for sub_path in sub_paths:
                _path = os.path.join(self.path, sub_path, '{}.csv'.format(symbol))
                # print('_path: {}'.format(_path))
                if os.path.exists(_path):
                    data = pd.read_csv(_path, encoding='gbk')
                    temp_data.append(copy.deepcopy(data))
            temp_data = pd.concat(temp_data, ignore_index=True)
            temp_data['close'] = temp_data['close']/10000
            temp_data['open'] = temp_data['open']/10000
            temp_data['high'] = temp_data['high']/10000
            temp_data['low'] = temp_data['low']/10000
            name_dict = {}
            [name_dict.update({name: name.capitalize()}) for name in temp_data.columns]
            name_dict['volumw'] = 'Volume'
            name_dict['match_items'] = 'MatchItem'
            temp_data.rename(columns=name_dict, inplace=True)
            target_path = os.path.join(self.target_path, '{}.csv'.format(symbol))
            temp_data.to_csv(target_path)
            data_insert(target_path, symbol)

        return data


class DataAnalysis:
    def __init__(self):
        pass

    def data_analysis(self):
        pass
if __name__ == '__main__':
    path = 'C://Users//gxy//Desktop//FutureData//rb1710'
    data_generate = DataGenerate(path)
    data_generate.data_get()
