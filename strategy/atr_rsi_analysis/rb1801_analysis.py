# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: rb1801_analysis.py
@time: 2017/9/6 10:07
"""
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from core.ctaBase import *
from data_handle.data_load import data_load

plt.style.use('ggplot')


class DataAnalysis:
    """
    分析数据，提取特征
    """
    def __init__(self, start_date, end_date, db_name, symbol):
        self.data = data_load(db_name=db_name, symbol=symbol, start_date=start_date, end_date=end_date)

    def amplitude(self):
        """
        特定周期内的振幅统计及绘图：振幅=(high-low)/low
        :return: 
        """
        amplitude = (self.data['high'] - self.data['low'])/self.data['low']
        amplitude.plot()
        plt.show()

    def fluctuation_culculate(self):
        """
        及于价差的频次统计及绘图
        :return: 
        """
        spread = self.data['high'] - self.data['low']
        description = spread.describe()
        self.data['spread'] = spread
        self.data.groupby('spread').size
        count = spread.count()
        spread.plot()
        plt.show()


if __name__ == '__main__':
    start_date = '20160701'
    end_date = '20180823'
    db_name = FUTURE_1MIN
    symbol = 'RB1801'
    data_analysis = DataAnalysis(start_date, end_date, db_name, symbol)
    # data_analysis.amplitude()
    data_analysis.fluctuation_culculate()