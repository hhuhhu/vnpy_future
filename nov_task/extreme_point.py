# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: extreme_point.py
@time: 2017/11/28 14:45
"""
import math
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nov_task.data_get import data_get


class ExtremePoint:

    def __init__(self, file_path='F:/NovTask/XAUUSDh4.csv', ext_pattern=2, short_ma_period=24, long_ma_period=90):
        self.data = data_get(file_path)
        self.data['short_ma'] = self.data['close'].rolling(short_ma_period).mean()
        self.data['long_ma'] = self.data['close'].rolling(long_ma_period).mean()
        self.data.dropna(inplace=True)
        self.ext_pattern = ext_pattern
        self.short_ma_buffer = pd.DataFrame()
        self.long_ma_buffer = pd.DataFrame()
        self.type = 0  # 短上为1，长上为-1；
        self.ext_pattern = ext_pattern
        self.count = 0
        self.m_signal = 0   # 1做多,-1做空
        self.latest_index = datetime.now()

    def extreme_point_location(self, ):
        res = 0
        for i in range(1, self.count+1):
            var = self.short_ma_buffer[-i] - self.short_ma_buffer[-(i+1)]
            if res > 0:
                if var < 0:
                    break
                res += 1
                continue
            if res < 0:
                if var > 0:
                    break
                res -= 1
            if var > 0:
                res += 1
            if var < 0:
                res -= 1
        return res

    def extreme_point_v2(self, idx):
        res = self.extreme_point_location()
        if math.fabs(res) > idx or res == 0:
            return False
        else:
            if res>0:
                self.m_signal = 1
            else:
                self.m_signal = -1
            return True

    def extreme_point_v1(self, short_ma, long_ma):
        pass

    def search_extreme(self):
        if self.type == 0:
            return
        idx = 0
        # 记录长短均线交叉点位置
        if self.count<2:return
        temp = self.short_ma_buffer[-(idx+1)] - self.long_ma_buffer[-(idx+1)]
        while self.type * temp >= 0:
            idx += 1
        if idx+1>=self.count:
            return
        else:
            # 极点1
            if self.ext_pattern == 1:
                self.extreme_point_v1(idx)
            # 极点2
            elif self.ext_pattern == 2:
                self.extreme_point_v2(idx)
            else:
                raise ValueError('sorry but you have got a wrong ext_pattern, it should be 1 or 2')

    def on_bar(self):
        for index in self.data.index:
            self.count +=1
            short_ma = self.data.loc[index]['short_ma']
            long_ma = self.data.loc[index]['long_ma']
            self.short_ma_buffer = self.data.loc[:index]['short_ma']
            self.long_ma_buffer = self.data.loc[:index]['long_ma']
            if short_ma - long_ma > 0:
                self.type = 1
            elif short_ma - long_ma < 0:
                self.type = -1
            else:
                self.type = 1
            self.search_extreme()

    def ma_plot(self):
        self.data['short_ma'].plot()
        self.data['long_ma'].plot()
        plt.title('xauusd_short_long_ma')
        plt.ylabel('Price')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    exp = ExtremePoint()
    exp.on_bar()
