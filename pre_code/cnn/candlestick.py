#! /usr/bin/env python

import os

import pandas as pd
import plotly


class Candlestick(object):

    def __init__(self):
        self.table = []
        self.openPrice = []
        self.highPrice = []
        self.lowPrice = []
        self.closePrice = []
        self.date = []
        self.par_path = os.path.dirname(__file__)

    def candle_stick(self, filename):
        filename = os.path.join(self.par_path, 'data/daily', filename)
        header = ['date', 'flag', 'open', 'high', 'low', 'close', 'volume']
        self.table = pd.read_csv(filename, names=header)
        self.date = self.table['date']
        self.openPrice = self.table['open']
        self.highPrice = self.table['high']
        self.lowPrice = self.table['low']
        self.closePrice = self.table['close']

        fig = plotly.tools.FigureFactory.create_candlestick(self.openPrice, self.highPrice, self.lowPrice,
                                    self.closePrice, dates=self.date)
        plotly.plotly.image.ishow(fig)
        plotly.plotly.plot(fig, filename='aapl-candlestick', fileopt='new', validate=False)


if __name__ == '__main__':
    candlestick = Candlestick()
    print(candlestick.par_path)
    file_name = 'table_a.csv'
    candlestick.candle_stick(file_name)
