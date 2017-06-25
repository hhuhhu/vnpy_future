# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: kalman_learn.py
@time: 2017/5/26 16:24
"""
# Import libraries
from matplotlib import pyplot
from pykalman import KalmanFilter
import numpy
import scipy
import time
import datetime
import pandas_datareader.data as web
# Initialize a Kalman Filter.
# Using kf to filter does not change the values of kf, so we don't need to ever reinitialize it.
kf = KalmanFilter(transition_matrices=[1],
                  observation_matrices=[1],
                  initial_state_mean=0,
                  initial_state_covariance=1,
                  observation_covariance=1,
                  transition_covariance=.01)


# helper functions
# for converting dates to a plottable form
def convert_date(mydate):
    # return time.mktime(datetime.datetime.strptime(mydate, "%Y-%m-%d").timetuple())
    return datetime.datetime.strptime(mydate, "%Y-%m-%d")


def get_pricing(equity_name,
                 start_date='',
                 end_date='',
                 # fields=['close_price'],
                 frequency='daily'):
    data = web.DataReader(equity_name, data_source='google', start=start_date, end=end_date)
    # print("data: ", data)
    return data


# for grabbing dates and prices for a relevant equity
def get_data(equity_name, trading_start, trading_end='2015-07-20'):
    # using today as a default arg.
    stock_data = get_pricing(equity_name,
                             start_date=trading_start,
                             end_date=trading_end,
                             # fields=['close_price'],
                             frequency='daily')
    stock_data['date'] = stock_data.index
    # drop nans. For whatever reason, nans were causing the kf to return a nan array.
    stock_data = stock_data.dropna()
    # the dates are just those on which the prices were recorded
    dates = stock_data['date']
    dates = [convert_date(str(x)[:10]) for x in dates]
    prices = stock_data['Close']
    return dates, prices
# TSLA started trading on Jun-29-2010.
dates_tsla, scores_tsla = get_data('TSLA', '2010-06-29')
# Apply Kalman filter to get a rolling average
scores_tsla_means, _ = kf.filter(scores_tsla.values)
# Use a scatterplot instead of a line plot because a line plot would be far too noisy.
pyplot.scatter(dates_tsla,scores_tsla,c='gray',label='TSLA Price')
pyplot.plot(dates_tsla,scores_tsla_means, c='red', label='TSLA MA')
pyplot.ylabel('TSLA Price')
pyplot.xlabel('Date')
pyplot.ylim([0,300])
pyplot.legend(loc=2)
pyplot.show()
# Get UGA data and apply the Kalman filter to get a moving average.
dates_uga, scores_uga = get_data('UGA', '2013-06-01')
scores_uga_means, _ = kf.filter(scores_uga.values)

# Get TSLA for June 2013 onwards, and apply the Kalman Filter.
dates_tsla2, scores_tsla2 = get_data('TSLA', '2013-06-01')
scores_tsla_means2, _ = kf.filter(scores_tsla2.values)
_, ax1 = pyplot.subplots()
ax1.plot(dates_tsla2,scores_tsla_means2, c='red', label='TSLA MA')
pyplot.xlabel('Date')
pyplot.ylabel('TSLA Price MA')
pyplot.legend(loc=2)
# twinx allows us to use the same plot
ax2 = ax1.twinx()
ax2.plot(dates_uga, scores_uga_means, c='black', label='UGA Price MA')
pyplot.ylabel('UGA Price MA')
pyplot.legend(loc=3)
pyplot.show()


def find_offset(ts1, ts2, window):
    """ Finds the offset between two equal-length timeseries that maximizies correlation. 
        Window is # of days by which we want to left- or right-shift.
        N.B. You'll have to adjust the function for negative correlations."""
    l = len(ts1)
    if l != len(ts2):
        raise Exception("Error! Timeseries lengths not equal!")
    max_i_spearman = -1000
    max_spearman = -1000
    spear_offsets = []

    # we try all possible offsets from -window to +window.
    # we record the spearman correlation for each offset.
    for i in range(window, 0, -1):
        series1 = ts1[i:]
        series2 = ts2[:l - i]
        # spearmanr is a correlation test
        spear = scipy.stats.spearmanr(series1, series2)[0]
        spear_offsets.append(spear)

        if spear > max_spearman:
            # update best correlation
            max_spearman = spear
            max_i_spearman = -i

    for i in range(0, window):
        series1 = ts1[:l - i]
        series2 = ts2[i:]
        spear = scipy.stats.spearmanr(series1, series2)[0]
        spear_offsets.append(spear)
        if spear > max_spearman:
            max_spearman = spear
            max_i_spearman = i

    print("Max Spearman:", max_spearman, " At offset: ", max_i_spearman)
    pyplot.plot(range(-window, window), spear_offsets, c='green', label='Spearman Correlation')
    pyplot.xlabel('Offset Size (Number of Business Days)')
    pyplot.ylabel('Spearman Correlation')
    pyplot.legend(loc=3)
    pyplot.show()
print("Kalman-Filtered Smoothed Data")
find_offset(scores_tsla_means2,scores_uga_means,200)
print("Raw Data")
find_offset(scores_tsla2,scores_uga,150)
# plotting formalities for 126-day offset

d = 126
cseries1 = scores_tsla_means2[d:]
cseries2 = scores_uga_means[:len(scores_tsla_means2)-d]
r = range(len(cseries1))

_, ax1 = pyplot.subplots()
ax1.plot(r, cseries1, c='red', label='TSLA MA')
pyplot.xlabel('Number of Business Days Elapsed')
pyplot.ylabel('TSLA Price MA')
pyplot.legend(loc=2)
ax2 = ax1.twinx()
ax2.plot(r, cseries2, c='black', label='UGA Price MA')
pyplot.ylabel('UGA Price MA')
pyplot.legend(loc=4)
pyplot.title("-126 Day Offset")
pyplot.show()

# plotting for 50-day offset

d = 50
cseries1 = scores_tsla_means2[d:]
cseries2 = scores_uga_means[:len(scores_tsla_means2)-d]
r = range(len(cseries1))

_, ax1 = pyplot.subplots()
ax1.plot(r, cseries1, c='red', label='TSLA MA')
pyplot.xlabel('Number of Business Days Elapsed')
pyplot.ylabel('TSLA Price MA')
pyplot.legend(loc=2)
ax2 = ax1.twinx()
ax2.plot(r, cseries2, c='black', label='UGA Price MA')
pyplot.ylabel('UGA Price MA')
pyplot.legend(loc=4)
pyplot.title("-50 Day Offset")
pyplot.show()
print(scipy.stats.spearmanr(scores_tsla_means2[d:][250:], scores_uga_means[:len(scores_tsla_means2)-d][250:]))