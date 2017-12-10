# coding=utf-8

#############################################
# 过滤爬取的新闻事件文件，若该股相关的新闻未#
#         导致股价爆发，则删除该文件        #
#############################################

import numpy as np
import pandas as pd
import os
import tushare as ts
from datetime import datetime, timedelta

dest_dir = 'E:/stock_data/stock_news/'

def is_bursted(stock, d):

    startd = datetime.strptime(d, '%Y-%m-%d')
    endd = startd + timedelta(30)
    enddstr = endd.strftime('%Y-%m-%d')
    df = ts.get_hist_data(stock, d, enddstr)
    df.sort_index(inplace=True)

    # 如果接下来两天涨停则算爆发
    if len(df) >= 3:
        if df.ix[1, 'p_change'] >= 9.5 and df.ix[2, 'p_change'] >= 9.5:
            return True

    # 如果未来一个月内上涨了30%也算爆发
    if len(df) > 4:
        if df.ix[4, 'ma5']/df.ix[0, 'ma5'] - 1. > 0.3:
            return True
    if len(df) > 9:
        if df.ix[9, 'ma5']/df.ix[0, 'ma5'] - 1. > 0.3:
            return True
    if len(df) >14:
        if df.ix[14,'ma5']/df.ix[0, 'ma5'] - 1. > 0.3:
            return True
    if len(df) >19:
        if df.ix[19,'ma5']/df.ix[0, 'ma5'] - 1. > 0.3:
            return True
    return False


def run():
    burst_count = 0
    walker = os.walk(dest_dir)
    for parent, dirs, files in walker:
        for f in files:
            fpath = os.path.join(parent, f)
            df = pd.read_csv(fpath, encoding='gbk')
            stock, surfix = f.split('.')
            dates = np.asarray(df.ix[:, 'date'])
            remain = False
            for d in dates:
                ib = is_bursted(stock, d)
                if ib:
                    remain = True
                    burst_count += 1
                    print(stock, 'has a burst')
                    break
            if not remain:
                print(stock, ' has no burst. Deleting...')
                os.remove(fpath)
                print(stock, ' deleted')
    print('total burst stocks: ', burst_count)


if __name__ == '__main__':
    run()
