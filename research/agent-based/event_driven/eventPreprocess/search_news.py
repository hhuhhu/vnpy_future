# coding=utf-8
#############################################
# 爬取重大事件，用于后续的事件驱动模型开发。#
#############################################
import tushare as ts
from datetime import datetime, timedelta
import pickle
import pandas as pd
import threading

to_file = 'E:/stock_data/stock_news/'
key_word = '上市|并购|重组|突破|订单'
key_word2 = '上市公告书|券商报告'
thread_num = 5
threads = []  # 线程篮子


def get_stock_list():
    f = open('E:/stock_data/stock_name_list.pkl', 'rb')
    pk = pickle.load(f)
    total_number = len(pk)
    return pk, total_number

def get_stock_list():
    f =
stock_list, total_number = get_stock_list()


class ospider(threading.Thread):
    def __init__(self, sl, index):
        threading.Thread.__init__(self)
        self.stocklist = sl
        self.stop = False
        self.ix = index

    def run(self):
        while not self.stop:
            for i, stock in enumerate(self.stocklist):
                print('processing: {0}.   By thread:{1}'.format(stock, self.ix))
                t = datetime(2015, 6, 20)
                t_delta = datetime.now() - t
                dframe = pd.DataFrame()
                while t_delta.days > 0:
                    t = t + timedelta(1)
                    t_delta = datetime.now() - t
                    subdf = ts.get_notices(stock, t)
                    if not (subdf is None or subdf is [None]):
                        subdf = subdf.ix[subdf.ix[:, 0].str.contains(key_word)]
                        subdf = subdf.ix[subdf.ix[:, 1].str.contains(key_word2)]
                    dframe = dframe.append(subdf, ignore_index=True)
                if len(dframe) == 0:
                    continue
                dframe.to_csv(to_file + stock + '.csv', index=False)

    def stop(self):
        self.stop = True


def getnews():
    stock_list_thread = []
    num = int(total_number / thread_num)
    for i in range(thread_num):
        if i == thread_num - 1:
            stock_list_thread.append(stock_list[i * num:])
        else:
            stock_list_thread.append(stock_list[i * num: (i + 1) * num])
    for i in range(thread_num):
        thr = ospider(stock_list_thread[i], i)
        threads.append(thr)
        thr.start()
    for t in threads:
        t.join()
    return

if __name__=='__main__':
    getnews()
