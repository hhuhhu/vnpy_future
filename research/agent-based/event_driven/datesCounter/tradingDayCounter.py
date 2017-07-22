# coding=utf-8

import os
from bisect import bisect
import tushare as ts
from datetime import datetime, timedelta
import numpy as np
from collections import Iterable


class WorkDays:
    """
    计算两个日期之间的工作日数，非天数。
    """

    def __init__(self, start_date, end_date):
        """
        days_off:休息日,默认周六日, 以0(星期一)开始,到6(星期天)结束, 传入tupple
        没有包含法定节假日,
        """
        self.start_date = start_date
        self.end_date = end_date
        if self.start_date > self.end_date:
            raise ValueError('start_date must before end date.')

    def work_days(self):
        """
        实现工作日的 iter, 从start_date 到 end_date , 如果在工作日内,yield 日期
        """
        # 还没排除法定节假日
        tag_date = self.start_date
        while True:
            if tag_date > self.end_date:
                break
            # 非周末
            if tag_date.weekday() <= 4:
                yield tag_date
            tag_date += timedelta(days=1)

    def days_count(self):
        """
        工作日统计,返回数字
        """
        return len(list(self.work_days()))


def workdays_count(days, start=datetime(1991, 1, 1)):
    """
    返回days和start之间的工作日数。
    Parameters
    ----------
    days：一个Iterable的datetime，或者一个datetime
    start：起始日期，datetime类型

    Returns
    -------
    ndarray的int或者int，表示工作日数。
    """
    if isinstance(days, Iterable):
        return np.asarray([workdays_count(x, start) for x in days])
    if not isinstance(start, datetime):
        raise TypeError('start should be a datetime type.')
    if not isinstance(days, Iterable):
        if not isinstance(days, datetime):
            raise TypeError('days should be a datetime type.')
    wd = WorkDays(start, days)
    return wd.days_count()


def load_and_refresh_tradedays():
    """
    加载交易日数据文件，如果当前时间不在数据文件中，则更新交易日数据文件。
    Returns
    -------
    返回ndarray类型的所有交易日数据。
    """
    filepath = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(filepath, 'tradeday.txt'), 'r') as fhandler:
        dayseries = np.loadtxt(fhandler, dtype=np.string_, delimiter=',')
    dayseries = dayseries.astype(str)

    if (datetime.now() - datetime.strptime(dayseries[-1], '%Y%m%d')).days != 0:
        try:
            df = ts.get_h_data('000001',
                               start=datetime.strftime(datetime.strptime(dayseries[-1], '%Y%m%d') + timedelta(days=1), '%Y-%m-%d'),
                               end=datetime.strftime(datetime.now(), '%Y-%m-%d'), index=True)
        except IOError as e:
            print e
            df = None

        if df is None:
            return dayseries
        ar = np.asarray([datetime.strftime(x.to_pydatetime(), '%Y%m%d') for x in df.index][::-1])
        dayseries = np.concatenate((dayseries, ar))

        # 写文件用的ascii码，而python3 默认str都是unicode，之后转换成了b'...'类型，故mode需加上'b'。
        # 详情参考源码。
        with open(os.path.join(filepath, 'tradeday.txt'), mode='ab') as fhandler:
            np.savetxt(fhandler, ar, delimiter=',', fmt='%s')

    return dayseries


def __tradeday_count__(start, end, tradedaylist=None):
    """
    返回start和end之间的交易日天数。
    start和end都为19900101这样的8位字符。
    """
    if (datetime.strptime(end, '%Y%m%d') - datetime.strptime(start, '%Y%m%d')).days < 0:
        raise ValueError('end should be after start.')
    if tradedaylist is None:
        tradedaylist = load_and_refresh_tradedays()
    if (datetime.strptime(tradedaylist[-1], '%Y%m%d') - datetime.strptime(end, '%Y%m%d')).days < 0 or\
        (datetime.strptime(start, '%Y%m%d') - datetime.strptime(tradedaylist[0], '%Y%m%d')).days < 0:
        raise ValueError('start or end time out of bound.')
    start = int(start)
    end = int(end)
    tradedaylist = list(map(lambda x: int(x), tradedaylist))
    return bisect(tradedaylist, end) - bisect(tradedaylist, start)


def tradedays_count(start, end):
    """
    返回end和start之间的交易日数。
    Parameters
    ----------
    end:一个Iterable的datetime，或者一个datetime
    start:一个Iterable的datetime，或者一个datetime

    Returns
    -------
    ndarray对象，其元素为end和start间交易日的个数。
    若start为一个元素，end为一个元素，返回两个元素间的交易天数。
    若start为一个元素，end为可迭代对象，返回ndarray，保存各个end与start间交易日个数。
    若start为可迭代对象，end为一个元素，返回ndarray，保存end与各个start间交易日个数。
    若start和end都为可迭代对象，则其个数必须一致，返回ndarray，保存各个end与start间交易日个数。
    """
    if (not isinstance(start, Iterable) and not isinstance(start, datetime))\
        or (not isinstance(end, Iterable) and not isinstance(end, datetime)):
        raise TypeError('start or end should be iterable or datetime object.')

    if isinstance(start, Iterable) and not isinstance(start, str):
        start = np.asarray([datetime.strftime(d, '%Y%m%d') for d in start])
    if isinstance(end, Iterable) and not isinstance(end, str):
        end = np.asarray([datetime.strftime(d, '%Y%m%d') for d in end])
    if isinstance(start, datetime):
        start = datetime.strftime(start, '%Y%m%d')
    if isinstance(end, datetime):
        end = datetime.strftime(end, '%Y%m%d')

    if isinstance(start, str) and isinstance(end, str):
        return __tradeday_count__(start, end)

    dayseries = load_and_refresh_tradedays()
    if isinstance(start, Iterable) and not isinstance(start, str) and isinstance(end, Iterable) and not isinstance(end, str):
        if len(start) != len(end):
            raise ValueError('start dimension should be the same as the end.')
        return np.asarray([__tradeday_count__(tp[0], tp[1], dayseries) for tp in zip(start, end)])
    if isinstance(start, Iterable) and not isinstance(start, str) and isinstance(end, str):
        return np.asarray([__tradeday_count__(s, end, dayseries) for s in start])
    if isinstance(start, str) and isinstance(end, Iterable) and not isinstance(end, str):
        return np.asarray([__tradeday_count__(start, e, dayseries) for e in end])

    raise TypeError('input unknown type error.')


if __name__ == '__main__':
    a = [datetime(1992, 2, 5), datetime(1992, 3, 1), datetime(1993, 2, 9)]
    b = [datetime(1998, 2, 1), datetime(1992, 3, 2), datetime(2000, 1, 1)]
    ss = datetime(1991, 2, 1)
    ee = datetime(2000, 12, 12)
    print(tradedays_count(a, b))
