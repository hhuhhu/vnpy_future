# coding=utf-8

from datetime import datetime
import numpy as np
import pandas as pd
import anfis
import membership
from datesCounter.tradingDayCounter import tradedays_count
from functools import partial


def test():

    ts = pd.read_csv('event_driven_model.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8], encoding='gbk')

    # 处理异常值
    ts.fillna(0, inplace=True)

    # 设置日期的起点。
    dcounter = partial(tradedays_count, datetime(2010, 12, 30))
    # 输入时间规则化
    days = [datetime.strptime(str(time), '%Y%m%d') for time in ts['time']]
    days = dcounter(np.asarray(days))
    ts['time'] = days

    # 其他输入变量预处理
    ts['second_exist'] = ts['second_exist'].astype(np.int32)
    ts.ix[ts['second_exist'] == 0, [5, 6, 7]] = 0
    for col in range(ts.shape[1]):
        if col == 0:
            continue
        ts.ix[:, col] = ts.ix[:, col].astype(np.float32)
    ts['gap_time'] /= 360.0  # 归一化
    for col in [0, 1, 2, 3]:
        ts.ix[:, col] = (ts.ix[:, col] - ts.ix[:, col].min()) / (ts.ix[:, col].max() - ts.ix[:, col].min())
    print(ts)

    ts = np.asarray(ts)
    X = ts[:, 0:4]  # 输入变量
    Y = ts[:, 4:8]  # 输出变量

    # 设置隶属度函数
    # mf_time = [['sigmf', {'b': dcounter(datetime(2014, 1, 1)), 'c': -0.05}],
    #            ['gaussmf', {'mean': dcounter(datetime(2015, 1, 1)), 'sigma': 100}],
    #            ['sigmf', {'b': dcounter(datetime(2015, 6, 30)), 'c': 0.2}],
    #            ['gaussmf', {'mean': dcounter(datetime(2015, 11, 15)), 'sigma': 30}]]
    mf_time = [['gaussmf', {'mean': 0.87, 'sigma': 0.1}],
               ['gaussmf', {'mean': 0.92, 'sigma': 0.3}]]

    mf_class = [['gaussmf', {'mean': 0.1, 'sigma': 0.05}],
                ['gaussmf', {'mean': 0.3, 'sigma': 0.1}],
                ['gaussmf', {'mean': 0.6, 'sigma': 0.1}],
                ['gaussmf', {'mean': 0.9, 'sigma': 0.1}]]

    # mf_low = [['gaussmf', {'mean': 1, 'sigma': 5}],
    #           ['gaussmf', {'mean': 10, 'sigma': 10}],
    #           ['sigmf', {'b': 60, 'c': 0.01}]]
    mf_low = [['gaussmf', {'mean': 0.2, 'sigma': 0.1}],
              ['gaussmf', {'mean': 0.8, 'sigma': 0.1}]]

    mf_risepercent = [['gaussmf', {'mean': 0.1, 'sigma': 0.05}],
                      ['gaussmf', {'mean': 0.5, 'sigma': 0.3}]]
    # mf_risepercent = [['gaussmf', {'mean': 0.1, 'sigma': 0.1}],
    #                   ['gaussmf', {'mean': 0.5, 'sigma': 0.5}],
    #                   ['sigmf', {'b': 1, 'c': 0.02}]]

    mf = [mf_time, mf_class, mf_low, mf_risepercent]
    # mf = [mf_time, mf_class]
    and_func = ['mamdani', 'T-S']

    mfc = membership.membershipfunction.MemFuncs(mf)
    anf = anfis.ANFIS(X, Y, mfc, andfunc=and_func[0])
    anf.trainHybridJangOffLine(epochs=15, eta=0.00001)

    print(anf.fittedValues)
    print('mf list:')
    print(anf.memFuncs)
    anf.plotErrors()

if __name__ == '__main__':
    test()
