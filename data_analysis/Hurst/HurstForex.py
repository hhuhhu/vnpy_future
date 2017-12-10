# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from Hurst import Hurst


# 取data文件中的收盘价，数据个数不超过max_num以优化速度(从后往前，保证使用最新数据)
def get_close_data(f_name, folder_path='./data/', max_num=200):
    data = pd.read_csv(folder_path + f_name)
    data_close = data.values[(data.shape[0]-min(max_num, data.shape[0])):data.shape[0], 5]
    print('data shape:', len(data_close))
    data_change = [i for i in data_close]
    return data_change


# 根据cal_type，计算不同类别的外汇hurst指数，
# 1：比较不同周期， 2：计算不同品种， 3：同一个周期下，减少运算量
# 4:计算随时间的变化，长度固定 5：计算随时间的变化，长度不断增长
def foreign_exchange_hurst(cal_type=1):
    if cal_type == 1:  # 计算xauusd的不同周期的Hurst指数
        file_name = ['XAUUSD_'+i+'.csv' for i in ['M', 'W', 'D', 'H4', 'M30']]
        # file_name = ['XAUUSD_' + i for i in ['M', 'W']]
        h_res = [Hurst(Hurst.cal_standard(get_close_data(f_name=i))) for i in file_name]
        plot_lots_hurst(h_res, file_name, 'Hurst in different period')
        plt.show()
    if cal_type == 2:  # 计算不同品种
        variety = [i+'_M30.csv' for i in ['AUDUSD', 'EURUSD', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY', 'XAUUSD']]
        h_res = [Hurst(Hurst.cal_standard(get_close_data(f_name=i)), s_step=2) for i in variety]
        plot_lots_hurst(h_res, variety, 'Hurst in different variety')
        plt.show()
    if cal_type == 3:  # 计算同一个周期，减少运算量(s_step增加)的影响
        s_step = [1, 2, 4, 8, 16, 32]
        # s_step = [1, 32]
        h_res = [Hurst(Hurst.cal_standard(get_close_data(f_name='XAUUSD_M30.csv', max_num=500)), s_step=i) for i in s_step]
        plot_lots_hurst(h_res, s_step, 'Hurst in different s-step')
        plt.show()
    if cal_type == 4:  # 计算hurst随时间的变化，周期固定
        tau = 100
        data_all = get_close_data(f_name='XAUUSD_M30.csv', max_num=1000)
        data_standard = Hurst.cal_standard(data_all)
        h_res = [Hurst(data_standard[i:i+tau], s_step=2).hurst for i in range(0, len(data_standard)-tau)]
        h_res_smooth = ndimage.gaussian_filter(h_res, sigma=10)

        h_res = np.hstack((np.repeat(np.nan, tau), h_res))
        h_res_smooth = np.hstack((np.repeat(np.nan, tau), h_res_smooth))

        fig = plt.figure(figsize=(20, 15))
        ax1 = fig.add_subplot(111)
        ax1.plot(data_all, 'r-', label='XAUUSD_M30')
        plt.ylabel("XAUUSD_M30")
        plt.xlabel("Time")
        plt.legend(loc=2)
        ax2 = ax1.twinx()
        ax2.plot(h_res, 'b.', label='Hurst')
        ax2.plot(h_res_smooth, 'b-', label='Hurst after smooth')
        plt.legend(loc=1)
        plt.title("Hurst change with the time")
        plt.ylabel("Hurst")
        plt.show()
    if cal_type == 5:  # 计算不同数据长度的影响，同一个数据，截取不同长度
        tau_min = 50
        data_all = get_close_data(f_name='XAUUSD_M30.csv', max_num=1000)
        data_standard = Hurst.cal_standard(data_all)
        # print([data_standard[0:i][::-1] for i in range(tau_min, len(data_standard))])
        # h_res = [Hurst(data_standard[0:i][::-1], s_step=2).hurst for i in range(tau_min, len(data_standard), int((len(data_standard)-tau_min)/20))]
        h_res = [Hurst(data_standard[0:i][::-1], s_step=2).hurst for i in range(tau_min, len(data_standard))]
        plt.plot(np.hstack((np.repeat(np.nan, tau_min), h_res)))
        plt.xlabel('time length')
        plt.ylabel('hurst')
        plt.title('different time length of Hurst')
        plt.xlim([0, 1000])
        plt.show()
    return


def plot_lots_hurst(h_res, labels, title):
    for i in range(len(h_res)):
        log_s = h_res[i].get_log_s()
        log_fs = h_res[i].get_log_fs()
        plt.plot(log_s, log_fs, '.', label=str(labels[i]) + ", Hurst:" + str(round(h_res[i].hurst, 2)))
        model = h_res[i].model_hurst
        plt.plot(log_s, model(log_s), 'k--')
    plt.legend(loc=2)
    plt.title(title)
    plt.xlabel('log(s)')
    plt.ylabel('log(fs)')

if __name__ == '__main__':
    foreign_exchange_hurst(cal_type=1)





