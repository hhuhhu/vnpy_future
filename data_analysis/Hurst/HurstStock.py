# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tushare as ts
from scipy import ndimage

from Hurst import Hurst


# 将股票数据下载到本地并保存到当
def download_stock(stock_code='sh', k_type='D', file_path="../data/"):
    stock_data = pd.DataFrame(ts.get_hist_data(stock_code, ktype=k_type))
    stock_data.to_csv(file_path+stock_code+"_"+k_type+".csv")


# 利用tushare在线数据分析股票数据的hurst指数--不同研究周期的影响
def hurst_stock(stock_code='sh', k_type='D', s_step=1):
    stock_data = ts.get_hist_data(stock_code, ktype=k_type)
    stock_data_close = stock_data.close
    stock_data_arr = stock_data_close.values
    stock_data_arr1 = stock_data_arr.flat[:]
    data_standard = Hurst.cal_standard(stock_data_arr1)
    result = Hurst(xts=data_standard, s_step=s_step)
    return result


# Hurst指数变化:1.不同分析周期
def hurst_change(change_type=1):
    if change_type == 1:  # 不同周期的影响
        tau_value = ['5', '15', '30', '60', 'D', 'W', 'M']
        # tau_value = ['5', '15']
        h_res = [hurst_stock(k_type=i) for i in tau_value]
        print('test')
        for i in range(len(h_res)):
            log_s = h_res[i].get_log_s()
            log_fs = h_res[i].get_log_fs()
            plt.plot(log_s, log_fs, '.', label=str(tau_value[i])+", Hurst:"+str(round(h_res[i].hurst, 2)))
            model = h_res[i].model_hurst
            plt.plot(log_s, model(log_s), 'k--')
        plt.legend(loc=2)
        plt.title('Hurst in different period')
        plt.xlabel('log(s)')
        plt.ylabel('log(fs)')
        plt.show()
    if change_type == 2:  # 不同品种的影响
        stock_code = ['sh', '600122', '600291', '600292', '600435']
        h_res = [hurst_stock(stock_code=i) for i in stock_code]
        for i in range(len(h_res)):
            log_s = h_res[i].get_log_s()
            log_fs = h_res[i].get_log_fs()
            plt.plot(log_s, log_fs, '.', label=str(stock_code[i]) + ", Hurst:" + str(round(h_res[i].hurst, 2)))
            model = h_res[i].model_hurst
            plt.plot(log_s, model(log_s), 'k--')
        plt.legend(loc=2)
        plt.title('Hurst in different stocks')
        plt.xlabel('log(s)')
        plt.ylabel('log(fs)')
        plt.show()
    if change_type == 3:  # 减少运算量的影响
        step_s = [1, 2, 4, 8, 16, 32]
        h_res = [hurst_stock(s_step=i) for i in step_s]  # 原则上下载一次数据即可，为了保持一致性，暂不修改
        for i in range(len(h_res)):
            log_s = h_res[i].get_log_s()
            log_fs = h_res[i].get_log_fs()
            plt.plot(log_s, log_fs, '.', label=str(step_s[i]) + ", Hurst:" + str(round(h_res[i].hurst, 2)))
            model = h_res[i].model_hurst
            plt.plot(log_s, model(log_s), 'k--')
        plt.legend(loc=2)
        plt.title('Hurst in different step_s(Data len:'+str(len(h_res[0].xts))+')')
        plt.xlabel('log(s)')
        plt.ylabel('log(fs)')
        plt.show()
    if change_type == 4:  # 计算随时间推移hurst指数的变化情况
        xts_len = 200
        data = pd.read_csv('./data/sh_D.csv')
        data_rate = Hurst.cal_standard(data.close.values.flat[:])
        h_res = [Hurst(data_rate[i:i + xts_len], s_step=10).hurst for i in range(len(data_rate) - xts_len)]
        h_res_smooth = ndimage.gaussian_filter(h_res, sigma=10)

        h_res_adj = np.hstack((h_res, np.repeat(np.nan, xts_len)))
        h_res_smooth_adj = np.hstack((h_res_smooth, np.repeat(np.nan, xts_len)))

        fig = plt.figure(figsize=(20, 15))
        ax1 = fig.add_subplot(111)
        ax1.plot(data.close.values.flat[:][::-1], 'r-', label='close price')
        plt.ylabel("SH close price")
        plt.xlabel("Time")
        plt.legend(loc=2)
        ax2 = ax1.twinx()
        ax2.plot(h_res_adj[::-1], 'b.', label='Hurst')
        ax2.plot(h_res_smooth_adj[::-1], 'b-', label='Hurst after smooth')
        plt.legend(loc=1)
        plt.title("Hurst change with the time")
        plt.ylabel("Hurst")
        plt.show()
    return

# 下载数据
# for i in tau_value:
#     download_stock(k_type=i)

if __name__ == '__main__':
    hurst_change(change_type=2)


