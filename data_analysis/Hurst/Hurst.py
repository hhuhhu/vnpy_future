# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
__author__ = 'czj'


class Hurst:

    @staticmethod
    def cal_standard(x):    # 对数差分标准化
        return np.diff(np.log(x), 1)

    def __init__(self, xts=np.random.normal(size=500), fit_degree=1, s_min=4, s_step=1, v_min=4):
        self.xts = xts
        self.fit_degree = fit_degree
        self.s_min = s_min
        self.s_step = s_step
        self.v_min = v_min

        self.__len_xts = len(self.xts)  # 序列长度
        self.__bl_input_correct = (self.__len_xts > self.s_min*self.v_min * 2*self.s_step)  # 输入是否符合要求判断
        self.__cum_res_of_xts = np.cumsum(self.xts-np.mean(self.xts))  # 原始序列xts的累计残差
        self.__s = range(self.s_min, self.__len_xts//self.v_min+1, self.s_step)  # 分段长度序列
        self.__m = [self.__len_xts//self.__s[i] for i in range(len(self.__s))]  # 获取区间数序列
        self.__bl_need_reverse = np.mod(self.__len_xts, self.__s) != 0   # 判断对应分段长度，是否需要序列逆序

        self.__sum_res_fit = self.__cal_sum_res()  # 计算给定分法下的每一个区间的拟合值和预测值之间的残差平方和，并累计给定方法下所有区间的残差和
        self.__fs = self.__cal_fs()  # 计算残差和的平均值
        self.model_hurst = np.poly1d(np.polyfit(np.log(self.__s), np.log(self.__fs), 1))
        self.hurst = self.model_hurst[1]

    def get_log_s(self):
        return np.log(self.__s)

    def get_log_fs(self):
        return np.log(self.__fs)

    def __cal_sum_res(self):
        result = np.zeros(len(self.__s))
        for i in range(len(self.__s)):
            x_v = range(self.__s[i])
            for v in range(self.__m[i]):
                y_v = self.__cum_res_of_xts[v*self.__s[i]:(v+1)*self.__s[i]]
                fit_model1 = np.poly1d(np.polyfit(x_v, y_v, self.fit_degree))
                result[i] = result[i]+np.sum((fit_model1(x_v)-y_v)**2)/self.__s[i]
                if self.__bl_need_reverse[i]:
                    y_inverse_v = self.__cum_res_of_xts[(self.__len_xts-(v+1)*self.__s[i]):(self.__len_xts-v*self.__s[i])]
                    fit_model2 = np.poly1d(np.polyfit(x_v, y_inverse_v, self.fit_degree))
                    result[i] = result[i]+np.sum((fit_model2(x_v)-y_inverse_v)**2)/self.__s[i]
        return result

    def __cal_fs(self):
        result = np.zeros(len(self.__m))
        for i in range(len(self.__m)):
            if self.__bl_need_reverse[i]:
                result[i] = np.sqrt(self.__sum_res_fit[i]/(2*self.__m[i]))
            else:
                result[i] = np.sqrt(self.__sum_res_fit[i]/self.__m[i])
        return result

    def __plot_xts(self):
        plt.plot(self.xts, "b.")
        plt.plot(self.__cum_res_of_xts, "r-")
        plt.legend(["xts", "cum_sum_res_xts"])
        plt.xlabel("Time", fontsize=16)
        plt.ylabel("xts & cum_sum_res", fontsize=16)
        plt.title("xts & cum_sum_res vs. time", fontsize=20)
        plt.draw()

    def __plot_cum_res(self):
        plt.plot(self.__cum_res_of_xts)
        plt.draw()

    def __plot_fs_s(self):
        plt.plot(self.__s, self.__fs, "b*")
        plt.xlabel("s")
        plt.ylabel("fs")
        plt.title("fs vs. s")
        plt.draw()

    def __plot_lfs_ls(self):
        plt.plot(np.log(self.__s), np.log(self.__fs), "r*", label="data")
        plt.plot(np.log(self.__s), self.model_hurst(np.log(self.__s)), "b--", label="fit line")
        plt.legend(loc=2)
        xloc_text = 0.6*(np.max(np.log(self.__s))-np.min(np.log(self.__s)))+np.min(np.log(self.__s))
        yloc_text = 0.4*(np.max(np.log(self.__fs))-np.min(np.log(self.__fs)))+np.min(np.log(self.__fs))
        text_content = "y="+str(self.model_hurst).replace("\n", "")+"\nHurst:%f" % self.model_hurst[1]
        print(text_content)
        plt.text(xloc_text, yloc_text, text_content)
        plt.xlabel("log(s)", fontsize=16)
        plt.ylabel("log(Fs)", fontsize=16)
        plt.title("log(Fs) vs. log(s)")
        plt.draw()

    def __plot_hist_fs(self):
        plt.hist(self.__fs)
        plt.draw()

    def __plot_hist_xts(self):
        plt.hist(self.xts)
        plt.title("xts distribution")
        plt.xlabel("xts value", fontsize=12)
        plt.ylabel("num", fontsize=12)
        print(plt.axis())
        xy_axis = plt.axis()
        x_loc = 0.1*(xy_axis[1]-xy_axis[0])+xy_axis[0]
        y_loc = 0.75*(xy_axis[3]-xy_axis[2])+xy_axis[2]
        plt.text(x_loc, y_loc, "xts num:"+str(self.__len_xts))
        plt.draw()

    def plot_all(self):
        plt.subplot(221)
        self.__plot_hist_xts()
        plt.subplot(222)
        self.__plot_xts()
        plt.subplot(223)
        self.__plot_fs_s()
        plt.subplot(224)
        self.__plot_lfs_ls()
        plt.show()


# 使用正态分布随机数计算结果
# myHurst = Hurst()
# myHurst.plot_all()


# 使用外汇xauusd数据计算结果
# data = pd.read_csv('xauusd_close.csv')
# data_values = data.values
# data_array = data_values.flat[:]
# data_standard = Hurst.cal_standard(data_array)
# myHurst = Hurst(data_standard[11000:12000])
# myHurst.plot_all()


# 使用tushare数据计算结果
# stock_data = ts.get_hist_data('sh')
# stock_data_close = stock_data.close
# stock_data_arr = stock_data_close.values
# stock_data_arr1 = stock_data_arr.flat[:]
# data_standard = Hurst.cal_standard(stock_data_arr1)
# myHurst = Hurst(data_standard[1:1000])
# myHurst.plot_all()

if __name__=="__main__":
    data = pd.read_csv('data/sh_5.csv')
    data_close = data.close
    # print(data_close)

    myHurst = Hurst(data_close)
    print(myHurst.hurst)
