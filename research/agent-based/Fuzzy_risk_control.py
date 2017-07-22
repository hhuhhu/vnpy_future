# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz


"""
本系统是一个简单的实时仓位监控系统 :根据即时价格波动的幅度，实时控制当前持仓比例的系统。
综合应用了Mamdani模糊逻辑控制和凯利公式，模糊控制部分主要输出两个变量（预期收益和预期损失），
在外汇中应用由于存在杠杆，则根据公式：
持仓比例={（1+|预期亏损|/预期盈利）*获胜概率-|预期亏损|}/杠杆倍数
此处获胜概率=0.5
    杠杆倍数=10
    需要安装模块：scikit-fuzzy
 python版本为 2.7
 """
#-------------------函数声明与定义---------------------------------------
# 关闭顶部和右侧的坐标，并采用紧凑布局
def SetAxes(_list):
    for ax in _list:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()    
    plt.tight_layout()
    plt.show()
 
def plot_menbership(universe=[],menbership=[],label=[],title=[]):
    """ 绘制隶属函数:输入变量分别是论域、隶属度、标签，图表标题"""
    color=['b','g','r','c','m','y']     #对应隶属函数曲线颜色
    fig,ax=plt.subplots(nrows=len(universe), figsize=(8, 9))
    for i in range(len(universe)):
        for j in range(len(menbership[i])):
                ax[i].plot(universe[i],menbership[i][j],color[j],label=label[i][j])
        ax[i].set_title(title[i])
        ax[i].legend() 
        
    SetAxes(ax)                         #关闭顶部和右侧的坐标，并采用紧凑布局
    
def plot_activity(output_universe=[],output_activity=[],output_membership=[]):
    """绘制激活规则后的输出隶属函数"""
    fig,ax = plt.subplots(nrows=len(output_universe),figsize=(8, 8))
    color=['b','g','r','c','m','y']
    title=['etLoss membership activity','etProfit membership activity']
    for i in range(len(output_universe)):
        output0= np.zeros_like(output_universe[i])    #绘图时所用的零向量
        for j in range(len(output_activity[i])):
            #填充推理后的预期损失范围
            ax[i].fill_between(output_universe[i],output0,output_activity[i][j],facecolor=color[j],alpha=0.7)
            ax[i].plot(output_universe[i],output_membership[i][j],color[j], linewidth=0.5, linestyle='--')
        ax[i].set_title(title[i])
            
    SetAxes(ax)
    
def plot_defuzzy(output_universe,output_membership,aggregated,defuzzValue):
    """绘制去模糊化的过程"""
    fig, ax= plt.subplots(nrows=len(output_universe),figsize=(8, 8))
    color=['b','g','r','c','m','y']
    title=['Expect Loss Aggregated membership and result (line)','Expect Profit Aggregated membership and result (line)']    
    for i in range(len(output_universe)):
        activation = fuzz.interp_membership(output_universe[i], aggregated[i],defuzzValue[i])  # for plot
        output0= np.zeros_like(output_universe[i])    #绘图时所用的零向量
        for j in range(len(output_membership[i])):
            ax[i].plot(output_universe[i],output_membership[i][j],color[j],linewidth=0.5, linestyle='--')
        if i == 0:
            ax[i].fill_between(etLoss_universe, output0, aggregated[i], facecolor='Orange', alpha=0.7)
        else:
            ax[i].fill_between(etProfit_universe, output0, aggregated[i], facecolor='Orange', alpha=0.7)
        ax[i].plot([defuzzValue[i],defuzzValue[i]],[0,activation],'k', linewidth=1.5, alpha=0.9)
        ax[i].set_title(title[i])
    SetAxes(ax)

#--------------------初始化系统------------------------------------   
#设置输入与输出变量的论域
amplitude_universe = np.linspace(-1, 1,num=100)
etLoss_universe=np.linspace(0, 0.5,num=100)
etProfit_universe=np.linspace(0, 1, num=100)

# 创建三个模糊隶属函数：一个价格波动幅度（amplitude）输入, 两个输出
amplitude_N_High=fuzz.zmf(amplitude_universe,-0.7,-0.3)         #波动幅度为负大
amplitude_N_Medin=fuzz.gaussmf(amplitude_universe, -0.2,0.1)    #波动幅度为负中
amplitude_N_Low=fuzz.gaussmf(amplitude_universe, -0.05, -0.1)   #波动幅度为负小
amplitude_P_Low=fuzz.gaussmf(amplitude_universe, 0.05, 0.1)     #波动幅度为正小
amplitude_P_Medin=fuzz.gaussmf(amplitude_universe, 0.1, 0.2)    #波动幅度为正中
amplitude_P_High=fuzz.smf(amplitude_universe, 0.3, 0.7)         #波动幅度为正大
#设置预期损失隶属函数
etLoss_Low=fuzz.gaussmf(etLoss_universe,0,0.1)
etLoss_Medin=fuzz.gaussmf(etLoss_universe,0.2,0.1)
etLoss_High=fuzz.smf(etLoss_universe,0.2,0.4)
#设置预期盈利隶属函数
etProfit_Low=fuzz.gaussmf(etProfit_universe,0,0.1)
etProfit_Medin=fuzz.gaussmf(etProfit_universe,0.2,0.1)
etProfit_High=fuzz.smf(etProfit_universe,0.2,0.4)

#--------------------模糊计算系统------------------------------------   
#设置输入向量
InputValue=0.4

# 模糊化：计算某个值的在各个输入模糊集上的隶属度
level_n_high = fuzz.interp_membership(amplitude_universe, amplitude_N_High, InputValue)
level_n_medin = fuzz.interp_membership(amplitude_universe, amplitude_N_Medin, InputValue)
level_n_low= fuzz.interp_membership(amplitude_universe, amplitude_N_Low, InputValue)
level_p_low= fuzz.interp_membership(amplitude_universe, amplitude_P_Low, InputValue)
level_p_medin = fuzz.interp_membership(amplitude_universe,amplitude_P_Medin, InputValue)
level_p_high= fuzz.interp_membership(amplitude_universe, amplitude_P_High, InputValue)

# 设置模糊推理规则：计算激励强度(即合成的过程)，这里规则均设为简单的“if x is A then y1 is B,y2 is C” 
etloss_activation_high=np.fmax(np.fmin(level_n_high,etLoss_High),
                               np.fmin(level_p_high,etLoss_High))
etloss_activation_medin=np.fmax(np.fmin(level_n_medin,etLoss_Medin),
                                np.fmin(level_p_medin,etLoss_Medin))
etloss_activation_low=np.fmax(np.fmin(level_n_low,etLoss_Low),
                              np.fmin(level_p_low,etLoss_Low))
etprofit_activation_low=np.fmax(np.fmin(level_p_low,etProfit_Low),
                                np.fmin(level_n_low,etProfit_Low))
etprofit_activation_medin=np.fmax(np.fmin(level_p_medin,etProfit_Medin),
                                  np.fmin(level_n_medin,etProfit_Medin))
etprofit_activation_high=np.fmax(np.fmin(level_p_high,etProfit_High),
                                 np.fmin(level_n_high,etProfit_High))

# 加总所有的规则：取每条规则的最大值
aggregated0 = np.fmax(etloss_activation_high,
                     np.fmax(etloss_activation_medin, etloss_activation_low))
aggregated1= np.fmax(etprofit_activation_high,
                     np.fmax(etprofit_activation_medin, etprofit_activation_low))
#计算去模糊化后的输出结果：采用centroid方法
ExpectLoss = fuzz.defuzz(etLoss_universe, aggregated0, 'centroid')
ExpectProfit = fuzz.defuzz(etProfit_universe, aggregated1, 'centroid')

#-----------------------------控制系统输出-----------------------------------
HoldingPercentage=((1+ExpectLoss/ExpectProfit)*0.5-ExpectLoss)/10  #凯利公式
print("InputValue:",InputValue)
print("ExpectLoss:",ExpectLoss)
print("ExpectProfit:",ExpectProfit)
print("HoldingPercentage:",HoldingPercentage)

#设置绘制隶属函数的输入参数
universe=[amplitude_universe,
          etLoss_universe,
          etProfit_universe]

menbership=[[amplitude_N_High,amplitude_N_Medin,amplitude_N_Low,
             amplitude_P_Low,amplitude_P_Medin,amplitude_P_High],
            [etLoss_Low,etLoss_Medin,etLoss_High],
            [etProfit_Low,etProfit_Medin,etProfit_High]]

label=[['Negative_High','Negative_Medin','Negative_Low',
        'Positive_Low','Positive_Medin','Positive_High'],
       ['Low','Medium','High'],
       ['Low','Medium','High']]

title=['amplitude classify','Expect Loss','Expect Profit']
plot_menbership(universe, menbership,label,title)  #绘制隶属函数

# 绘制推理后的输出变量的隶属函数
output_universe=[etLoss_universe,etProfit_universe]
output_activity=[[etloss_activation_high,etloss_activation_medin,etloss_activation_low],
                 [etprofit_activation_low,etprofit_activation_medin,etprofit_activation_high]]
output_membership=[[etLoss_High,etLoss_Medin,etLoss_Low],[etProfit_High,etProfit_Medin,etProfit_Low]]
plot_activity(output_universe, output_activity, output_membership)

# 绘制去模糊化的过程
aggregated=[aggregated0,aggregated1]
defuzzValue=[ExpectLoss,ExpectProfit]
plot_defuzzy(output_universe, output_membership, aggregated,defuzzValue)