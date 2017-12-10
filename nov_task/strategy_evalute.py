# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: strategy_evalute.py
@time: 2017/11/20 16:45
"""


def strategy_evalute(recovery_efficiency, max_drawdown, net_income_ratio, transaction_frequency, win_rate,
                     profit_loss_ratio):
    """
    
    计算策略得分，60分为合格
    :param recovery_efficiency: 采收率，按年化算； 乘子：3
    :param max_drawdown: 最大回测率； 乘子：1.5
    :param net_income_ratio: 净收益率，按年化算； 乘子：1.5
    :param transaction_frequency: 交易频率，年交易次数/252； 乘子：1
    :param win_rate: 胜率 获利交易次数/交易总次数； 乘子：1.5
    :param profit_loss_ratio: 平均盈利/平均亏损； 乘子1.5
    :return: 加权总得分
    """
    re_score = 0.0
    md_score = 0.0
    nir_score = 0.0
    tf_score = 0.0
    wr_score = 0.0
    plr_score = 0.0
    # 采收率得分
    if recovery_efficiency <= 0:
        re_score = 0
    elif recovery_efficiency <= 0.4:
        re_score = 1
    elif recovery_efficiency <= 0.8:
        re_score = 2
    elif recovery_efficiency <= 1.2:
        re_score = 3
    elif recovery_efficiency <= 1.6:
        re_score = 4
    elif recovery_efficiency <= 2.0:
        re_score = 5
    elif recovery_efficiency <= 2.5:
        re_score = 6
    elif recovery_efficiency <= 3.0:
        re_score = 7
    elif recovery_efficiency <= 4:
        re_score = 8
    elif recovery_efficiency <= 6:
        re_score = 9
    else:
        re_score = 10

    # 最大回撤得分
    if max_drawdown >= 0.4:
        md_score = 0
    elif max_drawdown >= 0.35:
        md_score = 1
    elif max_drawdown >= 0.30:
        md_score = 2
    elif max_drawdown >= 0.26:
        md_score = 3
    elif max_drawdown >= 0.22:
        md_score = 4
    elif max_drawdown >= 0.18:
        md_score = 5
    elif max_drawdown >= 0.14:
        md_score = 6
    elif max_drawdown >= 0.10:
        md_score = 7
    elif max_drawdown >= 0.06:
        md_score = 8
    elif max_drawdown >= 0.03:
        md_score = 9
    else:
        md_score = 10
    # 净收益率得分
    if net_income_ratio <= 0:
        nir_score = 0
    elif net_income_ratio <= 0.05:
        nir_score = 1
    elif net_income_ratio <= 0.10:
        nir_score = 2
    elif net_income_ratio <= 0.17:
        nir_score = 3
    elif net_income_ratio <= 0.35:
        nir_score = 4
    elif net_income_ratio <= 0.50:
        nir_score = 5
    elif net_income_ratio <= 0.75:
        nir_score = 6
    elif net_income_ratio <= 1.00:
        nir_score = 7
    elif net_income_ratio <= 1.50:
        nir_score = 8
    elif net_income_ratio <= 2.50:
        nir_score = 9
    else:
        nir_score = 10

    # 交易频率
    if transaction_frequency <= 0.1:
        tf_score = 0
    elif transaction_frequency <= 0.2:
        tf_score = 1
    elif transaction_frequency <= 0.3:
        tf_score = 2
    elif transaction_frequency <= 0.4:
        tf_score = 3
    elif transaction_frequency <= 0.5:
        tf_score = 4
    elif transaction_frequency <= 0.6:
        tf_score = 5
    elif transaction_frequency <= 0.8:
        tf_score = 6
    elif transaction_frequency <= 1.0:
        tf_score = 7
    elif transaction_frequency <= 1.5:
        tf_score = 8
    elif transaction_frequency <= 2.0:
        tf_score = 9
    else:
        tf_score = 10

    # 每次（对）交易胜率
    if win_rate <= 0.1:
        wr_score = 0
    elif wr_score <= 0.2:
        wr_score = 1
    elif wr_score <= 0.3:
        wr_score = 2
    elif wr_score <= 0.35:
        wr_score = 3
    elif wr_score <= 0.40:
        wr_score = 4
    elif wr_score <= 0.45:
        wr_score = 5
    elif wr_score <= 0.50:
        wr_score = 6
    elif wr_score <= 0.55:
        wr_score = 7
    elif wr_score <= 0.6:
        wr_score = 8
    elif wr_score <= 0.7:
        wr_score = 9
    else:
        wr_score = 10

    # 每次（对）交易平均利润亏损比
    if profit_loss_ratio <= 0.1:
        plr_score = 0
    elif profit_loss_ratio <= 0.3:
        plr_score = 1
    elif profit_loss_ratio <= 0.5:
        plr_score = 2
    elif profit_loss_ratio <= 0.65:
        plr_score = 3
    elif profit_loss_ratio <= 0.8:
        plr_score = 4
    elif profit_loss_ratio <= 1.0:
        plr_score = 5
    elif profit_loss_ratio <= 1.2:
        plr_score = 6
    elif profit_loss_ratio <= 1.6:
        plr_score = 7
    elif profit_loss_ratio <= 2.0:
        plr_score = 8
    elif profit_loss_ratio <= 3.0:
        plr_score = 9
    else:
        plr_score = 10

    # 总得分
    total_score = re_score * 3 + md_score * 1.5 + nir_score * 1.5 + tf_score * 1 + wr_score * 1.5 + plr_score * 1.5

    return total_score


if __name__ == '__main__':
    # recovery_efficiency, max_drawdown, net_income_ratio, transaction_frequency, win_rate,
    # profit_loss_ratio

    args = [1.12, 0.1649, 0.2640, 21 * 2 / 252, 12 / 21, 310.88 / 121.15]  # (62, 16) 得分40
    args = [2.26*2, 0.1287, 0.3175*2, 31 * 2 / 252, 21 / 31, 197.81 / 97.89]  # （36， 120）得分51.5 、63.5（年化）
    # args = [1.58, 0.2039, 0.5815, 52 * 2 / 252, 33 / 52, 238.92 / 108.91]  # 参数组合 得分47.5
    args = [1.74*6, 0.0216, 0.0395*6, 34 * 6 / 252, 20 / 34, 36.32 / 26.35]  # 萧哥参数1 50.5
    # args = [1.53, 0.021, 0.0336, 25 * 6 / 252, 12 / 25, 51.24 / 21.42]  # 萧哥参数2  48.5
    total_score = strategy_evalute(*args)
    print(total_score)
