# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: cost_compute.py
@time: 2017/8/7 16:17
"""


class Cost:

    def rb_cost(self, i, j):
        """
        计算螺纹钢成本
        :param i: 铁矿石价格
        :param j: 焦炭价格
        :param k: 环保成本（100~130元/吨,按宝钢股份13年治污成本推算得到）
        :return: 螺纹钢成本价格
        """
        k = 130
        rb_cost = (i*1.6+j*0.45)/0.9+k

        return rb_cost

    def jd_cost(self):
        """
        计算鸡蛋生产成本
        :return: 鸡蛋价格/斤
        """
        return 3.0

    def a_m_y(self, a, m ,y):
        """
        计算压榨利润
        :param a: 美式大豆价格
        :param m: 豆粕价格
        :param y: 豆油价格
        :return: 压榨利润
        """
        arbitrage = a - 0.80*m -0.18*y
        return -arbitrage

    def ru_cost(self):
        """
        国内两大橡胶产地云南和海南，云南亩产多，成本价在12000左右，海南成本在13000左右，这里取小值12000,橡胶的成本50%为人工成本，
        因此东南亚国家成本价应该会更低；另外，橡胶应用中，天然橡胶占比为50%；70%应用于轮胎制造。天然橡胶性能优于合成橡胶，并且天然橡胶成本
        通常大于合成橡胶
        :return: 
        """
        return 12000

    def pvc_cost(self):
        """
        pvc两种生产方式，电石法：6280~6453，且电石法污染严重，故不做考虑，因此按乙烯法计算；
        :return: 
        """

    def bu_profit(self):
        profit = 150
        return profit

    def ni_cost(self):

        return
if __name__ == '__main__':
    cost = Cost()
    i = 610
    j = 2547.5
    rb_cost = cost.rb_cost(i=i, j=j)
    print("螺纹钢成本价是：{}".format(rb_cost))
    artitrage = cost.a_m_y(3830,2722,6344)
    print(artitrage)
