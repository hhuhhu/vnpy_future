# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: fee_calc.py
@time: 2017/7/17 15:17
"""


class Fee:
    def __init__(self, fees):
        self.fees = fees
        self.total_fee = sum(self.fees)

    @property
    def fee_sum(self):
        fee_total = self.total_fee
        return fee_total

if __name__ == '__main__':
    fees = [1467.33, 3394.45, 676.26, 2300, 2300, 2829.81, 12673.60, 13717.60]
    fee_201707 = Fee(fees)
    print(fee_201707.total_fee)
