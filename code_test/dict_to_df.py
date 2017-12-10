# -*- coding: utf-8 -*-
"""
@author: Daniel
@contact: 511735184@qq.com
@file: dict_to_df.py
@time: 2017/11/10 9:47
"""
import pandas as pd

df={'low': 1254.37, 'open': 1262.78, 'volume': 80156, 'ask': 1262.78, 'symbol': 'XAUUSD', 'high': 1271.38, 'close': 1255.98, 'bid': 1262.78}
df = pd.DataFrame(df)
print(df)