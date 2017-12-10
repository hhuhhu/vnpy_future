# coding=utf-8
""""保存回测所需的所有常量"""
import logging
logging.basicConfig(level=logging.INFO)
# 全局变量
BUY_TYPE = "buy"    # 多单标识
SELL_TYPE = "sell"   # 空单标识
BUY_CLOSE_TYPE = "buy_close"  # 多单平仓标识
SELL_CLOSE_TYPE = "sell_close"  # 空单平仓标识
OPEN_ORDER = 0     # 开仓事件标识
CLOSE_ORDER = 1   # 平开仓事件标识
TICK_CHANGE = 2    # 价格变动
BAR_MODE = 3       # Bar数据模式，默认数据回测模式
TICK_MODE = 4      # Tick数据模式

# ###################### 数据识别常量 ###################################
DATETIME = 0
BID = 1
ASK = 2
TICK_DATA_FLAG = 3
BAR_DATA_FLAG = 8
# ########################不同金融产品计算模式常量 #######################
FOREX_TYPE = u"forex"               # 外汇，采用MT4的计算模式
STOCK_TYPE = u"stock"               # 股票
FUTURE_TYPE = u"future"             # 期货
# ########################错误常量#####################################
INPUT_TYPE_ERROR = u"{}:输入类型错误！"

# ############################外汇货币种类############################
USD_CURRENCY = 1
NO_USD_CURRENCY = 2
CROSS_CURRENCY = 3
# #####################DataManager中数据合并的方式#############################################
DATE_STR_MERGE_TYPE = 0
DATETIME_MERGE_TYPE = 1
# #####################################订单类型的标志############################################
NORMAL_ORDER = "normal"               # buy_condition中下的订单
STOP_LOSS_ORDER = "stop_loss"         # 止损订单
PROFIT_LIMIT_ORDER = "profit_limit"   # 止盈订单
RISK_MANAGER_ORDER = "risk_manager"   # 风险管理下的订单
LAST_FORCE_ORDER = "last_force"       # 最后强行平仓订单考虑盈亏
