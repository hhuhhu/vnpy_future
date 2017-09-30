# coding=utf-8
"""绘制k线图"""
from matplotlib.dates import date2num
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.dates import DateFormatter, MinuteLocator
from matplotlib.ticker import FormatStrFormatter
from collections import deque
from pandas.core.api import Timestamp
from DataManager import ReadForexData
from pylab import mpl
import matplotlib.pylab as plt
import datetime

mpl.rcParams['font.sans-serif'] = ['SimHei']  # matplotlib中文指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['figure.autolayout'] = True
plt.style.use('ggplot')


def _candlestick(ax, df, colorup='k', colordown='r', alpha=1.0, data_struct='ohcl'):

    """
    Plot the time, open, high, low, close as a vertical line ranging
    from low to high.  Use a rectangular bar to represent the
    open-close span.  If close >= open, use colorup to color the bar,
    otherwise use colordown

    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    df : pandas data ,[date, open, high, close, low]
    colorup : color
        the color of the rectangle where close >= open
    colordown : color
         the color of the rectangle where close <  open
    alpha : float
        the rectangle alpha level

    Returns
    -------
    ret : tuple
        returns (lines, patches) where lines is a list of lines
        added and patches is a list of the rectangle patches added

    """
    last_bar_time = deque(maxlen=2)
    lines = []
    patches = []
    for date_string, row in df.iterrows():
        if isinstance(date_string, Timestamp):
            date_time = date_string.to_datetime()
        else:
            date_time = datetime.datetime.strptime(date_string, '%Y-%m-%d')
        t = date2num(date_time)

        # 第一根Bar不画
        last_bar_time.append(t)
        if len(last_bar_time) == 2:
            width = (last_bar_time[1] - last_bar_time[0]) * 0.8
        else:
            continue

        offset = width / 2.0

        if data_struct is "ohcl":
            open, high, close, low = row[:4]
        elif data_struct is "ohlc":
            open, high, low, close = row[:4]
        else:
            raise ValueError(u"目前仅支持两类‘ohcl’和‘ohlc’模式！")

        if close >= open:
            color = colorup
            lower = open
            height = close - open
        else:
            color = colordown
            lower = close
            height = open - close

        vline = Line2D(
            xdata=(t, t), ydata=(low, high),
            color=color,
            linewidth=0.5,
            antialiased=True,
        )

        rect = Rectangle(
            xy=(t - offset, lower),
            width=width,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
        rect.set_alpha(alpha)

        lines.append(vline)
        patches.append(rect)
        ax.add_line(vline)
        ax.add_patch(rect)
    ax.autoscale_view()

    return lines, patches


def candles(df, code, name, data_struct="ohcl"):
    all_minute = MinuteLocator(interval=15)
    minute_formatter = DateFormatter('%m-%d-%Y %H:%M')  # 如：2-29-2015
    y_formatter = FormatStrFormatter('%1.1f')
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    ax.xaxis.set_major_locator(all_minute)
    ax.xaxis.set_major_formatter(minute_formatter)
    ax.yaxis.set_major_formatter(y_formatter)
    _candlestick(ax, df, colorup='r', colordown='g', data_struct=data_struct)
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    ax.grid(True)
    plt.title(name + '  ' + code)
    return fig


if __name__ == '__main__':
    reader = ReadForexData()
    reader.read_csv(["../Data/TestData/XAUUSD1.csv"], start="2016.01.01", end="2016.08.26", symbol_name=["XAUUSD"],
                    skiprows=2000000)
    xauusd = reader.data["XAUUSD"]
    candles(xauusd[["open", "high", "close", "low"]].head(100), u"前40个Bar", u"XAUUSD")
    plt.show()
