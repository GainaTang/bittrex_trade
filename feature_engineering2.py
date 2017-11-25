import numpy as np
from scipy.stats import linregress
import pandas as pd

# trade history#
def get_trade_mean(hist, curr_time, length, bos, mid):
    hist = hist.loc[(hist.timestamp >= curr_time - length) & (hist.timestamp <= curr_time)]
    if bos != 'both':
        hist = hist.loc[hist.order_type__name == bos]
    tmp = []
    quant = []
    if hist.shape[0] > 0:
        return np.log(np.dot(hist.quantity, hist.price) / hist.quantity.sum() / mid)
    else:
        return 0


def get_trade_relative_volume(hist, curr_time, length, bos, mid):
    hist = hist.loc[(hist.timestamp >= curr_time - length) & (hist.timestamp <= curr_time)]
    if bos != 'both':
        hist = hist.loc[hist.order_type__name == bos]
    tmp = []
    quant = []
    if hist.shape[0] > 0:
        return np.dot(hist.quantity, hist.price) / hist.quantity.count() / mid
    else:
        return 0


def get_trade_power_adjusted_rate(hist, curr_time, length, mid, spread, power):
    hist = hist.loc[(hist.timestamp >= curr_time - length) & (hist.timestamp <= curr_time)]
    if hist.shape[0] == 0:
        return 0
    else:
        buy_hist = hist.loc[hist.order_type__name == 'BUY']
        sell_hist = hist.loc[hist.order_type__name == 'SELL']

        sell_weights = 1 / (sell_hist.quantity * (0.5 * spread / (sell_hist.price * (1 + 0.000001) - mid)) ** power)
        buy_weights = 1 / (buy_hist.quantity * (0.5 * spread / (buy_hist.price * (1 - 0.000001) - mid)) ** power)

        return np.log((np.dot(sell_weights, sell_hist.price) + np.dot(buy_weights, buy_hist.price)) /
                      (np.sum(buy_weights) + np.sum(sell_weights)) / mid)


def get_trade_power_imbalance(hist, curr_time, length, mid, spread, power):
    hist = hist.loc[(hist.timestamp >= curr_time - length) & (hist.timestamp <= curr_time)]
    if hist.shape[0] == 0:
        return 0
    else:
        buy_hist = hist.loc[hist.order_type__name == 'BUY']
        sell_hist = hist.loc[hist.order_type__name == 'SELL']

        sell_weights = sell_hist.quantity * (0.5 * spread / (sell_hist.price * (1 + 0.000001) - mid)) ** power
        buy_weights = buy_hist.quantity * (0.5 * spread / (buy_hist.price * (1 - 0.000001) - mid)) ** power

        return (np.sum(sell_weights) - np.sum(buy_weights)) / (np.sum(sell_weights) + np.sum(buy_weights))


def get_trade_aggr(hist, curr_time, length):
    # print('fe_start')
    hist = hist.loc[(hist.timestamp >= curr_time - length) & (hist.timestamp <= curr_time)]
    # print('cutting_done')
    if hist.shape[0] == 0:
        trade_aggr = 0
    else:
        buy = hist.loc[hist.order_type__name == 'BUY'].quantity.sum()
        sell = hist.loc[hist.order_type__name == 'SELL'].quantity.sum()
        trade_aggr = (buy - sell) / (buy + sell)
        # print('calculation done')

    return trade_aggr


def get_trade_trend(hist, curr_time, length, bos):
    hist = hist.loc[(hist.timestamp >= curr_time - length) & (hist.timestamp <= curr_time)]
    if bos != 'both':
        hist = hist.loc[hist.order_type__name == bos]

    lt = list(hist.quantity)
    if len(lt) > 3:
        trend = linregress(range(len(lt)), lt)[0]/np.sum(lt)
    else:
        trend = 0
    return trend


def get_trade_partial(hist, curr_time, length):
    hist = hist.loc[(hist.timestamp >= curr_time - length) & (hist.timestamp <= curr_time)]
    if hist.shape[0] > 0:
        trade_partial = hist.loc[hist.fill_type__name == 'FILL'].count()[0] / hist.shape[0]
    else:
        trade_partial = 0

    return trade_partial


# orderbook#
def get_mid(sell, buy):
    return (sell[0][0] + buy[0][0]) / 2


def get_spread(sell, buy):
    return (sell[0][0] - buy[0][0])


def get_power_adjusted_rate_sell(book, spread, mid, n=10, power=2):
    sell_weights = []

    sell = []

    for x in book[:n]:
        #         print(x['Rate'],mid)
        #         if spread==0:
        #             ratio=1
        #         else:
        ratio = spread / (x[0] * (1 + 0.000001) - mid)
        weight = x[1] * (0.5 * ratio) ** power
        #         print(x['Quantity'])
        sell_weights.append(1 / weight)
        sell.append(x[0])
    return np.log((np.dot(sell_weights, sell)) /
                  (np.sum(sell_weights)) / mid)


def get_power_adjusted_rate_buy(book, spread, mid, n=10, power=2):
    buy_weights = []
    buy = []

    for x in book[:n]:
        #         print(x['Rate'],mid)
        #         if spread==0:
        #             ratio=1
        #         else:
        ratio = spread / (x[0] * (1 - 0.000001) - mid)
        weight = x[1] * (0.5 * ratio) ** power
        #         print(x['Quantity'])
        buy_weights.append(1 / weight)
        buy.append(x[0])
    return np.log((np.dot(buy_weights, buy)) /
                  (np.sum(buy_weights)) / mid)


def get_power_adjusted_rate(sellbook, buybook, spread, mid, n=10, power=2):
    sell_weights = []
    buy_weights = []
    buy = []
    sell = []

    for x in sellbook[:n]:
        #         print(x['Rate'],mid)
        #         if spread==0
        #             ratio=1
        #         else:
        ratio = spread / (x[0] * (1 + 0.000001) - mid)
        weight = x[1] * (0.5 * ratio) ** power
        #         print(x['Quantity'])
        sell_weights.append(1 / weight)
        sell.append(x[0])
    for x in buybook[:n]:
        #         print(x['Rate'],mid)
        ratio = spread / (x[0] * (1 - 0.000001) - mid)
        weight = x[1] * (0.5 * ratio) ** power
        #         print(x['Quantity'])
        buy_weights.append(1 / weight)
        buy.append(x[0])
    return np.log((np.dot(sell_weights, sell) + np.dot(buy_weights, buy)) /
                  (np.sum(buy_weights) + np.sum(sell_weights)) / mid)


def get_power_imbalance(sellbook, buybook, spread, mid, n=10, power=2):
    sell_weights = []
    buy_weights = []
    buy = []
    sell = []
    for x in sellbook[:n]:
        weight = x[0] * (0.5 * spread / (x[1] * (1 + 0.000001) - mid)) ** power
        sell_weights.append(weight)
    # sell.append(x['Rate'])
    for x in buybook[:n]:
        weight = x[0] * (0.5 * spread / (x[1] * (1 - 0.000001) - mid)) ** power
        buy_weights.append(weight)
    # buy.append(x['Rate'])
    length = min(len(buy_weights), len(sell_weights))
    return (np.sum(np.array(sell_weights[:length]) - np.array(buy_weights[:length])) /
            np.sum(np.array(sell_weights[:length]) + np.array(buy_weights[:length])))


def get_book_trend(book, topn, var):
    if var == 0:
        lt = [a for [a, b] in book[:topn]]
    else:
        lt = [b for [a, b] in book[:topn]]
    trend = linregress(range(len(lt)), lt)[0]
    trend = trend / np.mean(lt)
    return trend