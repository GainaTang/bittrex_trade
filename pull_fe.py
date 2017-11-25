from datetime import datetime, timezone
import pandas as pd
import io
from io import StringIO
import gc
import numpy as np
# datetime(2017,11,11,0,0).timestamp()
from feature_engineering2 import *
import requests
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import linregress
from multiprocessing import Pool
def parallelize_dataframe(df, func):
    df_split = np.array_split(df, 30)
    pool = Pool(30)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
def get_timestamp(data):
    data['timestamp'] = data.timestamp.apply(lambda x: pd.to_datetime(x, utc=True))
    return data

big_volume = [
    'BTC-KMD',
    'BTC-ADX',
    'BTC-GRS',
    'BTC-ETC',
    'BTC-SYS',
    'BTC-ZEC',
    'BTC-GRS',
    'BTC-LTC',
    'BTC-NEO',
    # 'BTC-XLM',
    # 'BTC-OMG',
    # 'BTC-ADA',
    # 'BTC-QTUM',
    # 'BTC-ARK',
    # 'BTC-WAVES',
    # 'BTC-XRP',
    # 'BTC-MCO',
    # 'BTC-XVG',
    # 'BTC-VTC',
    # 'BTC-ETH',
    # 'BTC-BCC',
    # 'BTC-STRAT',
    # 'BTC-ZEN',
    # 'BTC-LSK'
]
URL = 'https://sg.blockchainflow.io:188/api/1.0/json/order_book/'
URL2 = 'https://sg.blockchainflow.io:188/api/1.0/csv/market_history/'

USER = 'api'
PASS = 'VdhuDa4XCfDXXKfWDMGq5cp'
pairs = []
hists = []
for market in big_volume:
    timestart = datetime(2017, 11, 11, 0, 0,0,0,timezone.utc).timestamp()
    timeend = datetime(2017, 11, 22, 0, 0,0,0,timezone.utc).timestamp()
    payload = {
        'exchange': 'Bittrex',
        'market': market,
        'timestamp_from': timestart,
        'timestamp_to': timeend,
    }
    res = requests.get(URL, auth=(USER, PASS), params=payload)
    print('pulling on pair {} done'.format(market))
    dic = res.json()
    print('todict')
    data = pd.DataFrame.from_dict(dic, orient='index')
    data.index.names = ['timestamp']
    data.reset_index(inplace=True)
    data['timestamp'] = data.timestamp.apply(lambda x: datetime.fromtimestamp(float(x), timezone.utc))
    print('processed')
    for colum in ['buy', 'sell']:
        data = data.loc[data[colum].str.len() != 0]
    data.dropna(how='any', inplace=True, axis=0)
    data['mid'] = data.apply(lambda row: (row['sell'][0][0] + row['buy'][0][0]) / 2, axis=1)
    data['spread'] = data.apply(lambda row: (row['sell'][0][0] - row['buy'][0][0]), axis=1)
    # for topn in [10, 20, 60, 180]:
    #     for power in [2, 4, 8]:
    #         data['power_adjusted_price_top{}_power{}'.format(topn, power)] = data.apply(
    #             lambda row: get_power_adjusted_rate(
    #                 row['sell'],
    #                 row['buy'],
    #                 row['spread'],
    #                 row['mid'], n=topn, power=power), axis=1)
    #         data['power_adjusted_price_buy_top{}_power{}'.format(topn, power)] = data.apply(
    #             lambda row: get_power_adjusted_rate_buy(
    #                 row['buy'],
    #                 row['spread'],
    #                 row['mid'], n=topn, power=power), axis=1)
    #         data['power_adjusted_price_sell_top{}_power{}'.format(topn, power)] = data.apply(
    #             lambda row: get_power_adjusted_rate_sell(
    #                 row['sell'],
    #                 row['spread'],
    #                 row['mid'], n=topn, power=power), axis=1)
    #
    #         data['power_imbalance_top{}_power{}'.format(topn, power)] = data.apply(lambda row: get_power_imbalance(
    #             row['sell'],
    #             row['buy'],
    #             row['spread'],
    #             row['mid'], n=topn, power=power), axis=1)
    # for topn in [10, 20, 60, 180]:
    #     for var in [0, 1]:
    #         for bos in ['buy', 'sell']:
    #             data['book_trend_{}_{}_{}'.format(bos, topn, var)] = data.apply(
    #                 lambda x: get_book_trend(x[bos], topn, var)
    #                 , axis=1)
    # data['buy'] = data.buy.apply(lambda x: x[:60])
    # data['sell'] = data.sell.apply(lambda x: x[:60])
    # data['ask'] = data.sell.apply(lambda x: x[0][0])
    # data['bid'] = data.buy.apply(lambda x: x[0][0])
    # pairs.append(data)
    gc.collect()

    res = requests.get(URL2, auth=(USER, PASS), params=payload)
    hist = pd.read_csv(StringIO(res.content.decode('utf-8')))
    hist = parallelize_dataframe(hist, get_timestamp)
    hists.append(hist)
    print(market)

    gc.collect()
import pickle

with open('data1min_12-23.pkl', 'wb') as f:
    pickle.dump(pairs, f)

import pickle

with open('hist_11-22.pkl', 'wb') as f:
    pickle.dump(hists, f)



