import pickle
from feature_engineering2 import *
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import time
from multiprocessing import Pool
def parallelize_dataframe(df, func):
    df_split = np.array_split(df, 30)
    pool = Pool(30)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
def fe(data):
    for topn in [10, 20, 30]:
        for bos in ['buy', 'sell']:
            for var in [1]:
                data['{}_{}_max_{}'.format(bos, var, topn)] = data['{}'.format(bos)].apply(
                    lambda x: np.max([b for [a,b] in x[:topn]]))
                data['{}_{}_min_{}'.format(bos, var, topn)] = data['{}'.format(bos)].apply(
                    lambda x: np.min([b for [a,b] in x[:topn]]))
                data['{}_{}_maxind_{}'.format(bos, var, topn)] = data['{}'.format(bos)].apply(
                    lambda x: [b for [a,b] in x[:topn]].index(np.max([b for [a,b] in x[:topn]])))
                data['{}_{}_minind_{}'.format(bos, var, topn)] = data['{}'.format(bos)].apply(
                    lambda x: [b for [a,b] in x[:topn]].index(np.min([b for [a,b] in x[:topn]])))
    for topn in [10, 60]:
        for power in [2]:
            data['power_adjusted_price_top{}_power{}'.format(topn, power)] = data.apply(
                lambda row: get_power_adjusted_rate(
                    row['sell'],
                    row['buy'],
                    row['spread'],
                    row['mid'], n=topn, power=power), axis=1)
            data['power_adjusted_price_buy_top{}_power{}'.format(topn, power)] = data.apply(
                lambda row: get_power_adjusted_rate_buy(
                    row['buy'],
                    row['spread'],
                    row['mid'], n=topn, power=power), axis=1)
            data['power_adjusted_price_sell_top{}_power{}'.format(topn, power)] = data.apply(
                lambda row: get_power_adjusted_rate_sell(
                    row['sell'],
                    row['spread'],
                    row['mid'], n=topn, power=power), axis=1)

            data['power_imbalance_top{}_power{}'.format(topn, power)] = data.apply(lambda row: get_power_imbalance(
                row['sell'],
                row['buy'],
                row['spread'],
                row['mid'], n=topn, power=power), axis=1)
    for topn in [10, 60]:
        for var in [0, 1]:
            for bos in ['buy', 'sell']:
                data['book_trend_{}_{}_{}'.format(bos, topn, var)] = data.apply(
                    lambda x: get_book_trend(x[bos], topn, var)
                    , axis=1)
    # data['buy'] = data.buy.apply(lambda x: x[:60])
    # data['sell'] = data.sell.apply(lambda x: x[:60])
    data['v_ask60']=data.sell.apply(lambda x: np.sum([b for [a,b] in x[:topn]]))
    data['v_bid60']=data.buy.apply(lambda x: np.sum([b for [a,b] in x[:topn]]))
    deno=data['v_ask60']+data['v_bid60']
    data['v_ask60']=data['v_ask60']/deno
    data['v_bid60'] = data['v_bid60'] / deno
    data['ask'] = data.sell.apply(lambda x: x[0][0])
    data['bid'] = data.buy.apply(lambda x: x[0][0])
    for delta in [
        10,
        # 20,
        # 30,
        60,
        180
    ]:
        data['trade_aggr{}'.format(delta)] = data.apply(lambda row: get_trade_aggr(hist, curr_time=row['timestamp'],
                                                                               length=pd.Timedelta(seconds=delta)
                                                                               ), axis=1)
        data['trade_partial{}'.format(delta)] = data.apply(
            lambda row: get_trade_partial(hist, curr_time=row['timestamp'],
                                          length=pd.Timedelta(seconds=delta)
                                          ), axis=1)

        for power in [2,
                      # 4,
                      # 8
                      ]:
            data['trade_power_adjusted_rate_{}_{}'.format(delta, power)] = data.apply(
                lambda row: get_trade_power_adjusted_rate(hist,
                                                          curr_time=row['timestamp'],
                                                          length=pd.Timedelta(seconds=delta),
                                                          mid=row['mid'],
                                                          spread=row['spread'],
                                                          power=power
                                                          ), axis=1)

            data['trade_power_imbalance_{}_{}'.format(delta, power)] = data.apply(
                lambda row: get_trade_power_imbalance(hist,
                                                      curr_time=row['timestamp'],
                                                      length=pd.Timedelta(seconds=delta),
                                                      mid=row['mid'],
                                                      spread=row['spread'],
                                                      power=power
                                                      ), axis=1)
        for bos in ['BUY',
                    'SELL',
                    'both'
                    ]:
            data['trade_mean{}_{}'.format(delta, bos)] = data.apply(
                lambda row: get_trade_mean(hist, curr_time=row['timestamp'],
                                           length=pd.Timedelta(seconds=delta),
                                           bos=bos,
                                           mid=row['mid']
                                           ), axis=1)
            # print('trade_mean')

            data['trade_trend{}_{}'.format(delta, bos)] = data.apply(lambda row: get_trade_trend(hist,
                                                                                                 bos=bos,
                                                                                                 curr_time=row['timestamp'],
                                                                                                 length=pd.Timedelta(
                                                                                                     seconds=delta)
                                                                                                 ), axis=1)
            print('trade_trend')
            data['trade_relative_volume{}_{}'.format(delta, bos)] = data.apply(lambda row: get_trade_relative_volume(
                hist, curr_time=row['timestamp'],
                length=pd.Timedelta(seconds=delta),
                bos=bos,
                mid=row['mid']), axis=1)
            # print('volume')
    return data

# with open('data1min_12-22.pkl', 'rb') as f:
#     pairs=pickle.load( f)

with open('data1min_12-23.pkl', 'rb') as f:
    pairs=pickle.load(f)
with open('hist_11-22.pkl', 'rb') as f:
    hists=pickle.load(f)
new_pairs=[]
start_time = time.time()
count=0
for data, hist in zip(pairs, hists):
    data = data.loc[data.spread != 0]
    data=parallelize_dataframe(data, fe)
    new_pairs.append(data)
    count+=1
    print(count)
with open('newdata1min_12-23.pkl', 'wb') as f:
    pickle.dump(new_pairs, f)
