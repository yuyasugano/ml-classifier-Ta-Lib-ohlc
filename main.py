#!/usr/bin/python
import csv
import time
import json
import talib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta, timezone
from backtesting import Backtest
from backtesting import Strategy
from backtesting.lib import crossover
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold

headers = {'Content-Type': 'application/json'}
api_url_base = 'https://public.bitbank.cc'
pair = 'btc_jpy'
period = '1min'

today = datetime.today()
yesterday = today - timedelta(days=1)
today = "{0:%Y%m%d}".format(today)
yesterday = "{0:%Y%m%d}".format(yesterday)

def api_ohlcv(timestamp):
    api_url = '{0}/{1}/candlestick/{2}/{3}'.format(api_url_base, pair, period, timestamp)
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        ohlcv = json.loads(response.content.decode('utf-8'))['data']['candlestick'][0]['ohlcv']
        return ohlcv
    else:
        return None

def main():
    ohlcv = api_ohlcv('20190901')
    open, high, low, close, volume, timestamp = [],[],[],[],[],[]

    for i in ohlcv:
        open.append(int(i[0]))
        high.append(int(i[1]))
        low.append(int(i[2]))
        close.append(int(i[3]))
        volume.append(float(i[4]))
        time_str = str(i[5])
        timestamp.append(datetime.fromtimestamp(int(time_str[:10])).strftime('%Y/%m/%d %H:%M:%M'))

    date_time_index = pd.to_datetime(timestamp) # convert to DateTimeIndex type
    df = pd.DataFrame({'open': open, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=date_time_index)
    # df.index += pd.offsets.Hour(9) # adjustment for JST if required
    print(df.shape)
    print(df.columns)

    # pct_change
    f = lambda x: 1 if x>0.0001 else -1 if x<-0.0001 else 0 if -0.0001<=x<=0.0001 else np.nan
    y = df.rename(columns={'close': 'y'}).loc[:, 'y'].pct_change(1).shift(-1).fillna(0)
    X = df.copy()
    y_ = pd.DataFrame(y.map(f), columns=['y'])
    df_ = pd.concat([X, y_], axis=1)

    # check the shape
    print('----------------------------------------------------------------------------------------')
    print('X shape: (%i,%i)' % X.shape)
    print('y shape: (%i,%i)' % y_.shape)
    print('----------------------------------------------------------------------------------------')
    print(y_.groupby('y').size())
    print('y=1 up, y=0 stay, y=-1 down')
    print('----------------------------------------------------------------------------------------')

    # feature calculation
    open = pd.Series(df['open'])
    high = pd.Series(df['high'])
    low = pd.Series(df['low'])
    close = pd.Series(df['close'])
    volume = pd.Series(df['volume'])

    # pct_change for new column
    X['diff'] = y

    # Exponential Moving Average
    ema = talib.EMA(close, timeperiod=3)
    ema = ema.fillna(ema.mean())

    # Momentum
    momentum = talib.MOM(close, timeperiod=5)
    momentum = momentum.fillna(momentum.mean())

    # RSI
    rsi = talib.RSI(close, timeperiod=14)
    rsi = rsi.fillna(rsi.mean())

    # ADX
    adx = talib.ADX(high, low, close, timeperiod=14)
    adx = adx.fillna(adx.mean())

    # ADX change
    adx_change = adx.pct_change(1).shift(-1)
    adx_change = adx_change.fillna(adx_change.mean())

    # AD
    ad = talib.AD(high, low, close, volume)
    ad = ad.fillna(ad.mean())

    X_ = pd.concat([X, ema, momentum, rsi, adx_change, ad], axis=1).drop(['open', 'high', 'low', 'close'], axis=1)
    X_.columns = ['volume','diff', 'ema', 'momentum', 'rsi', 'adx', 'ad']
    X_.join(y_).head(10)

    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.33, random_state=42)
    print('X_train shape: {}'.format(X_train.shape))
    print('X_test shape: {}'.format(X_test.shape))
    print('y_train shape: {}'.format(y_train.shape))
    print('y_test shape: {}'.format(y_test.shape))

    pipe_knn = Pipeline([('scl', StandardScaler()), ('est', KNeighborsClassifier(n_neighbors=3))])
    pipe_logistic = Pipeline([('scl', StandardScaler()), ('est', LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=39))])
    pipe_rf = Pipeline([('scl', StandardScaler()), ('est', RandomForestClassifier(random_state=39))])
    pipe_gb = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingClassifier(random_state=39))])

    pipe_names = ['KNN','Logistic','RandomForest','GradientBoosting']
    pipe_lines = [pipe_knn, pipe_logistic, pipe_rf, pipe_gb]

    for (i, pipe) in enumerate(pipe_lines):
        pipe.fit(X_train, y_train.values.ravel())
        print('%s: %.3f' % (pipe_names[i] + ' Train Accuracy', accuracy_score(y_train.values.ravel(), pipe.predict(X_train))))
        print('%s: %.3f' % (pipe_names[i] + ' Test Accuracy', accuracy_score(y_test.values.ravel(), pipe.predict(X_test))))
        print('%s: %.3f' % (pipe_names[i] + ' Train F1 Score', f1_score(y_train.values.ravel(), pipe.predict(X_train), average='micro')))
        print('%s: %.3f' % (pipe_names[i] + ' Test F1 Score', f1_score(y_test.values.ravel(), pipe.predict(X_test), average='micro')))

    for (i, pipe) in enumerate(pipe_lines):
        predict = pipe.predict(X_test)
        cm = confusion_matrix(y_test.values.ravel(), predict, labels=[-1, 0, 1])
        print('{} Confusion Matrix'.format(pipe_names[i]))
        print(cm)

    cv = cross_val_score(pipe_gb, X_, y_.values.ravel(), cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=39))
    print('Cross Validation with StatifiedKFold mean: {}'.format(cv.mean()))

if __name__ == '__main__':
    main()

