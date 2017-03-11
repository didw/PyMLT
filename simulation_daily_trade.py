# -*- encoding: utf-8 -*-
import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.externals import joblib
from keras.models import Sequential
from keras.layers import Dense, Dropout, normalization
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import model_from_json
from keras import backend as K
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os


def load_model():
    model_name = "../model/reg_keras/30_5_20110101_20160331.h5"
    estimator = model_from_json(open(model_name.replace('h5', 'json')).read())
    estimator.load_weights(model_name)
    return estimator


def load_data(code, begin_date, end_date):
    con = sqlite3.connect('../data/stock.db')
    df = pd.read_sql("SELECT * from '%s'" % code, con, index_col='일자').sort_index()
    data = df.loc[df.index > str(begin_date)]
    data = data.loc[data.index < str(end_date)]
    for col in data.columns:
        try:
            data.loc[:, col] = data.loc[:, col].str.replace('--', '-')
            data.loc[:, col] = data.loc[:, col].str.replace('+', '')
        except AttributeError as e:
            pass
            print(e)
    data = data.reset_index()
    return data

def load_scaler(s_date):
    model_name = "../model/scaler_%s.pkl" % s_date
    return joblib.load(model_name)

def simulation_daily_trade(estimator, code, start_date, end_date):
    code_data = load_data(code, start_date, end_date)
    LEN_PAST = 30
    MONEY = 100000
    qty = 0
    account_balance = 0
    scaler = load_scaler("20110101_20160331")
    print(len(code_data))
    for idx in range(LEN_PAST, len(code_data)-1):
        X_data = code_data[idx - LEN_PAST: idx]
        X_data = X_data.reset_index()
        cur_real_price = int(X_data.loc[29, '현재가'])
        trade_price = int(code_data.loc[idx, '시가'])
        X_data.loc[:, 'month'] = X_data.loc[:, '일자'].str[4:6]
        X_data = X_data.drop(['index', '일자', '체결강도'], axis=1)
        X_data = scaler[code[0]].transform(X_data)
        cur_price = X_data[29][0]
        X_data = np.array(X_data).reshape(-1, 30*23)
        pred = estimator.predict(X_data)[0][0]
        pred_transform = scaler[code[0]].inverse_transform([pred] + [0]*22)[0]
        #print(pred, cur_price)
        if pred_transform > 2*cur_real_price and qty == 0:
            qty += (MONEY / cur_real_price + 1)
            account_balance -= cur_real_price * (MONEY / cur_real_price + 1)
            print("[BUY] balance: %d, price: %d qty: %d" % (account_balance, cur_real_price, qty))
        if pred < cur_price and qty > 0:
            account_balance += 0.995 * cur_real_price * qty
            qty = 0
            print("[SELL] balance: %d, price: %d, qty: %d" % (account_balance, cur_real_price, qty))
    if qty > 0:
        account_balance += 0.995 * cur_real_price * qty
        print("[L SELL] balance: %d, price: %d, qty: %d" % (account_balance, cur_real_price, qty))
    return account_balance

def simulation_all_daily_trade(start_date, end_date):
    estimator = load_model()
    con = sqlite3.connect('../data/stock.db')
    code_list = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    account_balance = 0
    for code in code_list:
        account_balance += simulation_daily_trade(estimator, code, start_date, end_date)
        print("balance: %d" % account_balance)

def set_config():
    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)


if __name__ == '__main__':
    set_config()
    simulation_all_daily_trade('20160301', '20160501')

