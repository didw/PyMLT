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
    model_name = "../model/reg_keras/30_5.h5"
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

def simulation_daily_trade(estimator, code, start_date, end_date):
    code_data = load_data(code, start_date, end_date)
    LEN_PAST = 30
    LEN_PREDICT = 5
    MONEY = 100000
    bought = 0
    account_balance = 0
    for idx in range(LEN_PAST, len(code_data) - LEN_PREDICT):
        X_data = code_data[idx - LEN_PAST: idx]
        X_data = X_data.reset_index()
        cur_price = int(X_data.loc[29, '현재가'])
        X_data.loc[:, 'month'] = X_data.loc[:, '일자'].str[4:6]
        X_data = X_data.drop(['index', '일자', '체결강도'], axis=1)
        Y_data = code_data.loc[idx + LEN_PREDICT, '현재가']
        X_data = np.array(X_data).reshape(-1, 30*23)
        pred = estimator.predict(X_data)
        if int(pred) > cur_price*1.1:
            bought += (MONEY / cur_price + 1)
            account_balance -= cur_price * (MONEY / cur_price + 1)
        if pred < cur_price and bought > 0:
            account_balance += 0.95 * cur_price * bought
            bought = 0
        #print("balance: %d, bought: %d" % (account_balance, bought))
    return account_balance

def simulation_all_daily_trade(start_date, end_date):
    estimator = load_model()
    con = sqlite3.connect('../data/stock.db')
    code_list = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    account_balance = 0
    for code in code_list:
        account_balance += simulation_daily_trade(estimator, code, start_date, end_date)
        print("balance: %d" % account_balance)


if __name__ == '__main__':
    simulation_all_daily_trade('20160101', '20170101')

