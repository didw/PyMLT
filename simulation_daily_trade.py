# -*- encoding: utf-8 -*-
import pandas as pd
import numpy as np
import sqlite3
#from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.externals import joblib
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, normalization
#from keras.wrappers.scikit_learn import KerasRegressor
#from keras.models import model_from_json
#from keras import backend as K
from sklearn.preprocessing import StandardScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf


class Simulation:
    def __init__(self):
        self.len_past = 30
        init_op = tf.global_variables_initializer()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init_op = tf.global_variables_initializer()
        self.s_date = "20120101_20160330"
        self.model_dir = '../model/tf/regression/%s/' % self.s_date

        # define variable
        self.W1 = tf.Variable(tf.random_normal([690, 200], stddev=0.35), name="W1")
        self.b1 = tf.Variable(tf.zeros([200]), name="b1")
        self.W2 = tf.Variable(tf.random_normal([200, 1], stddev=0.35), name="W2")
        self.b2 = tf.Variable(tf.zeros([1]), name="b2")

        # design graph
        self.scalarInput =  tf.placeholder(shape=[None,690],dtype=tf.float32)
        self.out1 = tf.matmul(self.scalarInput, self.W1) + self.b1
        self.stream1 = tf.layers.dropout(tf.nn.relu(self.out1), rate=0.5)
        #self.output = tf.layers.dense(self.stream1, 1)
        self.output = tf.matmul(self.stream1, self.W2) + self.b2

        self.saver = tf.train.Saver([self.W1, self.b1, self.W2, self.b2])

    def load_scaler(self):
        model_name = "../model/scaler_%s.pkl" % self.s_date
        self.scaler = joblib.load(model_name)

    def make_x(self, data, code):
        data_x = []
        days = []
        for col in data.columns:
            try:
                data.loc[:, col] = data.loc[:, col].str.replace('--', '-')
                data.loc[:, col] = data.loc[:, col].str.replace('+', '')
            except AttributeError as e:
                pass
                print(e)
        days = data.index[:]
        try:
            data.loc[:, 'month'] = data.index[:].str[4:6]
        except AttributeError as e:
            print(e)
            print(data.index[:])
        data = data.drop(['체결강도'], axis=1)

        # normalization
        data = np.array(data)
        if len(data) <= 0 :
            return np.array([]), np.array([])

        self.load_scaler()
        if code not in self.scaler:
            print("code %s is not exist in scaler" % code)
            return np.array([]), np.array([])
        else:
            data = self.scaler[code].transform(data)

        for i in range(self.len_past, len(data)):
            data_x.extend(np.array(data[i-self.len_past:i, :]))
        np_x = np.array(data_x).reshape(-1, 23*30)
        return np_x, days[self.len_past:]


    def load_data(self, code, begin_date, end_date):
        con = sqlite3.connect('../data/stock.db')
        df = pd.read_sql("SELECT * from '%s'" % code, con, index_col='일자').sort_index()
        data = df.loc[df.index > str(begin_date)]
        data = data.loc[data.index < str(end_date)]
        data_x, days = self.make_x(data, code)
        assert len(data_x) == len(days)
        return data_x, days

    def predict(self, X_data):
        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            return sess.run(self.output, feed_dict={self.scalarInput: X_data})

    def simulation_daily_trade(self, code, start_date, end_date):
        X_data, days = self.load_data(code[0], start_date, end_date)
        if len(X_data) == 0: return 0
        MONEY = 100000
        qty = 0
        account_balance = 0
        pred_list = self.predict(X_data)
        for idx in range(len(X_data)):
            pred = pred_list[idx][0]
            cur_price = X_data[idx][29*23]
            pred_transform = self.scaler[code[0]].inverse_transform([pred] + [0]*22)[0]
            cur_real_price = self.scaler[code[0]].inverse_transform([cur_price] + [0]*22)[0]
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

    def simulation_all_daily_trade(self, start_date, end_date):
        con = sqlite3.connect('../data/stock.db')
        code_list = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        account_balance = 0
        idx = 0
        for code in code_list:
            account_balance += self.simulation_daily_trade(code, start_date, end_date)
            idx += 1
            print("[%d] balance: %d" % (idx, account_balance))


if __name__ == '__main__':
    sm = Simulation()
    sm.simulation_all_daily_trade('20160220', '20160501')

