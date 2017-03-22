# -*- encoding: utf-8 -*-
import pandas as pd
import numpy as np
import sqlite3
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import datetime
import tflearn


class Simulation:
    def __init__(self):
        self.len_past = 30
        init_op = tf.global_variables_initializer()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.init_op = tf.global_variables_initializer()
        self.s_date = "20120101_20160330"
        self.model_dir = '../model/tflearn/regression/%s/' % self.s_date

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
        days = data.index[:]
        try:
            data.loc[:, 'month'] = data.index[:].str[4:6]
        except AttributeError as e:
            pass
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
        tf.reset_default_graph()
        tflearn.init_graph(gpu_memory_fraction=0.1)
        input_layer = tflearn.input_data(shape=[None, 690], name='input')
        dense1 = tflearn.fully_connected(input_layer, 128, name='dense1', activation='relu')
        dense2 = tflearn.fully_connected(dense1, 1, name='dense2')
        output = tflearn.single_unit(dense2)
        regression = tflearn.regression(output, optimizer='adam', loss='mean_square',
                                metric='R2', learning_rate=0.001)
        estimators = tflearn.DNN(regression)

        estimators.load('%s/model.tfl' % self.model_dir)
        return estimators.predict(X_data)

    def simulation_daily_trade(self, code, start_date, end_date):
        X_data, days = self.load_data(code[0], start_date, end_date)
        if len(X_data) == 0: return 0
        MONEY = 100000
        qty = 0
        account_balance = 0
        day_last = 0
        pred_list = self.predict(X_data)
        for idx in range(len(X_data)-1):
            pred = pred_list[idx][0]
            cur_price = X_data[idx][29*23]
            cur_volume = X_data[idx][29*23+1]
            buying_price = X_data[idx+1][29*23+3]
            #print("buying_price: %f" % buying_price)
            pred_transform = self.scaler[code[0]].inverse_transform([pred] + [0]*22)[0]
            cur_real_price = self.scaler[code[0]].inverse_transform([cur_price] + [0]*22)[0]
            cur_real_volume = self.scaler[code[0]].inverse_transform([0] + [cur_volume] + [0]*21)[0]
            #print([0]*3 + [buying_price] + [0]*19)
            buying_real_price = self.scaler[code[0]].inverse_transform([0]*3 + [buying_price] + [0]*19)[3]
            #print(pred, cur_price)
            day_last += 1
            if pred_transform > 2*cur_real_price and qty == 0 and cur_real_price*cur_real_volume > 1000000000:
                day_last = 0
                qty += (MONEY / buying_real_price + 1)
                account_balance -= buying_real_price * (MONEY / buying_real_price + 1)
                #print("pred: %.2f, %d, cur: %.2f, %d" % (pred, pred_transform, cur_price, cur_real_price))
                print("[BUY] balance: %d, price: %d qty: %d" % (account_balance, buying_real_price, qty))
            elif day_last >= 5 and qty > 0 and False:
                account_balance += 0.995 * buying_real_price * qty
                qty = 0
                print("[SELL] balance: %d, price: %d, qty: %d" % (account_balance, buying_real_price, qty))
            elif pred < cur_price and qty > 0:
                account_balance += 0.995 * buying_real_price * qty
                qty = 0
                print("[SELL] balance: %d, price: %d, qty: %d" % (account_balance, buying_real_price, qty))
        if qty > 0:
            account_balance += 0.995 * buying_real_price * qty
            print("[L SELL] balance: %d, price: %d, qty: %d" % (account_balance, buying_real_price, qty))
        return account_balance

    def simulation_monthly_daily_trade(self, start_date, end_date):
        con = sqlite3.connect('../data/stock.db')
        code_list = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        account_balance = 0
        idx = 0
        trade = 0
        for code in code_list:
            res = self.simulation_daily_trade(code, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
            idx += 1
            if res != 0:
                trade += 1
                account_balance += res
                print("[%d/%d] balance: %d" % (trade, idx, account_balance))
        return account_balance

    def simulation_all(self):
        begin_month = 201501
        res = 0
        while begin_month <= 201701:
            self.model_dir = '../model/tflearn/regression/%d01_%d01/' % (begin_month-500, begin_month)
            print(self.model_dir)
            begin_date = datetime.date(begin_month/100, begin_month%100, 1) - datetime.timedelta(days=40)
            end_date = datetime.date(begin_month/100, begin_month%100, 1) + datetime.timedelta(days=40)
            res += self.simulation_monthly_daily_trade(begin_date, end_date)
            print("[%d]total res: %d" % (begin_month, res))
            begin_month += 1
            if begin_month%100 == 13:
                begin_month += 88


if __name__ == '__main__':
    sm = Simulation()
    sm.simulation_all()
