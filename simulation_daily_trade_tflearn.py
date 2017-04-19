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
import glob


class Simulation:
    def __init__(self):
        self.len_past = 30
        #self.s_date = "20120101_20160330"
        #self.model_dir = '../model/tflearn/reg_l3_bn/big/%s/' % self.s_date

        tf.reset_default_graph()
        tflearn.init_graph(gpu_memory_fraction=0.05)
        input_layer = tflearn.input_data(shape=[None, 690], name='input')
        dense1 = tflearn.fully_connected(input_layer, 400, name='dense1', activation='relu')
        dense1n = tflearn.batch_normalization(dense1, name='BN1')
        dense2 = tflearn.fully_connected(dense1n, 100, name='dense2', activation='relu')
        dense2n = tflearn.batch_normalization(dense2, name='BN2')
        dense3 = tflearn.fully_connected(dense2n, 1, name='dense3')
        output = tflearn.single_unit(dense3)
        regression = tflearn.regression(output, optimizer='adam', loss='mean_square',
                                metric='R2', learning_rate=0.001)
        self.estimators = tflearn.DNN(regression)
        self.qty = {}
        self.day_last = {}
        self.currency = 100000000

    def load_scaler(self):
        model_name = "../model/tflearn/reg_l3_bn/big/%s/scaler.pkl" % self.s_date
        self.scaler = joblib.load(model_name)

    def make_x(self, data, code):
        data_x = []
        days = []
        days = data.index[:]
        data.loc[:, 'month'] = data.loc[:, '일자']%10000/100
        data = data.drop(['일자', '체결강도'], axis=1)

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
        df = pd.read_hdf('../data/hdf/%s.hdf'%code, 'day').sort_index()
        data = df.loc[df.index > int(begin_date)]
        data = data.loc[data.index < int(end_date)]
        data = data.reset_index()
        data_x, days = self.make_x(data, code)
        assert len(data_x) == len(days)
        return data_x, days

    def load_model(self):
        self.estimators.load('%s/model.tfl' % self.model_dir)

    def predict(self, X_data):
        return self.estimators.predict(X_data)

    def simulation_daily_trade(self, code, start_date, end_date):
        X_data, days = self.load_data(code, start_date, end_date)
        if len(X_data) == 0: return 0, 0, 0
        if code not in self.qty:
            self.qty[code] = 0
            self.day_last[code] = 0
        account_balance = 0
        stock_balance = 0
        pred_list = self.predict(X_data)
        total_day_last = 0
        for idx in range(len(X_data)-1):
            pred = pred_list[idx]
            cur_price = X_data[idx][29*23]
            cur_volume = X_data[idx][29*23+1]
            buying_price = X_data[idx+1][29*23+3]
            #print("buying_price: %f" % buying_price)
            pred_transform = self.scaler[code].inverse_transform([pred] + [0]*22)[0]
            cur_real_price = self.scaler[code].inverse_transform([cur_price] + [0]*22)[0]
            cur_real_volume = self.scaler[code].inverse_transform([0] + [cur_volume] + [0]*21)[1]
            #print([0]*3 + [buying_price] + [0]*19)
            buying_real_price = self.scaler[code].inverse_transform([0]*3 + [buying_price] + [0]*19)[3]
            #print(pred, cur_price)
            self.day_last[code] += 1
            if pred_transform > 1.1*cur_real_price and self.qty[code] == 0 and cur_real_price*cur_real_volume > 1000000000:
                self.day_last[code] = 0
                unit_buy = self.currency * 0.02
                self.qty[code] = (unit_buy / buying_real_price + 1)
                account_balance -= buying_real_price * self.qty[code]
                self.currency -= buying_real_price * self.qty[code]
                #print("pred: %.2f, %d, cur: %.2f, %d" % (pred, pred_transform, cur_price, cur_real_price))
                #print("[BUY] balance: %d, price: %d qty: %d" % (account_balance, buying_real_price, qty))
            elif self.day_last[code] >= 5 and self.qty[code] > 0 and False:
                account_balance += 0.995 * buying_real_price * self.qty[code]
                self.currency += 0.995 * buying_real_price * self.qty[code]
                self.qty[code] = 0
                #print("[SELL] balance: %d, price: %d, qty: %d" % (account_balance, buying_real_price, qty))
            elif pred < cur_price and self.qty[code] > 0:
                account_balance += 0.995 * buying_real_price * self.qty[code]
                self.currency += 0.995 * buying_real_price * self.qty[code]
                self.qty[code] = 0
                #print("[SELL] balance: %d, price: %d, qty: %d, day_last: %d" % (account_balance, buying_real_price, qty, day_last))
                total_day_last += self.day_last[code]
        if self.qty[code] > 0:
            stock_balance = 0.995 * buying_real_price * self.qty[code]
        else:
            stock_balance = 0
            #print("[L SELL] balance: %d, price: %d, qty: %d" % (account_balance, buying_real_price, qty))
        return account_balance, total_day_last, stock_balance

    def simulation_monthly_daily_trade(self, start_date, end_date):
        code_list = glob.glob('../data/hdf/*.hdf')
        code_list = list(map(lambda x: x.split('.hdf')[0][-6:], code_list))
        account_balance = 0
        idx = 0
        trade = 0
        day_last = 0
        stock_balance = 0
        for code in code_list:
            res, dl, sb = self.simulation_daily_trade(code, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
            idx += 1
            day_last += dl
            stock_balance += sb
            if res != 0:
                trade += 1
                account_balance += res
                print("[%d/%d] balance: %d (adl: %.1f, sb: %d)" % (trade, idx, account_balance, day_last/trade, stock_balance))
        return account_balance

    def simulation_all(self):
        begin_month = 201511
        res = 0
        while begin_month <= 201701:
            self.s_date = '%d01_%d01'%(begin_month-500, begin_month)
            self.model_dir = '../model/tflearn/reg_l3_bn/big/%s/' % (self.s_date)
            print(self.model_dir)
            self.load_model()
            begin_date = datetime.date(int(begin_month/100), begin_month%100, 1) - datetime.timedelta(days=40)
            end_date = datetime.date(int(begin_month/100), begin_month%100, 1) + datetime.timedelta(days=40)
            res += self.simulation_monthly_daily_trade(begin_date, end_date)
            print("[%d]total res: %d" % (begin_month, res))
            begin_month += 1
            if begin_month%100 == 13:
                begin_month += 88


if __name__ == '__main__':
    sm = Simulation()
    sm.simulation_all()
