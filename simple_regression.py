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
from tensorflow.python.ops import control_flow_ops 
tf.python.control_flow_ops = control_flow_ops

MODEL_TYPE = 'keras'
def baseline_model():
    # create model
    model = Sequential()
    #model.add(Dense(128, input_dim=174, init='he_normal', activation='relu'))
    model.add(Dense(128, input_dim=690, init='he_normal', activation='relu'))
    #model.add(normalization.BatchNormalization(epsilon=0.001, mode=0, axis=-1, momentum=0.99, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None))

    #model.add(Dropout(0.1))
    #model.add(Dense(128, init='he_normal'))
    model.add(Dense(1, init='he_normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


class SimpleModel:
    def __init__(self):
        self.data = dict()
        self.frame_len = 30
        self.predict_dist = 5
        self.scaler = dict()

    def load_all_data(self, begin_date, end_date):
        con = sqlite3.connect('../data/stock.db')
        code_list = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        X_data_list, Y_data_list, DATA_list = [0]*10, [0]*10, [0]*10
        idx = 0
        split = int(len(code_list) / 9)
        for code in code_list:
            data = self.load_data(code[0], begin_date, end_date)
            data = data.dropna()
            X, Y = self.make_x_y(data, code[0])
            if len(X) <= 1: continue
            code_array = [code[0]] * len(X)
            assert len(X) == len(data.loc[29:len(data)-6, '일자'])
            if idx%split == 0:
                X_data_list[int(idx/split)] = list(X)
                Y_data_list[int(idx/split)] = list(Y)
                DATA_list[int(idx/split)] = np.array([data.loc[29:len(data)-6, '일자'].values.tolist(), code_array, data.loc[29:len(data)-6, '현재가'], data.loc[34:len(data), '현재가']]).T.tolist()
            else:
                X_data_list[int(idx/split)].extend(X)
                Y_data_list[int(idx/split)].extend(Y)
                DATA_list[int(idx/split)].extend(np.array([data.loc[29:len(data)-6, '일자'].values.tolist(), code_array, data.loc[29:len(data)-6, '현재가'], data.loc[34:len(data), '현재가']]).T.tolist())
            #print(int(idx/split), np.shape(DATA_list[int(idx/split)]), np.shape(DATA_list[int(idx/split)][0]), np.shape(DATA_list[int(idx/split)][1]), np.shape(DATA_list[int(idx/split)][2]), np.shape(DATA_list[int(idx/split)][3]))
            print(idx, np.shape(X_data_list[int(idx/split)]), np.shape(Y_data_list[int(idx/split)]))
            idx += 1
        for i in range(10):
            if type(X_data_list[i]) == type(1):
                print("type is int, need to pass", X_data_list[i])
                continue
            if i == 0:
                X_data = X_data_list[i]
                Y_data = Y_data_list[i]
                DATA = DATA_list[i]
            else:
                X_data.extend(X_data_list[i])
                Y_data.extend(Y_data_list[i])
                DATA.extend(DATA_list[i])
            print("DATA", np.shape(DATA), np.shape(DATA[0]))
        return np.array(X_data), np.array(Y_data), np.array(DATA)

    def load_data(self, code, begin_date, end_date):
        con = sqlite3.connect('../data/stock.db')
        df = pd.read_sql("SELECT * from '%s'" % code, con, index_col='일자').sort_index()
        data = df.loc[df.index > str(begin_date)]
        data = data.loc[data.index < str(end_date)]
        data = data.reset_index()
        return data

    def make_x_y(self, data, code):
        data_x = []
        data_y = []
        for col in data.columns:
            try:
                data.loc[:, col] = data.loc[:, col].str.replace('--', '-')
                data.loc[:, col] = data.loc[:, col].str.replace('+', '')
            except AttributeError as e:
                pass
                print(e)
        data.loc[:, 'month'] = data.loc[:, '일자'].str[4:6]
        data = data.drop(['일자', '체결강도'], axis=1)

        # normalization
        data = np.array(data)
        if len(data) <= 0 :
            return np.array([]), np.array([])

        if code not in self.scaler:
            self.scaler[code] = StandardScaler()
            data = self.scaler[code].fit_transform(data)
        elif code not in self.scaler:
            return np.array([]), np.array([])
        else:
            data = self.scaler[code].transform(data)

        for i in range(self.frame_len, len(data)-self.predict_dist+1):
            data_x.extend(np.array(data[i-self.frame_len:i, :]))
            data_y.append(data[i+self.predict_dist-1][0])
        np_x = np.array(data_x).reshape(-1, 23*30)
        np_y = np.array(data_y)
        return np_x, np_y

    def train_model(self, X_train, Y_train):
        print("training model %d_%d.pkl" % (self.frame_len, self.predict_dist))
        model_name = "../model/simple_reg_model/%d_%d.pkl" % (self.frame_len, self.predict_dist)
        self.estimator = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
        self.estimator.fit(X_train, Y_train)
        print("finish training model")
        joblib.dump(self.estimator, model_name)

    def set_config(self):
        #Tensorflow GPU optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

    def train_model_keras(self, X_train, Y_train, date):
        print("training model %d_%d.h5" % (self.frame_len, self.predict_dist))
        model_name = "../model/reg_keras/%d_%d_%s.h5" % (self.frame_len, self.predict_dist, date)
        self.estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=200, batch_size=64, verbose=1)
        self.estimator.fit(X_train, Y_train)
        print("finish training model")
        # saving model
        json_model = self.estimator.model.to_json()
        open(model_name.replace('h5', 'json'), 'w').write(json_model)
        self.estimator.model.save_weights(model_name, overwrite=True)

    def evaluate_model(self, X_test, Y_test, orig_data, s_date):
        print("Evaluate model %d_%d.pkl" % (self.frame_len, self.predict_dist))
        if MODEL_TYPE == 'random_forest':
            model_name = "../model/simple_reg_model/%d_%d.pkl" % (self.frame_len, self.predict_dist)
            self.estimator = joblib.load(model_name)
        elif MODEL_TYPE == 'keras':
            model_name = "../model/reg_keras/%d_%d_%s.h5" % (self.frame_len, self.predict_dist, s_date)
            self.estimator = model_from_json(open(model_name.replace('h5', 'json')).read())
            self.estimator.load_weights(model_name)
        pred = self.estimator.predict(X_test)
        res = 0
        score = 0
        assert(len(pred) == len(Y_test))
        pred = np.array(pred).reshape(-1)
        Y_test = np.array(Y_test).reshape(-1)
        for i in range(len(pred)):
            score += (float(pred[i]) - float(Y_test[i]))*(float(pred[i]) - float(Y_test[i]))
        score = np.sqrt(score/len(pred))
        print("score: %f" % score)
        for idx in range(len(pred)):
            print(orig_data[idx])
            print(X_test[idx][23*29], pred[idx])
            buy_price = int(orig_data[idx][2])
            future_price = int(orig_data[idx][3])
            date = int(orig_data[idx][0])
            pred_transform = self.scaler[orig_data[idx][1]].inverse_transform([pred[idx]] + [0]*22)[0]
            cur_transform = self.scaler[orig_data[idx][1]].inverse_transform([X_test[idx][23*29]] + [0]*22)[0]
            print("cur price: %d, transformed_price: %d" % (buy_price, cur_transform))
            if pred_transform > buy_price * 1.01:
                res += (future_price - buy_price*1.005)*(100000/buy_price+1)
                print("[%s] buy: %6d, sell: %6d, earn: %6d" % (str(date), buy_price, future_price, (future_price - buy_price*1.005)*(100000/buy_price)))
        print("result: %d" % res)

    def load_current_data(self):
        con = sqlite3.connect('../data/stock.db')
        code_list = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        X_test = []
        DATA = []
        code_list = list(map(lambda x: x[0], code_list))
        first = True
        for code in code_list:
            df = pd.read_sql("SELECT * from '%s'" % code, con, index_col='일자').sort_index()
            data = df.iloc[-30:,:]
            data = data.reset_index()
            for col in data.columns:
                try:
                    data.loc[:, col] = data.loc[:, col].str.replace('--', '-')
                    data.loc[:, col] = data.loc[:, col].str.replace('+', '')
                except AttributeError as e:
                    pass
                    print(e)
            data.loc[:, 'month'] = data.loc[:, '일자'].str[4:6]
            data = data.drop(['일자', '체결강도'], axis=1)
            if len(data) < 30:
                code_list.remove(code)
                continue
            DATA.append(int(data.loc[len(data)-1, '현재가']))
            try:
                data = self.scaler[code].transform(np.array(data))
            except KeyError:
                code_list.remove(code)
                continue
            X_test.extend(np.array(data))
            print(np.shape(X_test))
        X_test = np.array(X_test).reshape(-1, 23*30) 
        return X_test, code_list, DATA

    def make_buy_list(self, X_test, code_list, orig_data, s_date):
        BUY_UNIT = 10000
        print("make buy_list")
        if MODEL_TYPE == 'random_forest':
            model_name = "../model/simple_reg_model/%d_%d.pkl" % (self.frame_len, self.predict_dist)
            self.estimator = joblib.load(model_name)
        elif MODEL_TYPE == 'keras':
            model_name = "../model/reg_keras/%d_%d_%s.h5" % (self.frame_len, self.predict_dist, s_date)
            self.estimator = model_from_json(open(model_name.replace('h5', 'json')).read())
            self.estimator.load_weights(model_name)
        pred = self.estimator.predict(X_test)
        res = 0
        score = 0
        pred = np.array(pred).reshape(-1)

        buy_item = ["매수", "", "시장가", 0, 0, "매수전"]  # 매수/매도, code, 시장가/현재가, qty, price, "주문전/주문완료"
        with open("../data/buy_list.txt", "wt") as f_buy:
            for idx in range(len(pred)):
                real_buy_price = int(orig_data[idx])
                buy_price = float(X_test[idx][23*29])
                pred_transform = self.scaler[orig_data[idx][1]].inverse_transform([pred[idx]] + [0]*22)[0]
                print("[BUY?] code: %s, cur: %fd, predict: %d" % (code_list[idx], real_buy_price, pred_transform))
                if pred_transform > real_buy_price * 1.01:
                    print("add to buy_list %d")
                    buy_item[1] = code_list[idx]
                    buy_item[3] = int(BUY_UNIT / real_buy_price) + 1
                    for item in buy_item:
                        f_buy.write("%s;"%str(item))
                    f_buy.write('\n')

    def load_data_in_account(self):
        # load code list from account
        DATA = []
        with open('../data/stocks_in_account.txt') as f_stocks:
            for line in f_stocks.readlines():
                data = line.split(',')
                DATA.append([data[6].replace('A', ''), data[1], data[0]])

        # load data in DATA
        con = sqlite3.connect('../data/stock.db')
        X_test = []
        idx_rm = []
        first = True
        for idx, code in enumerate(DATA):
            print(len(X_test)/30)
            print(len(DATA) - len(idx_rm))

            try:
                df = pd.read_sql("SELECT * from '%s'" % code[0], con, index_col='일자').sort_index()
            except pd.io.sql.DatabaseError as e:
                print(e)
                idx_rm.append(idx)
                continue
            data = df.iloc[-30:,:]
            data = data.reset_index()
            for col in data.columns:
                try:
                    data.loc[:, col] = data.loc[:, col].str.replace('--', '-')
                    data.loc[:, col] = data.loc[:, col].str.replace('+', '')
                except AttributeError as e:
                    pass
                    print(e)
            data.loc[:, 'month'] = data.loc[:, '일자'].str[4:6]
            DATA[idx].append(int(data.loc[len(data)-1, '현재가']))
            data = data.drop(['일자', '체결강도'], axis=1)
            if len(data) < 30:
                idx_rm.append(idx)
                continue
            try:
                data = self.scaler[code[0]].transform(np.array(data))
            except KeyError:
                idx_rm.append(idx)
                continue
            X_test.extend(np.array(data))
            print(np.shape(X_test))
        for i in idx_rm[-1:0:-1]:
            del DATA[i]
        X_test = np.array(X_test).reshape(-1, 23*30) 
        return X_test, DATA

    def make_sell_list(self, X_test, DATA, s_date):
        print("make sell_list")
        if MODEL_TYPE == 'random_forest':
            model_name = "../model/simple_reg_model/%d_%d.pkl" % (self.frame_len, self.predict_dist)
            self.estimator = joblib.load(model_name)
        elif MODEL_TYPE == 'keras':
            model_name = "../model/reg_keras/%d_%d_%s.h5" % (self.frame_len, self.predict_dist, s_date)
            self.estimator = model_from_json(open(model_name.replace('h5', 'json')).read())
            self.estimator.load_weights(model_name)
        pred = self.estimator.predict(X_test)
        res = 0
        score = 0
        pred = np.array(pred).reshape(-1)

        sell_item = ["매도", "", "시장가", 0, 0, "매도전"]  # 매수/매도, code, 시장가/현재가, qty, price, "주문전/주문완료"
        with open("../data/sell_list.txt", "wt") as f_sell:
            for idx in range(len(pred)):
                current_price = float(X_test[idx][23*29])
                current_real_price = int(DATA[idx][3])
                name = DATA[idx][2]
                print("[SELL?] name: %s, code: %s, cur: %f(%d), predict: %f" % (name, DATA[idx][0], current_price, current_real_price, pred[idx]))
                if pred[idx] < current_price:
                    print("add to sell_list %s" % name)
                    sell_item[1] = DATA[idx][0]
                    sell_item[3] = DATA[idx][1]
                    for item in sell_item:
                        f_sell.write("%s;"%str(item))
                    f_sell.write('\n')
    def save_scaler(self, s_date):
        model_name = "../model/scaler_%s.pkl" % s_date
        joblib.dump(self.scaler, model_name)

    def load_scaler(self, s_date):
        model_name = "../model/scaler_%s.pkl" % s_date
        self.scaler = joblib.load(model_name)


if __name__ == '__main__':
    sm = SimpleModel()
    sm.set_config()
    X_train, Y_train, _ = sm.load_all_data(20110101, 20170307)
    sm.train_model_keras(X_train, Y_train, "20110101_20170307")
    sm.save_scaler("20110101_20170307")
    #sm.load_scaler("20110101_20160331")
    #X_test, Y_test, Data = sm.load_all_data(20160303, 20160430)
    #sm.evaluate_model(X_test, Y_test, Data, "20110101_20160331")

    #X_data, code_list, data = sm.load_current_data()
    #sm.make_buy_list(X_data, code_list, data, "20110101_20170307")
    #X_data, data = sm.load_data_in_account()
    #sm.make_sell_list(X_data, data, "20110101_20170307")
