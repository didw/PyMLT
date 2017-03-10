# -*- encoding: utf-8 -*-
from __future__ import print_function
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
import os, sys
from etaprogress.progress import ProgressBar


class TensorflowRegressor():
    def __init__(self, s_date):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        # Create two variables.
        tf.reset_default_graph()
        self.num_epoch = 30
        self.lr = tf.placeholder(dtype=tf.float32)
        self.W1 = tf.Variable(tf.random_normal([690, 200], stddev=0.35), name="W1")
        self.b1 = tf.Variable(tf.zeros([200]), name="b1")
        self.W2 = tf.Variable(tf.random_normal([200, 1], stddev=0.35), name="W2")
        self.b2 = tf.Variable(tf.zeros([1]), name="b2")

        self.scalarInput =  tf.placeholder(shape=[None,690],dtype=tf.float32)
        self.out1 = tf.matmul(self.scalarInput, self.W1) + self.b1
        self.stream1 = tf.layers.dropout(tf.nn.relu(self.out1), rate=0.5)
        #self.output = tf.layers.dense(self.stream1, 1)
        self.output = tf.matmul(self.stream1, self.W2) + self.b2
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.target = tf.placeholder(shape=[None],dtype=tf.float32)
        self.error = tf.square(self.target - self.output)
        self.loss = tf.reduce_mean(self.error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.updateModel = self.trainer.minimize(self.loss)
        self.saver = tf.train.Saver([self.W1, self.b1, self.W2, self.b2])
        self.model_dir = '../model/tf/regression/%s/' % s_date

    def fit(self, X_data, Y_data):
        # Add an op to initialize the variables.
        init_op = tf.global_variables_initializer()
        batch_size = 64

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        lr = 0.0005
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            for i in range(self.num_epoch):
                lr *= 0.9
                print("\nEpoch %d/%d is started" % (i+1, self.num_epoch), end='\n')
                bar = ProgressBar(len(X_data)/batch_size, max_width=80)
                for j in range(int(len(X_data)/batch_size)-1):
                    X_batch = X_data[batch_size*j:batch_size*(j+1)]
                    Y_batch = Y_data[batch_size*j:batch_size*(j+1)]
                    _ = sess.run(self.updateModel, feed_dict={self.lr:lr, self.scalarInput: X_batch, self.target: Y_batch})

                    if j%10 == 0:
                        loss = sess.run(self.loss, feed_dict={self.lr:lr, self.scalarInput: X_batch, self.target: Y_batch})
                        bar.numerator = j+1
                        print("%s | loss: %f" % (bar, loss), end='\r')
                        sys.stdout.flush()

            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            save_path = self.saver.save(sess,'%s/model.ckpt' % self.model_dir)
            print("Model saved in file: %s" % save_path)

    def predict(self, X_data):
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            ckpt = tf.train.get_checkpoint_state(self.model_dir)
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            return sess.run(self.output, feed_dict={self.scalarInput: X_data})


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
        bar = ProgressBar(len(code_list), max_width=80)
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
            bar.numerator += 1
            print("%s | %d" % (bar, len(X_data_list[int(idx/split)])), end='\r')
            sys.stdout.flush()
            idx += 1
        print("%s" % bar)

        print("Merge splited data")
        bar = ProgressBar(10, max_width=80)
        for i in range(10):
            if type(X_data_list[i]) == type(1):
                continue
            if i == 0:
                X_data = X_data_list[i]
                Y_data = Y_data_list[i]
                DATA = DATA_list[i]
            else:
                X_data.extend(X_data_list[i])
                Y_data.extend(Y_data_list[i])
                DATA.extend(DATA_list[i])
            bar.numerator = i+1
            print("%s | %d" % (bar, len(DATA)), end='\r')
            sys.stdout.flush()
        print("%s | %d" % (bar, len(DATA)))
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

    def set_config(self):
        #Tensorflow GPU optimization
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

    def train_model_tensorflow(self, X_train, Y_train, s_date):
        print("training model %s model.cptk" % s_date)
        #model = BaseModel()
        self.estimator = TensorflowRegressor(s_date)
        self.estimator.fit(X_train, Y_train)
        print("finish training model")

    def evaluate_model(self, X_test, Y_test, orig_data, s_date, fname=None):
        print("Evaluate model test.ckpt")
        self.estimator = TensorflowRegressor(s_date)
        pred = self.estimator.predict(X_test)
        score = 0
        ratio = [1, 1.01, 1.02, 1.05, 1.1, 1.5, 2, 2.5, 3]
        freq = [0]*len(ratio)
        res = [0]*len(ratio)
        date_min, date_max = 99999999, 0
        assert(len(pred) == len(Y_test))
        pred = np.array(pred).reshape(-1)
        Y_test = np.array(Y_test).reshape(-1)
        for i in range(len(pred)):
            score += (float(pred[i]) - float(Y_test[i]))*(float(pred[i]) - float(Y_test[i]))
        score = np.sqrt(score/len(pred))
        print("score: %f" % score)
        for idx in range(len(pred)):
            buy_price = int(orig_data[idx][2])
            future_price = int(orig_data[idx][3])
            date = int(orig_data[idx][0])
            date_min = min(date_min, date)
            date_max = max(date_max, date)
            pred_transform = self.scaler[orig_data[idx][1]].inverse_transform([pred[idx]] + [0]*22)[0]
            cur_transform = self.scaler[orig_data[idx][1]].inverse_transform([X_test[idx][23*29]] + [0]*22)[0]
            for j in range(len(ratio)):
                if pred_transform > buy_price * ratio[j]:
                    res[j] += (future_price - buy_price*1.005)*(100000/buy_price+1)
                    freq[j] += 1
                    print("[%s, %d] buy: %6d, sell: %6d, earn: %6d" % (str(date), freq[j], buy_price, future_price, (future_price - buy_price*1.005)*(100000/buy_price)))
        print("date length: %d - %d (%d)" % (date_min, date_max, int(len(pred)/2500)))
        for i in range(len(res)):
            if freq[i] == 0: continue
            print("%5d times trade, ratio: %1.2f, result: %8d (%4d)" %(freq[i], ratio[i], res[i], res[i]/freq[i]))
        if fname is not None:
            fout = open(fname, 'wt')
            fout.write("date length: %d - %d (%d)\n" % (date_min, date_max, int(len(pred)/2500)))
            for i in range(len(res)):
                if freq[i] == 0: continue
                fout.write("%5d times trade, ratio: %1.2f, result: %8d (%4d)\n" %(freq[i], ratio[i], res[i], res[i]/freq[i]))

    def load_current_data(self):
        con = sqlite3.connect('../data/stock.db')
        code_list = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        X_test = []
        DATA = []
        code_list = list(map(lambda x: x[0], code_list))
        first = True
        bar = ProgressBar(len(code_list), max_width=80)
        for code in code_list:
            bar.numerator += 1
            print("%s | %d" % (bar, len(X_test)), end='\r')
            sys.stdout.flush()
            df = pd.read_sql("SELECT * from '%s'" % code, con, index_col='일자').sort_index()
            data = df.iloc[-30:,:]
            data = data.reset_index()
            for col in data.columns:
                try:
                    data.loc[:, col] = data.loc[:, col].str.replace('--', '-')
                    data.loc[:, col] = data.loc[:, col].str.replace('+', '')
                except AttributeError as e:
                    pass
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
        X_test = np.array(X_test).reshape(-1, 23*30) 
        return X_test, code_list, DATA

    def make_buy_list(self, X_test, code_list, orig_data, s_date):
        BUY_UNIT = 10000
        print("make buy_list")
        self.estimator = TensorflowRegressor(s_date)
        pred = self.estimator.predict(X_test)
        res = 0
        score = 0
        pred = np.array(pred).reshape(-1)

        # load code list from account
        set_account = set([])
        with open('../data/stocks_in_account.txt') as f_stocks:
            for line in f_stocks.readlines():
                data = line.split(',')
                set_account.add(str(data[6].replace('A', '')))

        buy_item = ["매수", "", "시장가", 0, 0, "매수전"]  # 매수/매도, code, 시장가/현재가, qty, price, "주문전/주문완료"
        with open("../data/buy_list.txt", "wt") as f_buy:
            for idx in range(len(pred)):
                real_buy_price = int(orig_data[idx])
                buy_price = float(X_test[idx][23*29])
                try:
                    pred_transform = self.scaler[code_list[idx]].inverse_transform([pred[idx]] + [0]*22)[0]
                except KeyError:
                    continue
                print("[BUY PREDICT] code: %s, cur: %5d, predict: %5d" % (code_list[idx], real_buy_price, pred_transform))
                if pred_transform > real_buy_price * 1.1 and code_list[idx] not in set_account:
                    print("add to buy_list %s" % code_list[idx])
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
        bar = ProgressBar(len(DATA), max_width=80)
        for idx, code in enumerate(DATA):
            bar.numerator += 1
            print("%s | %d" % (bar, len(X_test)), end='\r')
            sys.stdout.flush()

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
        for i in idx_rm[-1:0:-1]:
            del DATA[i]
        X_test = np.array(X_test).reshape(-1, 23*30) 
        return X_test, DATA

    def make_sell_list(self, X_test, DATA, s_date):
        print("make sell_list")
        self.estimator = TensorflowRegressor(s_date)
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
                print("[SELL PREDICT] name: %s, code: %s, cur: %f(%d), predict: %f" % (name, DATA[idx][0], current_price, current_real_price, pred[idx]))
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
    X_train, Y_train, _ = sm.load_all_data(20120101, 20160730)
    sm.train_model_tensorflow(X_train, Y_train, "20120101_20160730")
    sm.save_scaler("20120101_20160730")
    sm.load_scaler("20120101_20160730")
    X_test, Y_test, Data = sm.load_all_data(20160620, 20160910)
    sm.evaluate_model(X_test, Y_test, Data, "20120101_20160730")

    #sm.load_scaler("20120101_20170309")
    #X_data, code_list, data = sm.load_current_data()
    #sm.make_buy_list(X_data, code_list, data, "20120101_20170309")
    #X_data, data = sm.load_data_in_account()
    #sm.make_sell_list(X_data, data, "20120101_20170309")
"""
result
1. DATA: 20120101_20160330
a. date length: 20160404 - 20160428 (16)
13672 times trade, ratio: 1.00, result:  8518917 ( 623)
10568 times trade, ratio: 1.01, result:  7399085 ( 700)
 8059 times trade, ratio: 1.02, result:  6774314 ( 840)
 4243 times trade, ratio: 1.05, result:  5188120 (1222)
 2079 times trade, ratio: 1.10, result:  3236100 (1556)
  298 times trade, ratio: 1.50, result:  -218290 (-732)
  125 times trade, ratio: 2.00, result:  -104943 (-839)
   95 times trade, ratio: 2.50, result:  -133305 (-1403)
   63 times trade, ratio: 3.00, result:   -43232 (-686)
b. date length: 20160502 - 20160601 (19)
19490 times trade, ratio: 1.00, result: -2425263 (-124)
15151 times trade, ratio: 1.01, result: -1000211 ( -66)
11791 times trade, ratio: 1.02, result:  -608315 ( -51)
 6262 times trade, ratio: 1.05, result:   732069 ( 116)
 2699 times trade, ratio: 1.10, result:   969196 ( 359)
  284 times trade, ratio: 1.50, result:   697218 (2454)
  134 times trade, ratio: 2.00, result:   766307 (5718)
   87 times trade, ratio: 2.50, result:   335627 (3857)
   69 times trade, ratio: 3.00, result:   246596 (3573)
c. date length: 20160603 - 20160701 (18)
19671 times trade, ratio: 1.00, result: -5165972 (-262)
15553 times trade, ratio: 1.01, result: -1869095 (-120)
12102 times trade, ratio: 1.02, result:   186830 (  15)
 6049 times trade, ratio: 1.05, result:  2628491 ( 434)
 2434 times trade, ratio: 1.10, result:  1216399 ( 499)
  218 times trade, ratio: 1.50, result:  -471260 (-2161)
  102 times trade, ratio: 2.00, result:  -140877 (-1381)
   67 times trade, ratio: 2.50, result:  -131047 (-1955)
   57 times trade, ratio: 3.00, result:  -161333 (-2830)


2. DATA: 20120101_20160430
a. date length: 20160502 - 20160601 (19)
20167 times trade, ratio: 1.00, result: -2006599 ( -99)
15018 times trade, ratio: 1.01, result:  -488002 ( -32)
11392 times trade, ratio: 1.02, result:   332280 (  29)
 5418 times trade, ratio: 1.05, result:  1042224 ( 192)
 2216 times trade, ratio: 1.10, result:   786968 ( 355)
  233 times trade, ratio: 1.50, result:   717990 (3081)
  112 times trade, ratio: 2.00, result:    66322 ( 592)
   76 times trade, ratio: 2.50, result:     8554 ( 112)
   55 times trade, ratio: 3.00, result:   -56765 (-1032)
b. date length: 20160603 - 20160701 (18)
20122 times trade, ratio: 1.00, result: -5786758 (-287)
15625 times trade, ratio: 1.01, result: -1162031 ( -74)
11871 times trade, ratio: 1.02, result:  2202977 ( 185)
 5518 times trade, ratio: 1.05, result:  3180208 ( 576)
 2174 times trade, ratio: 1.10, result:  1784537 ( 820)
  205 times trade, ratio: 1.50, result:  -280261 (-1367)
   93 times trade, ratio: 2.00, result:  -103007 (-1107)
   60 times trade, ratio: 2.50, result:  -168534 (-2808)
   43 times trade, ratio: 3.00, result:  -124300 (-2890)


2. DATA: 20120101_20160630
a. date length: 20160704 - 20160802 (20)
12811 times trade, ratio: 1.00, result:  5413983
 9150 times trade, ratio: 1.01, result:  4125016
 6666 times trade, ratio: 1.02, result:  3250145
 2825 times trade, ratio: 1.05, result:  1366772
  975 times trade, ratio: 1.10, result:  -141279
  105 times trade, ratio: 1.50, result:  -228538
   54 times trade, ratio: 2.00, result:   -93332
   43 times trade, ratio: 2.50, result:    -5653
   32 times trade, ratio: 3.00, result:   117652

b. date length: 20160801 - 20160902 (22)
14641 times trade, ratio: 1.00, result: -11762421 (-803)
10482 times trade, ratio: 1.01, result: -8508369 (-811)
 7801 times trade, ratio: 1.02, result: -6210787 (-796)
 3440 times trade, ratio: 1.05, result: -3305090 (-960)
 1183 times trade, ratio: 1.10, result: -1672804 (-1414)
  104 times trade, ratio: 1.50, result:  -149441 (-1436)
   57 times trade, ratio: 2.00, result:   -62254 (-1092)
   40 times trade, ratio: 2.50, result:    38726 ( 968)
   33 times trade, ratio: 3.00, result:    50581 (1532)

3. DATA: 20120101_20160730
date length: 20160801 - 20160902 (22)
19906 times trade, ratio: 1.00, result: -17320220 (-870)
14760 times trade, ratio: 1.01, result: -13056026 (-884)
11313 times trade, ratio: 1.02, result: -10500989 (-928)
 5629 times trade, ratio: 1.05, result: -6039715 (-1072)
 2229 times trade, ratio: 1.10, result: -3263950 (-1464)
  341 times trade, ratio: 1.50, result:  -705468 (-2068)
  180 times trade, ratio: 2.00, result:  -247646 (-1375)
  118 times trade, ratio: 2.50, result:  -160064 (-1356)
  106 times trade, ratio: 3.00, result:  -125409 (-1183)
"""

