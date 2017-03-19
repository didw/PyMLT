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
from keras_regression import SimpleModel
import datetime


def simulate(bd, ed):
    bd = int(bd)
    ed = int(ed)
    s_date = "%d_%d" % (bd, ed)
    print("train model %s" % s_date)
    sm = SimpleModel()
    X_train, Y_train, _ = sm.load_all_data(bd, ed)
    sm.train_model_tensorflow(X_train, Y_train, s_date)
    sm.save_scaler(s_date)
    sm.load_scaler(s_date)
    test_bd = datetime.date(ed/10000, ed%10000/100, ed%100) - datetime.timedelta(days=40)
    test_ed = datetime.date(ed/10000, ed%10000/100, ed%100) + datetime.timedelta(days=40)
    test_bd = int(test_bd.strftime("%Y%m%d"))
    test_ed = int(test_ed.strftime("%Y%m%d"))
    print("Evaluation on %d - %d" % (test_bd, test_ed))
    X_test, Y_test, Data = sm.load_all_data(test_bd, test_ed)
    fname = "../experiments_2/%s/%d_%d.txt" % (s_date, test_bd, test_ed)
    if not os.path.exists("../experiments_2/%s/" % s_date):
        os.makedirs("../experiments_2/%s/" % s_date)
    print("Save results to %s" % fname)
    sm.evaluate_model(X_test, Y_test, Data, s_date, fname)


def simulate_all():
    base_month = 201501
    while base_month <= 201701:
        end_date = datetime.date(base_month/100, base_month%100, 1)
        begin_date = datetime.date(base_month/100-5, base_month%100, 1)
        print(begin_date)
        simulate(begin_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
        base_month += 1
        if base_month%100 > 12:
            base_month += 88

            


if __name__ == '__main__':
    simulate_all()
