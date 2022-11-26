'''
# ======================================================================== #
# =========================== File Explanation =========================== #
# ======================================================================== #

- model: xgboost
- preprocessing: dropna, (wd, ws) -> (ws_x, ws_y), month, day, hour -> (m,d,h)
- file structure: main, lib_function
- datasets: original datasets + 2015 - 2018 datasets(external)
'''

# library
import os
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
from lib_func_xgboost import *
import pickle

# ======================================================================== #
# ===================== load data and preprocessing ====================== #
# ======================================================================== #

# load data
with open('./data/dangjin_data.pkl','rb') as f:
    dangjin_data = pickle.load(f)
with open('./data/ulsan_data.pkl','rb') as f:
    ulsan_data = pickle.load(f)

# energy = 0 drop
dangjin_data = dangjin_data[dangjin_data["dangjin"] != 0]
dangjin_data = dangjin_data[dangjin_data["dangjin_floating"] != 0]
dangjin_data = dangjin_data[dangjin_data["dangjin_warehouse"] != 0]
ulsan_data = ulsan_data[ulsan_data["ulsan"] != 0]

# ======================================================================== #
# ================== build model and training(xgboost) =================== #
# ======================================================================== #

# model architecture
import xgboost as xgb
esr = 400 # early stopping round

params = {
    "booster":"dart",
    #"objective":"reg:pseudohubererror",
    "objective":"reg:squarederror",
    "learning_rate":0.1,
    #"max_depth":7,
    "max_depth":9,
    "n_estimators":1000,
    "nthread":-1,
    "gamma":1.0,
    "subsample":0.7,
    "colsample_bytree":1.0,
    "colsample_bylevel":1.0,
    "min_child_weight":5, #5
    "reg_lambda":1.0,
    "reg_alpha":1.0, # 1.0
    "sample_type":"uniform"
}

# ulsan 

x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]
y_name = ["ulsan"]
ulsan_model = build_xgb(params)
x_train, x_val, y_train, y_val = data_generate_xgb(ulsan_data.iloc[0:-24*30], x_name, y_name, test_size = 0.2)

from sklearn.preprocessing import MinMaxScaler

'''
scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
'''

custom_eval_ulsan = lambda x,y : custom_evaluation(x, y, cap = "ulsan")

ulsan_model.fit(x_train, y_train, eval_set = [(x_val, y_val)], early_stopping_rounds = esr, eval_metric = custom_eval_ulsan)
xgb.plot_importance(ulsan_model, height = 0.9)

# dangjin floating
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]
y_name = ["dangjin_floating"]
dangjin_floating_model = build_xgb(params)
dangjin_data = dangjin_data.dropna()

x_train, x_val, y_train, y_val = data_generate_xgb(dangjin_data.iloc[0:-24*30], x_name, y_name, test_size = 0.2)
custom_eval_dangjin_floating = lambda x,y : custom_evaluation(x, y, cap = "dangjin_floating")
dangjin_floating_model.fit(x_train, y_train, eval_set = [(x_val, y_val)], early_stopping_rounds = esr, eval_metric = custom_eval_dangjin_floating)
xgb.plot_importance(dangjin_floating_model, height = 0.9)

# dangjin warehouse
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]
y_name = ["dangjin_warehouse"]
dangjin_warehouse_model = build_xgb(params)
x_train, x_val, y_train, y_val = data_generate_xgb(dangjin_data.iloc[0:-24*30], x_name, y_name, test_size = 0.2)
custom_eval_dangjin_warehouse = lambda x,y : custom_evaluation(x, y, cap = "dangjin_warehouse")
dangjin_warehouse_model.fit(x_train, y_train, eval_set = [(x_val, y_val)], early_stopping_rounds = esr, eval_metric = custom_eval_dangjin_warehouse)
xgb.plot_importance(dangjin_warehouse_model, height = 0.9)


# dangjin 
'''
params = {
    "booster":"dart",
    #"objective":"reg:pseudohubererror",
    "objective":"reg:squarederror",
    "learning_rate":0.1,
    #"max_depth":7,
    "max_depth":9,
    "n_estimators":1000,
    "nthread":-1,
    "gamma":1.0,
    "subsample":0.7,
    "colsample_bytree":1.0,
    "colsample_bylevel":1.0,
    "min_child_weight":5, #5
    "reg_lambda":10.0,
    "reg_alpha":10.0, # 1.0
    "sample_type":"uniform"
}
'''

x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]
y_name = ["dangjin"]
dangjin_model = build_xgb(params)
x_train, x_val, y_train, y_val = data_generate_xgb(dangjin_data.iloc[0:-24*30], x_name, y_name, test_size = 0.2)
custom_eval_dangjin = lambda x,y : custom_evaluation(x, y, cap = "dangjin")
dangjin_model.fit(x_train, y_train, eval_set = [(x_val, y_val)], early_stopping_rounds = esr, eval_metric = custom_eval_dangjin)
xgb.plot_importance(dangjin_model, height = 0.9)


# ======================================================================== #
# ================= forecasting and evaluate the model =================== #
# ======================================================================== #

# evaluation
term_3d = range(0, 24 * 3)
term_7d = range(0, 24 * 7)
term_30d = range(0, 24 * 30)

# ulsan evaluation
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]
y_name = ["ulsan"]

n_predict = 24 * 30
fcst_data = ulsan_data[x_name].iloc[-24 * 30 * 1 : ].values.reshape(-1, len(x_name))


yhat = submission_predict_xgb(ulsan_model, n_predict = n_predict, fcst_data = fcst_data)
yreal = ulsan_data[y_name].iloc[-24*30*1 : ].values.reshape(-1,1)

for i, term in enumerate([term_3d, term_7d, term_30d]):
    name = str(len(term) / 24) + " - days forecast: ulsan"
    plt.figure(i+1, figsize = (10,5))
    plt.plot(yreal[term], label = "real")
    plt.plot(yhat[term], label = "forecast")
    plt.ylabel("ulsan, unit:None")
    plt.title(name)
    plt.legend()
    plt.show()

ulsan_nmae = sola_nmae(yreal, yhat, cap = "ulsan")
print("nmae for ulsan: ", ulsan_nmae)

# dangjin_floating evaluation
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]
y_name = ["dangjin_floating"]

n_in = 1
n_predict = 24 * 30
fcst_data = dangjin_data[x_name].iloc[-24 * 30 * 1 : ].values.reshape(-1, len(x_name))

yhat = submission_predict_xgb(dangjin_floating_model, n_predict = n_predict, fcst_data = fcst_data)
yreal = dangjin_data[y_name].iloc[-24*30*1 : ].values.reshape(-1,1)

for i, term in enumerate([term_3d, term_7d, term_30d]):
    name = str(len(term) / 24) + " - days forecast: dangjin_floating"
    plt.figure(i+1, figsize = (10,5))
    plt.plot(yreal[term], label = "real")
    plt.plot(yhat[term], label = "forecast")
    plt.ylabel("dangjin_floating, unit:None")
    plt.title(name)
    plt.legend()
    plt.show()

dangjin_floating_nmae = sola_nmae(yreal, yhat, cap = "dangjin_floating")
print("nmae for dangjin_floating: ", dangjin_floating_nmae)


# dangjin_warehouse evaluation
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]
y_name = ["dangjin_warehouse"]

n_predict = 24 * 30
fcst_data = dangjin_data[x_name].iloc[-24 * 30 * 1 : ].values.reshape(-1, len(x_name))

yhat = submission_predict_xgb(dangjin_warehouse_model, n_predict = n_predict, fcst_data = fcst_data)
yreal = dangjin_data[y_name].iloc[-24*30*1 : ].values.reshape(-1,1)

for i, term in enumerate([term_3d, term_7d, term_30d]):
    name = str(len(term) / 24) + " - days forecast: dangjin_warehouse"
    plt.figure(i+1, figsize = (10,5))
    plt.plot(yreal[term], label = "real")
    plt.plot(yhat[term], label = "forecast")
    plt.ylabel("dangjin_warehouse, unit:None")
    plt.title(name)
    plt.legend()
    plt.show()
    
dangjin_warehouse_nmae = sola_nmae(yreal, yhat, cap = "dangjin_warehouse")
print("nmae for dangjin_warehouse: ", dangjin_warehouse_nmae)


# dangjin evaluation
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]
y_name = ["dangjin"]

n_in = 1
n_predict = 24 * 30
start_data_in = dangjin_data[x_name].iloc[-24*30*1 - n_in].values.reshape(1,-1)
fcst_data = dangjin_data[x_name].iloc[-24 * 30 * 1 : ].values.reshape(-1, len(x_name))

yhat = submission_predict_xgb(dangjin_model, n_predict = n_predict, fcst_data = fcst_data)
yreal = dangjin_data[y_name].iloc[-24*30*1 : ].values.reshape(-1,1)

for i, term in enumerate([term_3d, term_7d, term_30d]):
    name = str(len(term) / 24) + " - days forecast: dangjin"
    plt.figure(i+1, figsize = (10,5))
    plt.plot(yreal[term], label = "real")
    plt.plot(yhat[term], label = "forecast")
    plt.ylabel("dangjin, unit:None")
    plt.title(name)
    plt.legend()
    plt.show()

dangjin_nmae = sola_nmae(yreal, yhat, cap = "dangjin")
print("nmae for dangjin: ", dangjin_nmae)


# submission 
submission_path = "./sample_submission.csv"
submission = pd.read_csv(submission_path)


ulsan_obs_feb_path = "./data/ulsan_obs_2021-02.csv" 
dangjin_obs_feb_path = "./data/dangjin_obs_2021-02.csv"

ulsan_obs_feb = pd.read_csv(ulsan_obs_feb_path, encoding = "CP949" ) 
dangjin_obs_feb = pd.read_csv(dangjin_obs_feb_path, encoding = "CP949")

# preprocessing(변수명 변경 등..)
ulsan_obs_feb = obs_preprocessing(ulsan_obs_feb)
dangjin_obs_feb = obs_preprocessing(dangjin_obs_feb)
    

# ulsan forecasting
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]

obs_data = ulsan_obs_feb[x_name].iloc[1:,:].values.reshape(-1, len(x_name))
yhat = submission_predict_xgb(ulsan_model, n_predict = 24 * 27 - 1, fcst_data = obs_data)
submission.iloc[0:24*27 -1,4] = yhat

# dangjin_floating forecasting
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]

obs_data = dangjin_obs_feb[x_name].iloc[1:,:].values.reshape(-1, len(x_name))
yhat = submission_predict_xgb(dangjin_floating_model, n_predict = 24 * 27 - 1, fcst_data = obs_data)
submission.iloc[0:24*27 -1,1] = yhat

# dangjin_warehouse forecasting
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]

obs_data = dangjin_obs_feb[x_name].iloc[1:,:].values.reshape(-1, len(x_name))
yhat = submission_predict_xgb(dangjin_warehouse_model, n_predict = 24 * 27 - 1, fcst_data = obs_data)
submission.iloc[0:24*27 -1,2] = yhat

# dangjin forecasting
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]

obs_data = dangjin_obs_feb[x_name].iloc[1:,:].values.reshape(-1, len(x_name))
yhat = submission_predict_xgb(dangjin_model, n_predict = 24 * 27 - 1, fcst_data = obs_data)
submission.iloc[0:24*27 -1,3] = yhat

submission.to_csv("submission_xgb.csv", index = False)

print("expected evaluation nmae:", 0.25*(ulsan_nmae + dangjin_floating_nmae + dangjin_warehouse_nmae + dangjin_nmae))