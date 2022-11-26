from privateLB import *
from lib_func_lgbm import *
from main_lgbm import *
from lib_func_xgboost import *

import os
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
import pickle
import lightgbm as lgb
import joblib
from sklearn.ensemble import RandormForestRegressor


key = 'sNfoTDclWrvFGpIEFDEXvj+EaCjLrOILF7IYehdRCcYBxnMP0zna40R1UmY6qfWBG0gJ16c3T8ManHwvhACk7w=='

# index setting
start_date_api = "20210603"
start_date_sub = "2021-06-04 1:00"

# load data from fcst
dj = API(start_date_api,'2000','dangjin',key).get_data(preprocess=True)
uls = API(start_date_api,'2000','ulsan',key).get_data(preprocess=True) 

help(API.__init__)
help(API.get_data)

display(dj)
display(uls)

uls = uls.join(preprocess_wind(uls))
dj = dj.join(preprocess_wind(dj))

# model loaded
ulsan_model = joblib.load("ulsan_model.pkl")
dangjin_floating_model = joblib.load("dangjin_floating_model.pkl")
dangjin_warehouse_model = joblib.load("dangjin_warehouse_model.pkl")
dangjin_model = joblib.load("dangjin_model.pkl")

# rf model
ulsan_rf = joblib.load("ulsan_rf.pkl")
dangjin_rf = joblib.load("dangjin_rf.pkl")

# lgbm model
ulsan_lgbm = joblib.load("ulsan_lgbm.pkl")
dangjin_lgbm = joblib.load("dangjin_lgbm.pkl")

# xgboost model
ulsan_xgb = joblib.load('ulsan_xgb.pkl')
dangjin_xgb = joblib.load("dangjin_xgb.pkl")


def submission_predict_total(models, df = uls):
    yhats = []
    
    x_name_rf = ['Temperature', 'Humidity', 'Cloud', 'Day_cos', 'Day_sin', 'Year_cos', 'Year_sin']
    fcst_data = df[x_name_rf].iloc[0:,:].values.reshape(-1, len(x_name_rf))
    yhat_rf = models["rf"].predict(fcst_data)
    
    x_name_lgbm = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]
    fcst_data = df[x_name_lgbm].iloc[0:,:].values.reshape(-1, len(x_name_lgbm))
    yhat_lgbm = models["lgbm"].predict(fcst_data)
    
    x_name_xgb = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]
    fcst_data = df[x_name_xgb].iloc[0:,:].values.reshape(-1, len(x_name_xgb))
    yhat_xgb = models["xgb"].predict(fcst_data)
    
    yhats = [yhat_rf, yhat_lgbm, yhat_xgb]
    yhats = np.array(yhats).reshape(-1,3)
    yhats_avg = np.mean(yhats, axis = 1).reshape(-1,1)
    
    return yhats_avg
    

ulsan_model = {
    "rf":ulsan_rf,
    "lgbm":ulsan_lgbm,
    "xgb":ulsan_xgb
    }

dangjin_model = {
    "rf":dangjin_rf,
    "lgbm":dangjin_lgbm,
    "xgb":dangjin_xgb
    }


# submission 
submission_path = "./test_sample_submission.csv"
submission = pd.read_csv(submission_path)
start_idx = int(submission[submission["time"] == start_date_sub].index.values)

# ulsan forecasting
yhat = submission_predict_total(ulsan_model, df = uls)
submission.iloc[start_idx:start_idx + 24,4] = yhat

# dangjin forecasting
yhat = submission_predict_total(dangjin_model, df = dj)
submission.iloc[start_idx:start_idx + 24,1] = yhat


'''
# dangjin_floating forecasting
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]
fcst = dj[x_name].iloc[0:,:].values.reshape(-1, len(x_name))

yhat = submission_predict_lgb(dangjin_floating_model, n_predict = 24, fcst_data = fcst_data)
submission.iloc[start_idx:start_idx + 24,1] = yhat

# dangjin_warehouse forecasting
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]

yhat = submission_predict_lgb(dangjin_warehouse_model, n_predict = 24, fcst_data = fcst_data)
submission.iloc[start_idx:start_idx + 24,2] = yhat

# dangjin forecasting
x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]

yhat = submission_predict_lgb(dangjin_model, n_predict = 24, fcst_data = fcst_data)
submission.iloc[start_idx:start_idx + 24,3] = yhat

'''

submission.to_csv("test_sample_submission.csv", index = False)