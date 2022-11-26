import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import os
import glob
import pickle
import xgboost as xgb
import joblib
from privateLB import *
from lib_func_xgboost import *

def privateLB_xgb(start_date_api, start_date_sub):
    # model loaded
    ulsan_model = joblib.load("ulsan_xgb.pkl")
    dangjin_model = joblib.load("dangjin_xgb.pkl")

    key = 'sNfoTDclWrvFGpIEFDEXvj+EaCjLrOILF7IYehdRCcYBxnMP0zna40R1UmY6qfWBG0gJ16c3T8ManHwvhACk7w=='

    # load data from fcst
    dj = API(start_date_api,'2000','dangjin',key).get_data(preprocess=True)
    uls = API(start_date_api,'2000','ulsan',key).get_data(preprocess=True) 
 
    # submission 
    submission_path = "./test_sample_submission.csv"
    submission = pd.read_csv(submission_path)

    start_idx = int(submission[submission["time"] == start_date_sub].index.values)

    # ulsan forecasting
    x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]
    fcst_data = uls[x_name].iloc[0:,:].values.reshape(-1, len(x_name))

    yhat = submission_predict_xgb(ulsan_model, n_predict = 24, fcst_data = fcst_data)
    submission.iloc[start_idx:start_idx + 24,4] = yhat

    # dangjin forecasting
    x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Humidity", "Wind_X", "Wind_Y", "Cloud"]
    fcst = dj[x_name].iloc[0:,:].values.reshape(-1, len(x_name))

    yhat = submission_predict_xgb(dangjin_model, n_predict = 24, fcst_data = fcst_data)
    submission.iloc[start_idx:start_idx + 24,1] = yhat

    xgb_sub = submission.copy()
    submission.to_csv("test_sample_submission_xgb.csv", index = False)

    return xgb_sub