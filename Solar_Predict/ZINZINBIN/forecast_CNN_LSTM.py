# ======================================================================== #
# ====================== Prediction for Submission = ===================== #
# ======================================================================== #

# import library
import os
import pickle
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from lib_func_CNN_LSTM import *


if __name__ == "__main__":

    '''
    # load pre-trained model
    ulsan_model = tf.keras.models.load_model("ulsan_model.h5")
    dangjin_floating_model = tf.keras.models.load_model("dangjin_floating_model.h5")
    dangjin_warehouse_model = tf.keras.models.load_model("dangjin_warehouse_model.h5")
    dangjin_model = tf.keras.models.load_model("dangjin_model.h5")
    '''
    
    # ulsan
    
    params = {
        "filters":[128,128,256,256],
        "kernel_size":[3,3,3,3],
        "strides":[1,1,1],
        "pool_size":2,
        "dropout":0.3,
        "units":[512, 512],
        "dense_units":[128, 128, 128],
        "n_predict":24,
        "l2_lambda":0.01
    }
    

    ulsan_model = build_CNN_LSTM((1, 24, 9), params)
    ulsan_model.load_weights("./model_weights/ulsan_weights.h5")
    
    # dangjin_floating

    dangjin_floating_model = build_CNN_LSTM((1, 24, 9), params)
    dangjin_floating_model.load_weights("./model_weights/dangjin_floating_weights.h5")   
    
    # dangjin_warehouse

    
    dangjin_warehouse_model = build_CNN_LSTM((1, 24, 9), params)
    dangjin_warehouse_model.load_weights("./model_weights/dangjin_warehouse_weights.h5")   

    # dangjin

    
    dangjin_model = build_CNN_LSTM((1, 24, 9), params)
    dangjin_model.load_weights("./model_weights/dangjin_weights.h5")       
    

    # submission
    # for Public LB
    submission_path = "./submission.csv"
    submission = pd.read_csv(submission_path, encoding = "CP949")

    ulsan_obs_feb_path = "./data/ulsan_obs_2021-02.csv" 
    dangjin_obs_feb_path = "./data/dangjin_obs_2021-02.csv"

    ulsan_obs_feb = pd.read_csv(ulsan_obs_feb_path, encoding = "CP949" ) 
    dangjin_obs_feb = pd.read_csv(dangjin_obs_feb_path, encoding = "CP949")

    # preprocessing(변수명 변경 등..)
    ulsan_obs_feb = obs_preprocessing(ulsan_obs_feb)
    dangjin_obs_feb = obs_preprocessing(dangjin_obs_feb)

    # ulsan forecasting
    x_name_fcst = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Wind_X", "Wind_Y", "Humidity", "Cloud"]
    last_row = ulsan_obs_feb.iloc[-1,:]
    ulsan_obs_feb.loc[len(ulsan_obs_feb)] = last_row

    input_fcst = ulsan_obs_feb[x_name_fcst].iloc[1:,:].values.reshape(27, 1, 24, 9)
    prediction = ulsan_model.predict(input_fcst).reshape(-1,1)
    submission.iloc[0:24*27,4] = prediction

    # dangjin_floating forecasting
    x_name_fcst = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Wind_X", "Wind_Y", "Humidity", "Cloud"]
    last_row = dangjin_obs_feb.iloc[-1,:]
    dangjin_obs_feb.loc[len(dangjin_obs_feb)] = last_row

    input_fcst = dangjin_obs_feb[x_name_fcst].iloc[1:,:].values.reshape(27, 1, 24, 9)
    prediction = dangjin_floating_model.predict(input_fcst).reshape(-1,1)
    submission.iloc[0:24*27,1] = prediction

    # dangjin_warehouse forecasting
    input_fcst = dangjin_obs_feb[x_name_fcst].iloc[1:,:].values.reshape(27, 1, 24, 9)
    prediction = dangjin_warehouse_model.predict(input_fcst).reshape(-1,1)
    submission.iloc[0:24*27,2] = prediction

    # dangjin_warehouse forecasting
    input_fcst = dangjin_obs_feb[x_name_fcst].iloc[1:,:].values.reshape(27, 1, 24, 9)
    prediction = dangjin_model.predict(input_fcst).reshape(-1,1)
    submission.iloc[0:24*27,3] = prediction

    submission.to_csv("submission_CNN_LSTM.csv", index = False)
