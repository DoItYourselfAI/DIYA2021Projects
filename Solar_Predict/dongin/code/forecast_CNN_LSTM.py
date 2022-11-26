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

    # load pre-trained model
    ulsan_model = tf.keras.models.load_model("ulsan_model.h5")
    dangjin_floating_model = tf.keras.models.load_model("dangjin_floating_model.h5")
    dangjin_warehouse_model = tf.keras.models.load_model("dangjin_warehouse_model.h5")
    dangjin_model = tf.keras.models.load_model("dangjin_model.h5")

    # submission
    # for Public LB
    submission_path = "./submission.csv"
    submission = pd.read_csv(submission_path, encoding = "CP949")

    ulsan_obs_feb_path = "../../external_data/ulsan_obs_2021-02.csv" 
    dangjin_obs_feb_path = "../../external_data/dangjin_obs_2021-02.csv"

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
