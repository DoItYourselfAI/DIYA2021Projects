# ======================================================================== #
# ===================== CNN_LSTM model training file ===================== #
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


if __name__=="__main__":

    # load data from pickle
    with open('./data/dangjin_data.pkl','rb') as f:
        dangjin_data = pickle.load(f)
    with open('./data/ulsan_data.pkl','rb') as f:
        ulsan_data = pickle.load(f)
        
    '''

    # time as index
    dangjin_data.set_index('time', inplace=True)
    ulsan_data.set_index('time', inplace=True)

    dangjin_data = dangjin_data[["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Wind_X_obs", "Wind_Y_obs", "Humidity_obs", "Cloud_obs", "dangjin_floating", "dangjin_warehouse", "dangjin"]]
    ulsan_data = ulsan_data[["Day_cos","Day_sin","Year_cos","Year_sin","Temperature_obs", "Wind_X_obs", "Wind_Y_obs", "Humidity_obs", "Cloud_obs", "ulsan"]]

    # column name change(..._obs -> ...)
    dangjin_data.rename(
        columns = {
            "Temperature_obs":"Temperature",
            "Wind_X_obs":"Wind_X",
            "Wind_Y_obs":"Wind_Y",
            "Humidity_obs":"Humidity",
            "Cloud_obs":"Cloud"
        }, inplace = True)

    ulsan_data.rename(
        columns = {
            "Temperature_obs":"Temperature",
            "Wind_X_obs":"Wind_X",
            "Wind_Y_obs":"Wind_Y",
            "Humidity_obs":"Humidity",
            "Cloud_obs":"Cloud"
        }, inplace = True)


    '''
    
    # common
    input_shape = (1,24,9)
    epochs = 64
    batch_size = 64
    
    # model training
    # model parameter setting

    ulsan_params = {
        "filters":[64,64,128],
        "kernel_size":[3,3,2],
        "strides":[1,1,1,1],
        "pool_size":2,
        "dropout":0.3,
        "units":[256, 256],
        "dense_units":[128, 128, 64],
        "n_predict":24,
        "l2_lambda":0.1
        }

    dangjin_floating_params = {
        "filters":[64,64,128],
        "kernel_size":[3,3,2],
        "strides":[1,1,1,1],
        "pool_size":2,
        "dropout":0.3,
        "units":[256, 256],
        "dense_units":[128, 128, 64],
        "n_predict":24,
        "l2_lambda":0.1
        }

    dangjin_warehouse_params = {
        "filters":[64,64,128],
        "kernel_size":[3,3,2],
        "strides":[1,1,1,1],
        "pool_size":2,
        "dropout":0.3,
        "units":[256, 256],
        "dense_units":[128, 128, 64],
        "n_predict":24,
        "l2_lambda":0.1
        }

    dangjin_params = {
        "filters":[64,64,128],
        "kernel_size":[3,3,2],
        "strides":[1,1,1,1],
        "pool_size":2,
        "dropout":0.3,
        "units":[256, 256],
        "dense_units":[128, 128, 64],
        "n_predict":24,
        "l2_lambda":0.1
        }


    # callbacks params

    ulsan_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 16, mode = "min"), # Early Stopping
        tf.keras.callbacks.ModelCheckpoint("ulsan_weights.h5", monitor = "val_loss", save_best_only = True, save_weights_only = True, mode = "min"), # ModelCheckpoint
        tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = np.sqrt(0.1), patience = 16, verbose = 0, mode = "min") # Learning Rate 
    ]

    dangjin_floating_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 16, mode = "min"), # Early Stopping
        tf.keras.callbacks.ModelCheckpoint("dangjin_floating_weights.h5", monitor = "val_loss", save_best_only = True, save_weights_only = True, mode = "min"), # ModelCheckpoint
        tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = np.sqrt(0.1), patience = 16, verbose = 0, mode = "min") # Learning Rate 
    ]   

    dangjin_warehouse_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 16, mode = "min"), # Early Stopping
        tf.keras.callbacks.ModelCheckpoint("dangjin_warehouse_weights.h5", monitor = "val_loss", save_best_only = True, save_weights_only = True, mode = "min"), # ModelCheckpoint
        tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = np.sqrt(0.1), patience = 16, verbose = 0, mode = "min") # Learning Rate 
    ]

    dangjin_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 16, mode = "min"), # Early Stopping
        tf.keras.callbacks.ModelCheckpoint("dangjin_weights.h5", monitor = "val_loss", save_best_only = True, save_weights_only = True, mode = "min"), # ModelCheckpoint
        tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_loss", factor = np.sqrt(0.1), patience = 16, verbose = 0, mode = "min") # Learning Rate 
    ]

    # train datasets

    x_ulsan, y_ulsan = dataloader(ulsan_data.iloc[:-24*30,:], y_name = ["ulsan"])
    x_dangjin_floating, y_dangjin_floating = dataloader(dangjin_data.iloc[:-24*30,:], y_name = ["dangjin_floating"])
    x_dangjin_warehouse, y_dangjin_warehouse = dataloader(dangjin_data.iloc[:-24*30,:], y_name = ["dangjin_warehouse"])
    x_dangjin, y_dangjin = dataloader(dangjin_data.iloc[:-24*30,:], y_name = ["dangjin"])

    # model training

    ulsan_model = build_CNN_LSTM(input_shape, ulsan_params)
    dangjin_floating_model = build_CNN_LSTM(input_shape, dangjin_floating_params)
    dangjin_warehouse_model = build_CNN_LSTM(input_shape, dangjin_warehouse_params)
    dangjin_model = build_CNN_LSTM(input_shape, dangjin_params)

    ulsan_model.fit(x_ulsan, y_ulsan, validation_split = 0.2, epochs = epochs, batch_size = batch_size, callbacks = ulsan_callbacks)
    ulsan_model.save_weights("./model_weights/ulsan_weights.h5")
    ulsan_model.save("ulsan_model.h5")

    dangjin_floating_model.fit(x_dangjin_floating, y_dangjin_floating, validation_split = 0.2, epochs = epochs, batch_size = batch_size, callbacks = dangjin_floating_callbacks)
    dangjin_floating_model.save_weights("./model_weights/dangjin_floating_weights.h5")
    dangjin_floating_model.save("dangjin_floating_model.h5")

    dangjin_warehouse_model.fit(x_dangjin_warehouse, y_dangjin_warehouse, validation_split = 0.2, epochs = epochs, batch_size = batch_size, callbacks = dangjin_warehouse_callbacks)
    dangjin_warehouse_model.save_weights("./model_weights/dangjin_warehouse_weights.h5")
    dangjin_warehouse_model.save("dangjin_warehouse_model.h5")

    dangjin_model.fit(x_dangjin, y_dangjin, validation_split = 0.2, epochs = epochs, batch_size = batch_size, callbacks = dangjin_callbacks)
    dangjin_model.save_weights("./model_weights/dangjin_weights.h5")
    dangjin_model.save("dangjin_model.h5")

    # model evaluation
    ulsan_nmae = evaluation_plot(ulsan_model, ulsan_data.iloc[-24*30:], name = "ulsan", output_columns = ["ulsan"])
    dangjin_floating_nmae = evaluation_plot(dangjin_floating_model, dangjin_data.iloc[-24*30:], name = "dangjin_floating", output_columns = ["dangjin_floating"])
    dangjin_warehouse_nmae = evaluation_plot(dangjin_warehouse_model, dangjin_data.iloc[-24*30:], name = "dangjin_warehouse", output_columns = ["dangjin_warehouse"])
    dangjin_nmae = evaluation_plot(dangjin_model, dangjin_data.iloc[-24*30:], name = "dangjin", output_columns = ["dangjin"])