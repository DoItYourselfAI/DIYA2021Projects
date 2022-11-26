# import library
import os
import pickle
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import matplotlib.pyplot as plt
import glob

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# ======================================================================== #
# ================ Preprocessing with ulsan / danjin data ================ #
# ======================================================================== #

def preprocess_wind(data):
    '''
    data: pd.DataFrmae which contains the columns 'WindSpeed' and 'WindDirection'
    '''

    # degree to radian
    wind_direction_radian = data['WindDirection'] * np.pi / 180

    # polar coordinate to cartesian coordinate
    wind_x = data['WindSpeed'] * np.cos(wind_direction_radian)
    wind_y = data['WindDirection'] * np.sin(wind_direction_radian)

    # name pd.series
    wind_x.name = 'Wind_X'
    wind_y.name = 'Wind_Y'

    return wind_x, wind_y

def day_of_year(datetime): #pd.datetime
    return pd.Period(datetime, freq='D').dayofyear

def add_seasonality(df):
    new_df = df.copy()
    
    new_df['Day_cos'] = new_df['time'].apply(lambda x: np.cos(x.hour * (2 * np.pi) / 24))
    new_df['Day_sin'] = new_df['time'].apply(lambda x: np.sin(x.hour * (2 * np.pi) / 24))

    new_df['Year_cos'] = new_df['time'].apply(lambda x: np.cos(day_of_year(x) * (2 * np.pi) / 365))
    new_df['Year_sin'] = new_df['time'].apply(lambda x: np.sin(day_of_year(x) * (2 * np.pi) / 365))

    return new_df

# obs_preprocessing

def obs_preprocessing(df):

    df_obs = df.copy()
    df_obs.rename(
        columns = {
                "일시":"time",
                "기온(°C)":"Temperature",
                "풍속(m/s)":"WindSpeed",
                "풍향(16방위)":"WindDirection",
                "습도(%)":"Humidity",
                "전운량(10분위)":"Cloud"
            }, inplace = True)

    df_obs.drop(columns = ["지점", "지점명"])

    df_obs = df_obs.join(preprocess_wind(df_obs))

    for i in range(df_obs.shape[0]):
        df_obs["time"][i] = pd.to_datetime(df_obs["time"][i])
    
    df_obs = df_obs.astype({"time":"object"})
    df_obs = add_seasonality(df_obs)
        
    return df_obs

# scaling

from sklearn.preprocessing import MinMaxScaler

def preprocessing(data, col_name):
    # col_name: column name(string type)
    # data: pd.DataFrame
    
    data_np = data[col_name].values
    
    scaler = MinMaxScaler()
    data_np = scaler.fit_transform(data_np).reshape(-1, len(col_name))
    
    data = data.drop(columns = col_name)
    data[col_name] = data_np
    
    return data, scaler
    
def PG_preprocessing(data, name = "ulsan"):
    data_np = data[name].values.reshape(-1,1)
    
    scaler = MinMaxScaler()
    
    data_np = scaler.fit_transform(data_np).reshape(-1,1)
    
    data = data.drop(columns = name)
    data[name] = data_np
    
    return data, scaler

def PG_inverse(y, scaler):
    # ulsan, dangjin_floating, warehouse, dangjin data scaler
    # inverse transform to its own unit
    
    if y.shape[1] != 1:
        y = y.reshape(-1,1)
    y_rs = scaler.inverse_transform(y).reshape(-1,1)
    return y_rs


# conversion for supervised learning
def series_to_supervised(data, x_name, y_name, n_in, n_out, dropnan = False):

    # x_name: Temp...etc except PG
    # y_name: PG
    # n_in and n_out: equal

    data_copy = data.copy()
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        #cols.append(data_copy[x_name].shift(i)) # col: [data_copy.shift(n_in), .... data_copy.shift(1)]
        cols.append(data_copy[x_name].shift(i))
        names += [("%s(t-%d)"%(name, i)) for name in x_name]
    
    for i in range(n_out, 0, -1):
        y = data_copy[y_name]
        cols.append(y.shift(i))
        # cols:[data_copy.shift(n_in-1), .... data_copy.shift(1), data_copy[y_name].shift(0)....data_copy[y_name].shift(-n_out + 1)]

        names += [("%s(t-%d)"%(name, i)) for name in y_name]

    agg = pd.concat(cols, axis = 1)
    agg.columns = names

    if dropnan:
        agg.dropna(inplace = True)
    
    return agg

# dataloader

def dataloader(
    df, 
    x_name = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Wind_X", "Wind_Y", "Humidity", "Cloud"], 
    y_name = ["ulsan"]
    ):

    n_in = 24
    n_out = 24
    n_timesteps = n_in
    n_features = len(x_name)

    df_supervised = series_to_supervised(df, x_name = x_name, y_name = y_name, n_in = n_in, n_out = n_out, dropnan = True)
    x, y = df_supervised.iloc[:,: -n_out].values, df_supervised.iloc[:, -n_out:].values
    x = x.reshape(-1,1,n_timesteps, n_features).astype("float32")
    y = y.reshape(-1, n_out, 1).astype("float32")

    return (x,y)



# ======================================================================== #
# ================== build model and training(CNN_LSTM) ================== #
# ======================================================================== #

sample_params = {
    "filters":[64,64,128,128],
    "kernel_size":[2,2,2,2],
    "strides":[1,1,1,1],
    "pool_size":2,
    "dropout":0.3,
    "units":[128, 128],
    "dense_units":[64, 64, 64],
    "n_predict":24,
    "l2_lambda":0.1
}

def build_CNN_LSTM(input_shape, params = sample_params):

    filters = params["filters"]
    kernel_size = params["kernel_size"]
    strides = params["strides"]
    pool_size = params["pool_size"]
    dropout = params["dropout"]
    units = params["units"]
    dense_units = params["dense_units"]
    n_predict = params["n_predict"]
    l2_lambda = params["l2_lambda"]

    inputs = tf.keras.layers.Input(shape = input_shape, name = "input_layer")

    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.BatchNormalization()
    )(inputs)

 # CNN Encoder
    for i, filter, kernel, stride in zip(range(1,len(filters)+1), filters, kernel_size, strides):
        x = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv1D(
                filters = filter,
                kernel_size = kernel,
                strides = stride,
                padding = "valid",
                kernel_initializer = "glorot_uniform",
                activation = "relu",
                kernel_regularizer= tf.keras.regularizers.l2(l2_lambda)
                )
            )(x)
        
        if i % 2 == 0:
            x =  tf.keras.layers.TimeDistributed(
                tf.keras.layers.BatchNormalization()
                )(x)

            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.AveragePooling1D(pool_size = pool_size)
                )(x)
            
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dropout(dropout)
                )(x)

    # Flatten
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Flatten()
    )(x)

    # LSTM 

    for i, unit in enumerate(units):
        if i == 0:
            x = tf.keras.layers.LSTM(
                    units = unit,
                    activation = "tanh",
                    recurrent_activation = "tanh",
                    return_sequences = False,
                    kernel_regularizer = tf.keras.regularizers.l2(l2_lambda),
                    dropout = dropout
                )(x)
            
            x = tf.keras.layers.RepeatVector(n_predict)(x)

        elif i != 0:
            x = tf.keras.layers.LSTM(
                    units = unit,
                    activation = "tanh",
                    recurrent_activation = "tanh",
                    return_sequences = True,
                    kernel_regularizer = tf.keras.regularizers.l2(l2_lambda),
                    dropout = dropout
                )(x)
        
        
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.BatchNormalization()
    )(x)
    
    # Regression

    for i, unit in enumerate(dense_units):
        if i != (len(dense_units) - 1):
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    unit,
                    activation = "relu",
                    kernel_regularizer = tf.keras.regularizers.l2(l2_lambda)
                    )
            )(x)

            x =  tf.keras.layers.TimeDistributed(
                tf.keras.layers.BatchNormalization()
                )(x)


        elif i == len(dense_units) - 1:
            x = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(
                    unit,
                    activation = "relu",
                    kernel_regularizer = tf.keras.regularizers.l2(l2_lambda)
                    )
                )(x)   


    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, name = "output_layer")
        )(x)
    
    # tf.debugging.set_log_device_placement(True)
    # with tf.device('/GPU:0'):

    
    model = tf.keras.models.Model(inputs, outputs, name = "CNN_LSTM")
    model.compile(
        loss = tf.keras.losses.MeanSquaredError(),
        optimizer = tf.keras.optimizers.Adam(lr = 1e-4),
        run_eagerly = True
    )

    model.summary()  

    return model

# TPU initialize and version check
# can be activated only for COLAB
'''
def TPU_initialize():

    print('Pandas: %s'%(pd.__version__))
    print('Numpy: %s'%(np.__version__))
    print('Scipy: %s'%(sp.__version__))
    print('Tensorflow: %s'%(tf.__version__))
    print('Keras: %s'%(tf.keras.__version__))

    # Detect hardware
    try:
        tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
    except ValueError:
        tpu_resolver = None
        gpus = tf.config.experimental.list_logical_devices("GPU")

    # Select appropriate distribution strategy
    if tpu_resolver:
        tf.config.experimental_connect_to_cluster(tpu_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
        strategy = tf.distribute.experimental.TPUStrategy(tpu_resolver)
        print('Running on TPU ', tpu_resolver.cluster_spec().as_dict()['worker'])
    elif len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
        print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
        print('Running on single GPU ', gpus[0].name)
    else:
        strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
        print('Running on CPU')
    print("Number of accelerators: ", strategy.num_replicas_in_sync)
'''

# evaluation: NMAE function and ploting

capacity = {
    'dangjin_floating':1000, # 당진수상태양광 발전용량
    'dangjin_warehouse':700, # 당진자재창고태양광 발전용량
    'dangjin':1000, # 당진태양광 발전용량
    'ulsan':500 # 울산태양광 발전용량
}


def sola_nmae(y_true, y_pred, cap = "ulsan"):
    if type(y_pred) is tf.float32:
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()

    abs_err = np.abs(y_true - y_pred)
    abs_err /= capacity[cap]
    target_idx = np.where(y_pred >= capacity[cap] * 0.1)
    result = 100 * abs_err[target_idx].mean()
    return result


def evaluation_plot(
    model, 
    d_obs, 
    
    name = "ulsan", 
    input_columns = ["Day_cos","Day_sin","Year_cos","Year_sin","Temperature", "Wind_X", "Wind_Y", "Humidity", "Cloud"],
    output_columns = ["ulsan"]
    ):

    '''
    - d_obs: 예측하고자 하는 시간대의 기상값, datashape: ()
    - d_label: 실제 발전량, datashape: (timesteps, 1), dtype: np.ndarray
    - model: 예측에 활용할 모델(선 학습)
    - name: name of datasets
    - input_columns: name of columns for input data
    - output_columns: name of columns for prediction
    '''

    term_3d = range(0, 24 * 3)
    term_7d = range(0, 24 * 7)
    term_30d = range(0, 24 * 30)

    x = d_obs[input_columns].values.reshape(-1,1,24,len(input_columns))
    y = d_obs[output_columns].values.reshape(-1,1)

    terms = [term_3d, term_7d, term_30d]
    yhat = model.predict(x).reshape(-1,1)

    for term in terms:
        day = int(len(term) / 24)
        title = name + ", evaluation for " + str(day) + " days"
        y_label = name + ", unit:None"
        plt.plot(y[term], 'r', label = "actual data")
        plt.plot(yhat[term], 'b', label = "forecast")
        plt.legend()
        plt.xlabel("time(unit:hour)")
        plt.ylabel(y_label)
        plt.title(title)
        plt.show()

    nmae = sola_nmae(y, yhat, name)
    print("sola_nmae, %s: %.2f"%(name, nmae))

    return nmae