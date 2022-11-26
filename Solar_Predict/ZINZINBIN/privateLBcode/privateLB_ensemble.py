import os
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import glob
import pickle

from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import joblib

from privateLB import *
from lib_func_lgbm import *
from lib_func_xgboost import *
from privateLB_xgb import *
from privateLB_lgbm import *
from privateLB_rf import *


# index setting
start_date_api = "20210605"
start_date_sub = "2021-06-06 01:00:00"

submission_rf = privateLB_rf(start_date_api, start_date_sub)
submission_xgb = privateLB_xgb(start_date_api, start_date_sub)
submission_lgbm = privateLB_lgbm(start_date_api, start_date_sub)

# submission 
submission_path = "./test_sample_submission.csv"
submission = pd.read_csv(submission_path)

submission_ensemble = submission.add(submission_rf, fill_value = 0)
submission_ensemble = submission_ensemble.add(submission_xgb, fill_value = 0)
submission_ensemble = submission_ensemble.add(submission_lgbm, fill_value = 0)

submission_ensemble["dangjin_floating"] = 1 / 3 * submission_ensemble["dangjin_floating"]
submission_ensemble["ulsan"] = (1 / 3) * submission_ensemble["ulsan"]

submission_ensemble.to_csv("test_sample_submission_ensemble.csv", index = False)