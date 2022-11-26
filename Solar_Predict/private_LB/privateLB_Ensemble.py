# for private LB
# should be run in /Solar_Predict/private_LB

#######################################################################################
#### submissions/test_submission_for_code.csv should be deleted in real private LB ####
#######################################################################################

import pandas as pd
from privateLB_models import RfPredict, LgbmPredict, XgbPredict

# date of prediction
base_date = "20210610"
base_time = "2000"

# predictions from the models
submission_rf = RfPredict(base_date, base_time).get_submission()
submission_lgbm = LgbmPredict(base_date, base_time).get_submission()
submission_xgb = XgbPredict(base_date, base_time).get_submission()

# add each predictions
# "ulsan" is an arbitrary column to store the sum of all predictions
submission_ensemble = submission_rf
submission_ensemble["ulsan"] = submission_ensemble["ulsan"].add(submission_lgbm["ulsan"], fill_value = 320)
submission_ensemble["ulsan"] = submission_ensemble["ulsan"].add(submission_xgb["ulsan"], fill_value = 320)

submission_ensemble["ulsan"] = submission_ensemble["ulsan"] / 3 
 
# write CSV
# show the predictions for two days(=48 rows)
submission_ensemble.to_csv(f"./submissions/{base_date}_{base_time}.csv", index = False)
# be careful when modifying the write_path. Unintended file in "submissions" directory might cause code to misbehave.
