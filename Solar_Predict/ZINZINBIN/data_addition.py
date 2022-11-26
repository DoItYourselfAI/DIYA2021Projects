'''
- 일기 예보 데이터: https://data.kma.go.kr/data/rmt/rmtList.do?code=420&pgmNo=572
- 한국동서발전 시간대별 태양광 발전량 데이터: https://www.data.go.kr/data/15003553/fileData.do (페이지 하단부분 -> 주기성 과거 데이터)
- 한국동서발전 시간대별 태양광 발전량 현황 정보(2015 - 2017)
- 한국동서발전 시간대별 태양광 및 풍력 발전량 현황 정보(2018 - 2019)
- 15년도 1월 ~ 18년도 2월 기상 관측(obs) 및 발전량 데이터 추가
'''

import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# obs data
dangjin_2015 = pd.read_csv("./original_dataset/external_data/SURFACE_ASOS_129_HR_2015_2015_2016.csv", engine = "python", encoding = "cp949")
dangjin_2016 = pd.read_csv("./original_dataset/external_data/SURFACE_ASOS_129_HR_2016_2016_2017.csv", engine = "python", encoding = "cp949")
dangjin_2017 = pd.read_csv("./original_dataset/external_data/SURFACE_ASOS_129_HR_2017_2017_2018.csv", engine = "python", encoding = "cp949")
dangjin_2018 = pd.read_csv("./original_dataset/external_data/SURFACE_ASOS_129_HR_2018_2018_2019.csv", engine = "python", encoding = "cp949")

ulsan_2015 = pd.read_csv("./original_dataset/external_data/SURFACE_ASOS_152_HR_2015_2015_2016.csv", engine = "python", encoding = "cp949")
ulsan_2016 = pd.read_csv("./original_dataset/external_data/SURFACE_ASOS_152_HR_2016_2016_2017.csv", engine = "python", encoding = "cp949")
ulsan_2017 = pd.read_csv("./original_dataset/external_data/SURFACE_ASOS_152_HR_2017_2017_2018.csv", engine = "python", encoding = "cp949")
ulsan_2018 = pd.read_csv("./original_dataset/external_data/SURFACE_ASOS_152_HR_2018_2018_2019.csv", engine = "python", encoding = "cp949")

# energy data
energy_2015_2017 = pd.read_csv("./original_dataset/external_data/한국동서발전 시간대별 태양광 발전량 현황(2015_2017).csv", engine = "python", encoding = "cp949")

energy_2018_2019 = pd.read_csv("./original_dataset/external_data/한국동서발전 시간대별 태양광 및 풍력 발전량 현황(2018_2019).csv", engine = "python", encoding = "cp949")

# obs data handling
dangjin_2015 = dangjin_2015.drop(columns = ['지점',  '강수량(mm)', '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)', '일조(hr)','일사(MJ/m2)', '적설(cm)', '3시간신적설(cm)', '중하층운량(10분위)','운형(운형약어)', '최저운고(100m )', '시정(10m)', '지면상태(지면상태코드)', '현상번호(국내식)','지면온도(°C)', '5cm 지중온도(°C)', '10cm 지중온도(°C)', '20cm 지중온도(°C)','30cm 지중온도(°C)'])

dangjin_2016 = dangjin_2016.drop(columns = ['지점',  '강수량(mm)', '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)', '일조(hr)','일사(MJ/m2)', '적설(cm)', '3시간신적설(cm)', '중하층운량(10분위)','운형(운형약어)', '최저운고(100m )', '시정(10m)', '지면상태(지면상태코드)', '현상번호(국내식)','지면온도(°C)', '5cm 지중온도(°C)', '10cm 지중온도(°C)', '20cm 지중온도(°C)','30cm 지중온도(°C)'])

dangjin_2017 = dangjin_2017.drop(columns = ['지점',  '강수량(mm)', '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)', '일조(hr)','일사(MJ/m2)', '적설(cm)', '3시간신적설(cm)', '중하층운량(10분위)','운형(운형약어)', '최저운고(100m )', '시정(10m)', '지면상태(지면상태코드)', '현상번호(국내식)','지면온도(°C)', '5cm 지중온도(°C)', '10cm 지중온도(°C)', '20cm 지중온도(°C)','30cm 지중온도(°C)'])

dangjin_2018 = dangjin_2018.drop(columns = ['지점',  '강수량(mm)', '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)', '일조(hr)','일사(MJ/m2)', '적설(cm)', '3시간신적설(cm)', '중하층운량(10분위)','운형(운형약어)', '최저운고(100m )', '시정(10m)', '지면상태(지면상태코드)', '현상번호(국내식)','지면온도(°C)', '5cm 지중온도(°C)', '10cm 지중온도(°C)', '20cm 지중온도(°C)','30cm 지중온도(°C)'])


ulsan_2015 = ulsan_2015.drop(columns = ['지점',  '강수량(mm)', '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)', '일조(hr)','일사(MJ/m2)', '적설(cm)', '3시간신적설(cm)', '중하층운량(10분위)','운형(운형약어)', '최저운고(100m )', '시정(10m)', '지면상태(지면상태코드)', '현상번호(국내식)','지면온도(°C)', '5cm 지중온도(°C)', '10cm 지중온도(°C)', '20cm 지중온도(°C)','30cm 지중온도(°C)'])

ulsan_2016 = ulsan_2016.drop(columns = ['지점',  '강수량(mm)', '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)', '일조(hr)','일사(MJ/m2)', '적설(cm)', '3시간신적설(cm)', '중하층운량(10분위)','운형(운형약어)', '최저운고(100m )', '시정(10m)', '지면상태(지면상태코드)', '현상번호(국내식)','지면온도(°C)', '5cm 지중온도(°C)', '10cm 지중온도(°C)', '20cm 지중온도(°C)','30cm 지중온도(°C)'])

ulsan_2017 = ulsan_2017.drop(columns = ['지점',  '강수량(mm)', '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)', '일조(hr)','일사(MJ/m2)', '적설(cm)', '3시간신적설(cm)', '중하층운량(10분위)','운형(운형약어)', '최저운고(100m )', '시정(10m)', '지면상태(지면상태코드)', '현상번호(국내식)','지면온도(°C)', '5cm 지중온도(°C)', '10cm 지중온도(°C)', '20cm 지중온도(°C)','30cm 지중온도(°C)'])

ulsan_2018 = ulsan_2018.drop(columns = ['지점',  '강수량(mm)', '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '해면기압(hPa)', '일조(hr)','일사(MJ/m2)', '적설(cm)', '3시간신적설(cm)', '중하층운량(10분위)','운형(운형약어)', '최저운고(100m )', '시정(10m)', '지면상태(지면상태코드)', '현상번호(국내식)','지면온도(°C)', '5cm 지중온도(°C)', '10cm 지중온도(°C)', '20cm 지중온도(°C)','30cm 지중온도(°C)'])



dangjin_2015_2018 = pd.concat([dangjin_2015, dangjin_2016, dangjin_2017, dangjin_2018])
ulsan_2015_2018 = pd.concat([ulsan_2015, ulsan_2016, ulsan_2017, ulsan_2018])

display(dangjin_2015_2018)
display(ulsan_2015_2018)

dangjin_2015_2018.rename(
    columns = {
        "일시":"time",
        "기온(°C)":"Temperature",
        "풍속(m/s)":"WindSpeed",
        "풍향(16방위)":"WindDirection",
        "습도(%)":"Humidity",
        "전운량(10분위)":"Cloud"
    }, inplace = True)

ulsan_2015_2018.rename(
    columns = {
        "일시":"time",
        "기온(°C)":"Temperature",
        "풍속(m/s)":"WindSpeed",
        "풍향(16방위)":"WindDirection",
        "습도(%)":"Humidity",
        "전운량(10분위)":"Cloud"
    }, inplace = True)

# time: convert datatype object
dangjin_2015_2018["time"] = pd.to_datetime(dangjin_2015_2018["time"].copy(), format='%Y-%m-%d %H:%M:%S')
dangjin_2015_2018 = dangjin_2015_2018.astype({"time":"object"})

ulsan_2015_2018["time"] = pd.to_datetime(ulsan_2015_2018["time"].copy(), format='%Y-%m-%d %H:%M:%S')
ulsan_2015_2018 = ulsan_2015_2018.astype({"time":"object"})

# function for preprocessing
def preprocess_wind(data):
    '''
    - data: pd.DataFrmae which contains the columns 'WindSpeed' and 'WindDirection'
    - for Neural Network version
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

# WindSpeed, WindDirection -> Wind_X, Wind_Y
dangjin_obs_2015_2018 = dangjin_2015_2018.join(preprocess_wind(dangjin_2015_2018))
ulsan_obs_2015_2018 = ulsan_2015_2018.join(preprocess_wind(ulsan_2015_2018))

display(dangjin_obs_2015_2018)
display(ulsan_obs_2015_2018)

# energy_2015_2017 preprocessing
energy_2015_2017["시간"] = pd.to_datetime(energy_2015_2017["시간"])
energy_2018_2019["시간"] = pd.to_datetime(energy_2018_2019["시간"])
energy_2015_2019 = pd.concat([energy_2015_2017, energy_2018_2019])
energy_2015_2019.sort_values(by = ["시간"], ascending = True, inplace = True)

dangjin_energy = energy_2015_2019[energy_2015_2019['태양광명']=='당진태양광']
dangjin_float_energy = energy_2015_2019[(energy_2015_2019['태양광명']=='당진수상태양광') | (energy_2015_2019['태양광명']=='당진취수로태양광')]
dangjin_warehouse_energy = energy_2015_2019[energy_2015_2019['태양광명']=='당진자재창고태양광']
ulsan_energy = energy_2015_2019[energy_2015_2019['태양광명']=='울산태양광']



ulsan_energy=ulsan_energy[ulsan_energy['시간'].dt.year <= 2018]
ulsan_energy=ulsan_energy[~((ulsan_energy['시간'].dt.year == 2018) >= 3)]
dangjin_energy=dangjin_energy[dangjin_energy['시간'].dt.year <= 2018]
dangjin_energy=dangjin_energy[~((dangjin_energy['시간'].dt.year == 2018) >= 3)]
dangjin_float_energy=dangjin_float_energy[dangjin_float_energy['시간'].dt.year <= 2018]
dangjin_float_energy=dangjin_float_energy[~((dangjin_float_energy['시간'].dt.year == 2018) >= 3)]
dangjin_warehouse_energy=dangjin_warehouse_energy[dangjin_warehouse_energy['시간'].dt.year <= 2018]
dangjin_warehouse_energy=dangjin_warehouse_energy[~((dangjin_warehouse_energy['시간'].dt.year == 2018) >= 3)]
    

ulsan_energy=ulsan_energy.reset_index(drop=True)
dangjin_energy=dangjin_energy.reset_index(drop=True)
dangjin_float_energy=dangjin_float_energy.reset_index(drop=True)
dangjin_warehouse_energy=dangjin_warehouse_energy.reset_index(drop=True)

dangjin=pd.DataFrame(columns=['시간','dangjin'])
dangjin_floating=pd.DataFrame(columns=['시간','dangjin_floating'])
dangjin_warehouse=pd.DataFrame(columns=['시간','dangjin_warehouse'])
ulsan=pd.DataFrame(columns=['시간','ulsan'])

dangjin['dangjin']=dangjin['dangjin'].astype(float)
dangjin_floating['dangjin_floating']=dangjin_floating['dangjin_floating'].astype(float)
dangjin_warehouse['dangjin_warehouse']=dangjin_warehouse['dangjin_warehouse'].astype(float)
ulsan['ulsan']=ulsan['ulsan'].astype(float)

for i in range(24):
    dangjin=dangjin.append(dangjin_energy,ignore_index = True)
    dangjin_floating=dangjin_floating.append(dangjin_float_energy,ignore_index = True)
    dangjin_warehouse=dangjin_warehouse.append(dangjin_warehouse_energy,ignore_index = True)
    ulsan=ulsan.append(ulsan_energy,ignore_index = True)

dangjin.sort_values(by=['시간'],ascending=True, inplace=True)
dangjin_floating.sort_values(by=['시간'],ascending=True, inplace=True)
dangjin_warehouse.sort_values(by=['시간'],ascending=True, inplace=True)
ulsan.sort_values(by=['시간'],ascending=True, inplace=True)

dangjin.reset_index(drop=True, inplace=True)
dangjin_floating.reset_index(drop=True, inplace=True)
dangjin_warehouse.reset_index(drop=True, inplace=True)
ulsan.reset_index(drop=True, inplace=True)

for i in range(24):
    dangjin.iloc[i::24,0]=pd.to_datetime('20' + dangjin.iloc[i::24,0].dt.strftime('%y-%m-%d')+ ' ' + str(i) + ':00') + pd.DateOffset(hours=1)
    dangjin_floating.iloc[i::24,0]=pd.to_datetime('20' + dangjin_floating.iloc[i::24,0].dt.strftime('%y-%m-%d')+ ' ' + str(i) + ':00') + pd.DateOffset(hours=1)
    dangjin_warehouse.iloc[i::24,0]=pd.to_datetime('20' + dangjin_warehouse.iloc[i::24,0].dt.strftime('%y-%m-%d')+ ' ' + str(i) + ':00') + pd.DateOffset(hours=1)
    ulsan.iloc[i::24,0]=pd.to_datetime('20' + ulsan.iloc[i::24,0].dt.strftime('%y-%m-%d')+ ' ' + str(i) + ':00') + pd.DateOffset(hours=1)

    dangjin.iloc[i::24,1]=dangjin.iloc[i::24,i+3].astype(float)
    dangjin_floating.iloc[i::24,1]=dangjin_floating.iloc[i::24,i+3].astype(float)
    dangjin_warehouse.iloc[i::24,1]=dangjin_warehouse.iloc[i::24,i+3].astype(float)
    ulsan.iloc[i::24,1]=ulsan.iloc[i::24,i+3]

dangjin=dangjin.iloc[:,0:2]
dangjin_floating=dangjin_floating.iloc[:,0:2]
dangjin_warehouse=dangjin_warehouse.iloc[:,0:2]
ulsan=ulsan.iloc[:,0:2]
dangjin.columns = ['time', 'dangjin']
dangjin_floating.columns = ['time', 'dangjin_floating']
dangjin_warehouse.columns = ['time', 'dangjin_warehouse']
ulsan.columns = ['time', 'ulsan']

new_energy= pd.DataFrame(columns=['time'])
new_energy['time'] = pd.date_range(start='2015-01-01 01:00:00', end='2018-03-01 00:00:00', freq='H')
new_energy = pd.merge(new_energy, dangjin_floating, on=['time'], how='outer')
new_energy = pd.merge(new_energy, dangjin_warehouse, on=['time'],how='outer')
new_energy = pd.merge(new_energy, dangjin, on=['time'], how='outer')
new_energy = pd.merge(new_energy, ulsan, on=['time'],how='outer')
display(new_energy)

# groupby를 통해 평균처리하여 중복 데이터를 제거
dangjin_obs_2015_2018 = dangjin_obs_2015_2018.groupby("time", as_index = False).mean()
ulsan_obs_2015_2018 = ulsan_obs_2015_2018.groupby("time", as_index = False).mean()

# csv file upload
new_energy.to_csv("energy_2015_2018.csv")
dangjin_obs_2015_2018.to_csv("dangjin_obs_2015_2018.csv")
ulsan_obs_2015_2018.to_csv("ulsan_obs_2015_2018.csv")


 # function for day_of_year
def day_of_year(datetime): #pd.datetime
    return pd.Period(datetime, freq='D').dayofyear

# function
def add_seasonality(df):
    new_df = df.copy()
    
    new_df['Day_cos'] = new_df['time'].apply(lambda x: np.cos(x.hour * (2 * np.pi) / 24))
    new_df['Day_sin'] = new_df['time'].apply(lambda x: np.sin(x.hour * (2 * np.pi) / 24))

    new_df['Year_cos'] = new_df['time'].apply(lambda x: np.cos(day_of_year(x) * (2 * np.pi) / 365))
    new_df['Year_sin'] = new_df['time'].apply(lambda x: np.sin(day_of_year(x) * (2 * np.pi) / 365))

    return new_df





