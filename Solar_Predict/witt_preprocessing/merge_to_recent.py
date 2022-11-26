import pandas as pd

# Last edit on 2021.05.01
# 사용 맥락 확인을 위해서는 witt_preprocessing branch에서 witt_preprocessing/advanced_preprocessing.ipynb 확인


def merge_to_recent(data, time, due_hour=21):
    """
    @warning
    data에 'Forecast time'(기상 예측이 이루어진 시점) column이 존재해야 함.

    @description
    fcst에서 'time'이 중복되는 row들을 하나로 통합한다.
    이때, 가장 최근의 fcst data로 통합한다. (즉, 가장 최근의 fcst data 이외의 row는 모두 삭제한다.)
    '가장 최근'이라는 것은, 해당 'time'의 전날의 due_hour까지다. 
    예를 들어 2021-05-02 14:00:00(=time)를 예측하고, due_hour = 21이라면,
    2021-05-01 21:00:00 전까지의 row 중 가장 최근의 fcst로 통합한다.

    @parameters
    data: pd.DataFrame ; 통합이 이루어질 dataframe
    time: pd.Timestamp ; 통합하려는 time
    due_hour: int ; description 참고

    @return
    pd.Series ; 통합된 하나의 row

    """
    # data only with the given time
    data_only_with_time = data.loc[data["time"] == time]

    # shift time
    time = time - pd.Timedelta(1, unit = "days")  # shifting 1 day to the past (yesterday)
    time = time.replace(hour = due_hour)  # change 'hour' to DUE_HOUR

    # most recent forecast
    return data_only_with_time[data_only_with_time["Forecast time"] <= time].iloc[
        -1:,
    ]

