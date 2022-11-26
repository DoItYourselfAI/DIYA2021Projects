# code for private LB

import pandas as pd
import numpy as np
import urllib
import urllib.request
import json

from pandas.core.indexing import convert_to_index_sliceable


def date_ctrl(date, shift, format):
    """
    @params
    date: string; format supported by pd.Timestamp
    shift: int; days to shift forward
    format: str; "pandas", "json", or other date format

    @return
    str; date format
    """

    date = pd.Timestamp(date) + pd.Timedelta(days=shift)
    if format == "pandas":
        return date.strftime("%Y-%m-%d")
    elif format == "json":
        return date.strftime("%Y%m%d")
    else:
        return date.strftime(format)


class API:
    def __init__(self, base_date, base_time, location, key):
        """
        base_date: str, format:'YYYYMMDD'(e.g. '20210522'). 예보 발표일자. API로는 최근 1일간의 자료만을 불러올 수 있다는 점 유의.
        base_time: str, format: 'HHMM'(e.g. '2000' for 20시). 예보 발표시각. 0200, 0500, 0800, 1100, 1400, 1700, 2000, 2300 중 하나여야 함.
        location: str, 'dangjin' or 'ulsan'.
        key: 기상청 API 일반 인증키(decoding)
        """
        self.base_date = base_date
        self.base_time = base_time
        self.key = key
        self.location = location

        if self.location == "dangjin":
            self.nx = "53"
            self.ny = "114"
        elif self.location == "ulsan":
            self.nx = "102"
            self.ny = "83"
        else:
            raise Exception("Location should be either 'dangjin' or 'ulsan'")

    def get_data(self, gap=1, preprocess=True, itp_method="linear"):
        """
        gap: int; 몇 일 뒤를 예측할 것인지. Default = 1
        preprocess: bool, True by default. Set preprocess = False to get raw data.
        itp_method: str, interpolation method supported by pandas.DataFrame.interpolate. Commonly used are 'linear' and 'quadratic'.
        """
        # 예측 대상 일자 (base_date + 1day)
        self.predict_date_json = date_ctrl(self.base_date, gap, "json")
        self.predict_date_pandas = date_ctrl(self.base_date, gap, "pandas")

        self.predict_date_tmrw_json = date_ctrl(
            self.base_date, gap + 1, "json")
        self.predict_date_tmrw_pandas = date_ctrl(
            self.base_date, gap + 1, "pandas")

        # pd DataFrame
        fcst_df = pd.DataFrame()
        fcst_df["time"] = [
            f"{self.predict_date_pandas} {hour}:00:00" for hour in range(25)
        ]  # not 24; for better interpolation

        response = self._api()
        row_idx = 0

        for data in response["response"]["body"]["items"]["item"]:
            if (data["fcstDate"] == self.predict_date_json) or (
                (data["fcstDate"] == self.predict_date_tmrw_json)
                and (data["fcstTime"] == "0000")
            ):

                if data["category"] == "REH":
                    fcst_df.loc[row_idx, "Humidity"] = float(data["fcstValue"])
                elif data["category"] == "T3H":
                    fcst_df.loc[row_idx, "Temperature"] = float(
                        data["fcstValue"])
                elif data["category"] == "SKY":
                    fcst_df.loc[row_idx, "Cloud"] = float(data["fcstValue"])
                elif data["category"] == "VEC":
                    fcst_df.loc[row_idx, "WindDirection"] = float(
                        data["fcstValue"])
                elif data["category"] == "WSD":
                    fcst_df.loc[row_idx, "WindSpeed"] = float(
                        data["fcstValue"])

                    # because WSD always comes at last
                    row_idx += 3

        if preprocess:
            fcst_df = self._preprocess(fcst_df, itp_method)

        return fcst_df

    # for internal use
    def _preprocess(self, data, itp_method):
        # cloud preprocess to match the unit in obs data
        data = self._preprocess_cloud(data)

        # interpolate
        data = data.interpolate(method=itp_method)

        # convert 24:00:00 to tomorrow 00:00:00
        data.loc[24, "time"] = f"{self.predict_date_tmrw_pandas} 00:00:00"

        # to_datetime
        data["time"] = pd.to_datetime(data["time"])

        # add seasonality
        data = self._add_seasonality(data)

        # preprocess 'WindSpeed' and 'WindDirection' to 'Wind_X' and 'Wind_Y'
        data = data.join(self._preprocess_wind(data))
        data.drop(columns=["WindSpeed", "WindDirection"], inplace=True)

        # exclude first row
        data = data.drop(index=0)

        return data

    def _api(self):
        url = "http://apis.data.go.kr/1360000/VilageFcstInfoService/getVilageFcst"

        queryParams = "?" + urllib.parse.urlencode(
            {
                urllib.parse.quote_plus("serviceKey"): self.key,
                urllib.parse.quote_plus(
                    "numOfRows"
                ): "300",  # 그냥 넉넉하게 설정; 데이터가 뒤에서 잘렸다면 이 값을 높여야 함.
                urllib.parse.quote_plus("dataType"): "JSON",
                urllib.parse.quote_plus("base_date"): self.base_date,
                urllib.parse.quote_plus("base_time"): self.base_time,
                urllib.parse.quote_plus("nx"): self.nx,
                urllib.parse.quote_plus("ny"): self.ny,
            }
        )

        response = urllib.request.urlopen(url + queryParams).read()
        response = json.loads(response)

        return response

    def _day_of_year(self, datetime):  # pd.datetime
        return pd.Period(datetime, freq="D").dayofyear

    def _add_seasonality(self, df):
        new_df = df.copy()

        new_df["Day_cos"] = new_df["time"].apply(
            lambda x: np.cos(x.hour * (2 * np.pi) / 24))
        new_df["Day_sin"] = new_df["time"].apply(
            lambda x: np.sin(x.hour * (2 * np.pi) / 24))
        new_df["Year_cos"] = new_df["time"].apply(
            lambda x: np.cos(self._day_of_year(x) * (2 * np.pi) / 365))
        new_df["Year_sin"] = new_df["time"].apply(
            lambda x: np.sin(self._day_of_year(x) * (2 * np.pi) / 365))

        return new_df

    def _preprocess_wind(self, data):
        """
        data: pd.DataFrame which contains the columns 'WindSpeed' and 'WindDirection'
        """

        # degree to radian
        wind_direction_radian = data["WindDirection"] * np.pi / 180

        # polar coordinate to cartesian coordinate
        wind_x = data["WindSpeed"] * np.cos(wind_direction_radian)
        wind_y = data["WindDirection"] * np.sin(wind_direction_radian)

        # name pd.series
        wind_x.name = "Wind_X"
        wind_y.name = "Wind_Y"

        return wind_x, wind_y

    def _preprocess_cloud(self, data):
        # get dictionary to convert from cloud_fcst to cloud_obs
        if self.location == 'dangjin':
            convert_cloud = {1.0: 2.7635372029606544,
                             2.0: 3.8820678513731823,
                             3.0: 6.18494516450648,
                             4.0: 7.961345381526105}
        elif self.location == 'ulsan':
            convert_cloud = {2.0: 3.5910064239828694,
                             1.0: 1.721059516023545,
                             3.0: 6.145117540687161,
                             4.0: 8.638197424892704}

        new_data = data.copy()
        new_data['Cloud'].replace(convert_cloud, inplace=True)

        return new_data
