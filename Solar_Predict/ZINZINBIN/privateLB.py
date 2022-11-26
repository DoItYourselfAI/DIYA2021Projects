## code for private LB

import pandas as pd
import numpy as np
import urllib
import urllib.request
import json


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
        key: 기상청 API 일반 인증키(decoding)`
        """
        self.base_date = base_date
        self.base_time = base_time
        self.key = key

        if location == "dangjin":
            self.nx = "53"
            self.ny = "114"
        elif location == "ulsan":
            self.nx = "102"
            self.ny = "83"
        else:
            raise Exception("Location should be either 'dangjin' or 'ulsan'")

        # 예측 대상 일자 (base_date + 1day)
        self.predict_date_json = date_ctrl(base_date, 1, "json")
        self.predict_date_pandas = date_ctrl(base_date, 1, "pandas")

        self.predict_date_tmrw_json = date_ctrl(base_date, 2, "json")
        self.predict_date_tmrw_pandas = date_ctrl(base_date, 2, "pandas")

    def get_data(self, preprocess=True):
        """
        preprocess = True by default. Set preprocess = False to get raw data.
        """
        # pd DataFrame
        fcst_df = pd.DataFrame()
        fcst_df["time"] = [
            f"{self.predict_date_pandas} {hour}:00:00" for hour in range(25)
        ]  # not 24; for better interpolation

        response = self._api()
        row_idx = 0

        for i, data in enumerate(response["response"]["body"]["items"]["item"]):
            if (data["fcstDate"] == self.predict_date_json) or (
                (data["fcstDate"] == self.predict_date_tmrw_json)
                and (data["fcstTime"] == "0000")
            ):

                if data["category"] == "REH":
                    fcst_df.loc[row_idx, "Humidity"] = float(data["fcstValue"])
                elif data["category"] == "T3H":
                    fcst_df.loc[row_idx, "Temperature"] = float(data["fcstValue"])
                elif data["category"] == "SKY":
                    fcst_df.loc[row_idx, "Cloud"] = float(data["fcstValue"])
                elif data["category"] == "VEC":
                    fcst_df.loc[row_idx, "WindDirection"] = float(data["fcstValue"])
                elif data["category"] == "WSD":
                    fcst_df.loc[row_idx, "WindSpeed"] = float(data["fcstValue"])

                    # because WSD always comes at last
                    row_idx += 3

        if preprocess:
            fcst_df = self._preprocess(fcst_df)

        return fcst_df

    # for internal use
    def _preprocess(self, data):
        # interpolate
        data = data.interpolate(method="linear")

        # convert 24:00:00 to tomorrow 00:00:00
        data.loc[24, "time"] = f"{self.predict_date_tmrw_pandas} 00:00:00"

        # to_datetime
        data["time"] = pd.to_datetime(data["time"])

        # add seasonality
        data = self._add_seasonality(data)

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
                ): "150",  # 그냥 넉넉하게 설정; 데이터가 뒤에서 잘렸다면 이 값을 높여야 함.
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
            lambda x: np.cos(x.hour * (2 * np.pi) / 24)
        )
        new_df["Day_sin"] = new_df["time"].apply(
            lambda x: np.sin(x.hour * (2 * np.pi) / 24)
        )

        new_df["Year_cos"] = new_df["time"].apply(
            lambda x: np.cos(self._day_of_year(x) * (2 * np.pi) / 365)
        )
        new_df["Year_sin"] = new_df["time"].apply(
            lambda x: np.sin(self._day_of_year(x) * (2 * np.pi) / 365)
        )

        return new_df
