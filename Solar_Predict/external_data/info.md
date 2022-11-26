### dangjin_obs_2021-02.csv / ulsan_obs_2021-02.csv
- **parameter로 encoding 추가해야함**
- e.g. pd.read_csv(PATH, encoding='euc-kr')
- 2월 27일까지만 받아온 이유: 2월 28일이 마지막 예측 대상일. 즉, 예측이 마지막으로 이루어지는 일자는 2월 27일. 2월 27일에는 아무리 많아봐야 2월 27일까지의 obs_data만 알 수 있다.