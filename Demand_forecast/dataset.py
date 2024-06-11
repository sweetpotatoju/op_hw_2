import pandas as pd
from sklearn.preprocessing import LabelEncoder
from geneticalgorithm import geneticalgorithm as ga
import numpy as np

# 데이터 불러오기
data1 = pd.read_csv("C:\\Dataset\\merged_bike_all.csv", encoding='latin1')

# 라벨링 함수 정의
def label_temperature(temp):
    if temp < 0:
        return 'Very Cold'
    elif temp < 10:
        return 'Cold'
    elif temp < 20:
        return 'Mild'
    elif temp < 30:
        return 'Warm'
    else:
        return 'Hot'

def label_humidity(humidity):
    if humidity < 30:
        return 'Dry'
    elif humidity < 60:
        return 'Normal'
    else:
        return 'Humid'

def label_wind_speed(wind_speed):
    if wind_speed < 2:
        return 'Calm'
    elif wind_speed < 5:
        return 'Breeze'
    elif wind_speed < 10:
        return 'Windy'
    else:
        return 'Strong Wind'

def label_visibility(visibility):
    if visibility < 200:
        return 'Poor'
    elif visibility < 500:
        return 'Moderate'
    else:
        return 'Good'

def label_dew_point(dew_point):
    if dew_point < 0:
        return 'Very Low'
    elif dew_point < 10:
        return 'Low'
    elif dew_point < 20:
        return 'Comfortable'
    else:
        return 'High'

def label_solar_radiation(solar_radiation):
    if solar_radiation < 5:
        return 'Low'
    elif solar_radiation < 15:
        return 'Moderate'
    else:
        return 'High'

def label_rainfall(rainfall):
    if rainfall == 0:
        return 'No Rain'
    elif rainfall < 5:
        return 'Light Rain'
    else:
        return 'Heavy Rain'

def label_snowfall(snowfall):
    if snowfall == 0:
        return 'No Snow'
    elif snowfall < 5:
        return 'Light Snow'
    else:
        return 'Heavy Snow'

def label_season(season):
    if season == 'Spring':
        return 'Spring'
    elif season == 'Summer':
        return 'Summer'
    elif season == 'Autumn':
        return 'Autumn'
    elif season == 'Winter':
        return 'Winter'

def label_binary(value):
    return 'Yes' if value == 'Y' else 'No'

# 라벨링 적용
data1['Temperature_Label'] = data1['Temperature(ÃÂ°C)'].apply(label_temperature)
data1['Humidity_Label'] = data1['Humidity(%)'].apply(label_humidity)
data1['Wind_Speed_Label'] = data1['Wind speed (m/s)'].apply(label_wind_speed)
data1['Visibility_Label'] = data1['Visibility (10m)'].apply(label_visibility)
data1['Dew_Point_Label'] = data1['Dew point temperature(ÃÂ°C)'].apply(label_dew_point)
data1['Solar_Radiation_Label'] = data1['Solar Radiation (MJ/m2)'].apply(label_solar_radiation)
data1['Rainfall_Label'] = data1['Rainfall(mm)'].apply(label_rainfall)
data1['Snowfall_Label'] = data1['Snowfall (cm)'].apply(label_snowfall)
data1['Season_Label'] = data1['Seasons'].apply(label_season)
data1['Holiday_Label'] = data1['Holiday'].apply(label_binary)
data1['Functioning_Day_Label'] = data1['Functioning Day'].apply(label_binary)

# 라벨 인코딩
label_encoders = {}
for column in ['Temperature_Label', 'Humidity_Label', 'Wind_Speed_Label', 'Visibility_Label', 'Dew_Point_Label',
               'Solar_Radiation_Label', 'Rainfall_Label', 'Snowfall_Label', 'Season_Label', 'Holiday_Label', 'Functioning_Day_Label']:
    le = LabelEncoder()
    data1[column] = le.fit_transform(data1[column])
    label_encoders[column] = le

# 기존 데이터의 라벨링된 열 업데이트
data1['Temperature(ÃÂ°C)'] = data1['Temperature_Label']
data1['Humidity(%)'] = data1['Humidity_Label']
data1['Wind speed (m/s)'] = data1['Wind_Speed_Label']
data1['Visibility (10m)'] = data1['Visibility_Label']
data1['Dew point temperature(ÃÂ°C)'] = data1['Dew_Point_Label']
data1['Solar Radiation (MJ/m2)'] = data1['Solar_Radiation_Label']
data1['Rainfall(mm)'] = data1['Rainfall_Label']
data1['Snowfall (cm)'] = data1['Snowfall_Label']
data1['Seasons'] = data1['Season_Label']
data1['Holiday'] = data1['Holiday_Label']
data1['Functioning Day'] = data1['Functioning_Day_Label']

# 라벨링된 열 삭제
data1 = data1.drop(columns=[
    'Temperature_Label', 'Humidity_Label', 'Wind_Speed_Label', 'Visibility_Label',
    'Dew_Point_Label', 'Solar_Radiation_Label', 'Rainfall_Label', 'Snowfall_Label',
    'Season_Label', 'Holiday_Label', 'Functioning_Day_Label'
])

# 'Rental Station Name' 열 삭제
data1 = data1.drop(columns=['Rental Station Name'])

# 'Rental Hour' 열 삭제
data1 = data1.drop(columns=['Rental Hour'])

# 모든 열을 출력하도록 설정 변경
pd.set_option('display.max_columns', None)

# 변환된 데이터 출력
print(data1.head())


