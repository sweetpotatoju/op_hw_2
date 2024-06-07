import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm as ga

# 데이터 불러오기
expanded_data_file = "C:\\Dataset\\transformed_bike_data.csv"
data_expanded = pd.read_csv(expanded_data_file, encoding='latin1')

# 컬럼 이름 재설정
data_expanded.columns = [
    'Date', 'Hour', 'Temperature_C', 'Humidity', 'Wind_speed', 'Visibility',
    'Dew_point_temperature_C', 'Solar_Radiation', 'Rainfall', 'Snowfall',
    'Seasons', 'Holiday', 'Functioning_Day', 'Rental_Station_Number'
]

# 데이터프레임 확인
print(data_expanded.head(10))
print(data_expanded.columns)

# 데이터 집계: 시간대별로 각 대여소의 이용 건수를 집계
data_expanded['Rental_count'] = data_expanded.groupby(['Hour', 'Rental_Station_Number'])['Rental_Station_Number'].transform('count')

# 데이터 집계
grouped_data = data_expanded.groupby(['Hour', 'Rental_Station_Number', 'Dew_point_temperature_C',
                                      'Humidity', 'Rainfall', 'Snowfall', 'Solar_Radiation',
                                      'Temperature_C', 'Visibility', 'Wind_speed', 'Seasons',
                                      'Holiday', 'Functioning_Day']).agg({
    'Rental_count': 'sum'
}).reset_index()

print(grouped_data.head())

# 데이터 분포 확인
plt.figure(figsize=(10, 6))
plt.hist(grouped_data['Rental_count'], bins=50, edgecolor='k')
plt.title('Distribution of Rental_count')
plt.xlabel('Rental_count')
plt.ylabel('Frequency')
plt.show()

# 특성 및 타겟 정의
features = ['Hour', 'Rental_Station_Number', 'Dew_point_temperature_C', 'Humidity', 'Rainfall',
            'Snowfall', 'Solar_Radiation', 'Temperature_C', 'Visibility',
            'Wind_speed', 'Seasons', 'Holiday', 'Functioning_Day']
target = 'Rental_count'

# 데이터 분할
X = grouped_data[features]
y = grouped_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 정의 및 학습
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 교차 검증을 통한 모델 평가
lr_scores = cross_val_score(lr_model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validated MSE (Linear Regression): {-lr_scores.mean()}")

# 테스트 데이터셋 예측 및 평가
lr_y_pred = lr_model.predict(X_test)
lr_mse_test = np.mean((y_test - lr_y_pred) ** 2)
print(f"Test MSE (Linear Regression): {lr_mse_test}")

# 실제 값과 예측 값 비교
lr_comparison_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': lr_y_pred})
print(lr_comparison_df.head(10))

# 유전 알고리즘 목표 함수 정의 (선형 회귀 모델 사용)
def fitness_lr(X):
    hour, station_id, dew_point, humidity, rainfall, snowfall, solar_radiation, temp, visibility, wind_speed, seasons, holiday, functioning_day = X
    features_df = pd.DataFrame(
        [[hour, station_id, dew_point, humidity, rainfall, snowfall, solar_radiation, temp, visibility, wind_speed, seasons, holiday, functioning_day]],
        columns=features)
    prediction = lr_model.predict(features_df)
    return -prediction[0]  # 수요를 최대화하기 위해 음수 사용

# 범주형 변수의 고유값 범위 설정
seasons_min, seasons_max = grouped_data['Seasons'].min(), grouped_data['Seasons'].max()
holiday_min, holiday_max = grouped_data['Holiday'].min(), grouped_data['Holiday'].max()
functioning_day_min, functioning_day_max = grouped_data['Functioning_Day'].min(), grouped_data['Functioning_Day'].max()

# 유전 알고리즘을 이용하여 특정 시간대에 최적의 대여소 번호를 찾는 함수
def find_best_station_for_hour_lr(hour, dew_point, humidity, rainfall, snowfall, solar_radiation, temp, visibility,
                                  wind_speed, seasons, holiday, functioning_day):
    varbound = np.array([
        [hour, hour],  # Hour 고정
        [grouped_data['Rental_Station_Number'].min(), grouped_data['Rental_Station_Number'].max()],  # Rental Station Number
        [grouped_data['Dew_point_temperature_C'].min(), grouped_data['Dew_point_temperature_C'].max()],  # Dew point temperature
        [grouped_data['Humidity'].min(), grouped_data['Humidity'].max()],  # Humidity
        [grouped_data['Rainfall'].min(), grouped_data['Rainfall'].max()],  # Rainfall
        [grouped_data['Snowfall'].min(), grouped_data['Snowfall'].max()],  # Snowfall
        [grouped_data['Solar_Radiation'].min(), grouped_data['Solar_Radiation'].max()],  # Solar Radiation
        [grouped_data['Temperature_C'].min(), grouped_data['Temperature_C'].max()],  # Temperature
        [grouped_data['Visibility'].min(), grouped_data['Visibility'].max()],  # Visibility
        [grouped_data['Wind_speed'].min(), grouped_data['Wind_speed'].max()],  # Wind speed
        [seasons_min, seasons_max],  # Seasons
        [holiday_min, holiday_max],  # Holiday
        [functioning_day_min, functioning_day_max]  # Functioning Day
    ])

    algorithm_param = {'max_num_iteration': 200,
                       'population_size': 50,
                       'mutation_probability': 0.1,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': None}

    model_ga = ga(function=fitness_lr, dimension=13, variable_type='real', variable_boundaries=varbound,
                  algorithm_parameters=algorithm_param)

    model_ga.run()
    best_solution = model_ga.output_dict['variable']
    best_station_id = best_solution[1]
    return best_station_id

# 시간대별 최적의 대여소 번호 찾기
best_stations_per_hour = {}
for hour in range(24):
    best_station = find_best_station_for_hour_lr(hour, dew_point=5, humidity=60, rainfall=0, snowfall=0,
                                                 solar_radiation=15, temp=20, visibility=10, wind_speed=3,
                                                 seasons=1, holiday=0, functioning_day=1)
    best_stations_per_hour[hour] = best_station
    print(f"Best station for hour {hour}: {best_station}")

print("Best stations per hour:", best_stations_per_hour)

# 시각화: 시간대별 최적의 대여소
hours = list(best_stations_per_hour.keys())
stations = list(best_stations_per_hour.values())

plt.figure(figsize=(12, 6))
bars = plt.bar(hours, stations, color='skyblue', edgecolor='k')
plt.title('Best Rental Station per Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Rental Station Number')
plt.xticks(hours)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 막대 위에 수치를 표시
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.0f}', va='bottom')

plt.show()

# 시각화: 시간대별 이용 건수
hourly_rental_counts = data_expanded.groupby('Hour')['Rental_count'].sum()

plt.figure(figsize=(12, 6))
plt.plot(hourly_rental_counts.index, hourly_rental_counts.values, marker='o', color='b', linestyle='-')
plt.title('Rental Counts per Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Rental Counts')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(hourly_rental_counts.index)
plt.show()
