import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from geneticalgorithm import geneticalgorithm as ga

# 확장된 데이터 불러오기
expanded_data_file = 'C:\\Dataset\\Expanded_SeoulBike_WeatherData_2018.csv'
data_expanded = pd.read_csv(expanded_data_file, encoding='utf-8-sig')

# 데이터프레임 확인
print(data_expanded.head(10))
print(data_expanded.columns)
print(data_expanded.shape)


# 필요 특성 선택 (한글 피처 이름을 영어로 변경)
data_expanded.columns = ['Date', 'Rental_hour', 'Rental_station_number', 'Rental_station_name', 'Rental_count',
                         'Dew_point_temperature', 'Humidity', 'Rainfall', 'Snowfall', 'Solar_Radiation',
                         'Temperature', 'Visibility', 'Wind_speed']

# 필요 없는 열 'Rental_station_name' 제거
data_expanded = data_expanded.drop(columns=['Rental_station_name'])

# Rental_hour와 Rental_station_number를 기준으로 대여 건수를 집계
grouped_data = data_expanded.groupby(['Rental_hour', 'Rental_station_number',
                                      'Dew_point_temperature', 'Humidity', 'Rainfall', 'Snowfall',
                                      'Solar_Radiation', 'Temperature', 'Visibility', 'Wind_speed']).agg({
    'Rental_count': 'sum'
}).reset_index()

# 데이터 분포 확인
plt.figure(figsize=(10, 6))
plt.hist(grouped_data['Rental_count'], bins=50, edgecolor='k')
plt.title('Distribution of Rental_count')
plt.xlabel('Rental_count')
plt.ylabel('Frequency')
plt.show()

# 특성 및 타겟 정의
features = ['Rental_hour', 'Rental_station_number', 'Dew_point_temperature', 'Humidity', 'Rainfall', 'Snowfall',
            'Solar_Radiation', 'Temperature', 'Visibility', 'Wind_speed']
target = 'Rental_count'

# 데이터 분할 및 스케일링
X = grouped_data[features]
y = grouped_data[target]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 회귀 모델 정의 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 교차 검증을 통한 모델 평가
scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validated MSE: {-scores.mean()}")

# 테스트 데이터셋 예측 및 평가
y_pred = model.predict(X_test)
mse_test = np.mean((y_test - y_pred)**2)
print(f"Test MSE: {mse_test}")

# 실제 값과 예측 값 비교
comparison_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print(comparison_df.head(10))

# 회귀 계수 확인
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})

print(coefficients)

# 시간대별 실제 대여 건수 분포 확인
plt.figure(figsize=(12, 6))
plt.hist(data_expanded['Rental_count'], bins=50, edgecolor='k')
plt.title('Distribution of Rental_count')
plt.xlabel('Rental_count')
plt.ylabel('Frequency')
plt.show()

# 시간대별 평균 대여 건수 확인
average_rentals_by_hour = data_expanded.groupby('Rental_hour')['Rental_count'].mean()
plt.figure(figsize=(12, 6))
average_rentals_by_hour.plot(kind='bar')
plt.title('Average Rentals by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Rentals')
plt.grid(True)
plt.show()

# 유전 알고리즘 목표 함수 정의 (fitness 함수)
def fitness(hour, station_id):
    features_df = pd.DataFrame(
        [[hour, station_id, dew_point, humidity, rainfall, snowfall, solar_radiation, temp, visibility, wind_speed]],
        columns=features)
    features_scaled = scaler.transform(features_df)
    prediction = model.predict(pd.DataFrame(features_scaled, columns=features))
    return -prediction[0]  # 수요를 최대화하기 위해 음수 사용

# 유전 알고리즘을 이용하여 특정 시간대에 최적의 대여소 번호를 찾는 함수
def find_best_station_for_hour(hour, dew_point, humidity, rainfall, snowfall, solar_radiation, temp, visibility,
                               wind_speed):
    def fitness_station(station_id_array):
        station_id = int(station_id_array[0])  # 배열의 첫 번째 요소로부터 정수형 대여소 번호를 가져옴
        return fitness(hour, station_id)

    varbound_station = np.array(
        [[min(X['Rental_station_number']), max(X['Rental_station_number'])]])  # station_id 범위 설정

    algorithm_param_station = {
        'max_num_iteration': 300,
        'population_size': 100,
        'mutation_probability': 0.2,
        'elit_ratio': 0.02,
        'crossover_probability': 0.7,
        'parents_portion': 0.4,
        'crossover_type': 'uniform',
        'max_iteration_without_improv': 50
    }

    model_ga_station = ga(function=fitness_station, dimension=1, variable_type='int',
                          variable_boundaries=varbound_station,
                          algorithm_parameters=algorithm_param_station)

    model_ga_station.run()

    best_station = int(model_ga_station.output_dict['variable'][0])  # 정수형 대여소 번호 반환
    best_station_predicted_demand = -model_ga_station.output_dict['function']
    return best_station, best_station_predicted_demand

# 시간대별 최적의 대여소 번호를 찾기 위한 함수
def find_best_stations_by_hour(dew_point, humidity, rainfall, snowfall, solar_radiation, temp, visibility, wind_speed):
    best_stations = {}
    for hour in range(24):
        best_station, predicted_demand = find_best_station_for_hour(hour, dew_point, humidity, rainfall, snowfall,
                                                                    solar_radiation, temp, visibility, wind_speed)
        best_stations[hour] = (best_station, predicted_demand)
    return best_stations

dew_point = 15
humidity = 50
rainfall = 1
snowfall = 1
solar_radiation = 20
temp = 25
visibility = 1500
wind_speed = 3

best_stations = find_best_stations_by_hour(dew_point, humidity, rainfall, snowfall, solar_radiation, temp, visibility,
                                           wind_speed)
for hour, (station, demand) in best_stations.items():
    print(f"시간대 {hour}시: 최적의 대여소 번호: {station}, 예측된 수요: {demand}")

# 시간대별 최적의 대여소 번호와 예측된 수요를 저장하는 데이터프레임 생성
results = pd.DataFrame.from_dict(best_stations, orient='index', columns=['Best_Station', 'Predicted_Demand'])
results.index.name = 'Hour'
results.reset_index(inplace=True)

# 시간대별 최적의 대여소 번호 시각화
plt.figure(figsize=(12, 6))
plt.plot(results['Hour'], results['Best_Station'], marker='o', linestyle='-', color='b')
plt.title('Best Station by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Best Station Number')
plt.grid(True)
plt.show()

# 시간대별 예측된 수요 시각화
plt.figure(figsize=(12, 6))
plt.plot(results['Hour'], results['Predicted_Demand'], marker='o', linestyle='-', color='r')
plt.title('Predicted Demand by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Predicted Demand')
plt.grid(True)
plt.show()
