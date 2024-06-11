import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import random

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
data_expanded['Rental_count'] = data_expanded.groupby(['Hour', 'Rental_Station_Number'])[
    'Rental_Station_Number'].transform('count')

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
features = ['Dew_point_temperature_C', 'Humidity', 'Rainfall', 'Snowfall',
            'Solar_Radiation', 'Temperature_C', 'Visibility',
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


# 시뮬레이티드 어닐링 알고리즘을 사용하여 최적의 대여소를 선택하는 함수
def simulated_annealing(hourly_data, model, feature_names):
    station_ids = hourly_data['Rental_Station_Number'].unique()

    # 초기 해 설정
    current_station_id = random.choice(station_ids)
    best_station_id = current_station_id

    def get_reward(station_id):
        station_data = hourly_data[hourly_data['Rental_Station_Number'] == station_id]
        if station_data.empty:
            return 0

        # 컨텍스트(피처) 정의
        context = station_data[feature_names].mean().values.reshape(1, -1)
        context_df = pd.DataFrame(context, columns=feature_names)
        reward = model.predict(context_df)[0]
        return reward

    current_reward = get_reward(current_station_id)
    best_reward = current_reward

    # 시뮬레이티드 어닐링 파라미터 설정
    T = 1.0  # 초기 온도
    T_min = 0.0001  # 최소 온도
    alpha = 0.9  # 냉각 비율

    while T > T_min:
        # 인접 해 생성
        next_station_id = random.choice(station_ids)
        next_reward = get_reward(next_station_id)

        # 수락 확률 계산
        delta = next_reward - current_reward
        ap = np.exp(min(delta / T, 700))  # 최대 700까지 제한하여 오버플로우 방지

        # 새로운 해 수락
        if next_reward > current_reward or random.uniform(0, 1) < ap:
            current_station_id = next_station_id
            current_reward = next_reward

        # 최적의 해 갱신
        if current_reward > best_reward:
            best_station_id = current_station_id
            best_reward = current_reward

        # 온도 감소
        T = T * alpha

    return best_station_id, best_reward


# 시간대별 최적의 대여소 번호 찾기
def find_best_stations_sa(data, model, feature_names):
    best_stations_sa = {}
    best_station_demands = {}
    for hour in range(24):
        hourly_data = data[data['Hour'] == hour]
        best_station, best_demand = simulated_annealing(hourly_data, model, feature_names)
        best_stations_sa[hour] = best_station
        best_station_demands[hour] = best_demand
        print(
            f"Best station for hour {hour} (Simulated Annealing): {best_station} with predicted demand: {best_demand:.2f}")
    return best_stations_sa, best_station_demands


best_stations_sa, best_station_demands = find_best_stations_sa(data_expanded, lr_model, features)
print("Best stations per hour (Simulated Annealing):", best_stations_sa)

# 시각화: 시간대별 최적의 대여소
hours = list(best_stations_sa.keys())
stations = list(best_stations_sa.values())
demands = list(best_station_demands.values())

plt.figure(figsize=(12, 6))
bars = plt.bar(hours, demands, color='skyblue', edgecolor='k')
plt.title('Best Rental Station per Hour (Simulated Annealing) Based on Predicted Demand')
plt.xlabel('Hour of the Day')
plt.ylabel('Predicted Rental Demand')
plt.xticks(hours)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar, station in zip(bars, stations):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{station}', ha='center', va='bottom')

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
