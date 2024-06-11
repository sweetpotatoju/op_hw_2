import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd


class LinUCB:
    def __init__(self, alpha, n_features):
        self.alpha = alpha
        self.n_features = n_features
        self.A = np.identity(n_features)
        self.b = np.zeros(n_features)

    def fit(self, X, y):
        self.A += np.dot(X.T, X)
        self.b += np.dot(X.T, y)
        self.theta = np.dot(np.linalg.pinv(self.A), self.b)  # Use pseudo-inverse for numerical stability

    def predict(self, X):
        p = np.dot(X, self.theta) + self.alpha * np.sqrt((X @ np.linalg.pinv(self.A) @ X.T).diagonal())
        return p


def linucb(data, hour, features, alpha=0.1):
    station_ids = data['Rental_Station_Number'].unique()
    n_features = len(features)
    model = LinUCB(alpha, n_features)

    # Normalize features for each station
    scaler = StandardScaler()

    for h in range(hour + 1):
        hourly_data = data[data['Hour'] == h]

        for station_id in station_ids:
            station_data = hourly_data[hourly_data['Rental_Station_Number'] == station_id]
            if station_data.empty:
                continue
            X = station_data[features].values
            y = station_data['Rental_count'].values
            X = scaler.fit_transform(X)  # Normalize features
            model.fit(X, y)

    hourly_data = data[data['Hour'] == hour]
    X_test = hourly_data[features].values
    X_test = scaler.fit_transform(X_test)
    p = model.predict(X_test)

    best_station_idx = np.argmax(p)
    best_station_id = hourly_data.iloc[best_station_idx]['Rental_Station_Number']
    best_station_demand = p[best_station_idx]
    return best_station_id, best_station_demand


def find_best_stations_linucb(data, features, alpha=0.1):
    best_stations_linucb = {}
    best_station_demands = {}
    for hour in range(24):
        best_station, best_demand = linucb(data, hour, features, alpha)
        best_stations_linucb[hour] = best_station
        best_station_demands[hour] = best_demand
        print(f"Best station for hour {hour} (LINUCB): {best_station} with predicted demand: {best_demand:.2f}")
    return best_stations_linucb, best_station_demands


if __name__ == "__main__":
    file_path = "C:\\Dataset\\transformed_bike_data.csv"
    data = pd.read_csv(file_path, encoding='latin1')

    data.columns = [
        'Date', 'Hour', 'Temperature_C', 'Humidity', 'Wind_speed', 'Visibility',
        'Dew_point_temperature_C', 'Solar_Radiation', 'Rainfall', 'Snowfall',
        'Seasons', 'Holiday', 'Functioning_Day', 'Rental_Station_Number'
    ]

    data['Rental_count'] = data.groupby(['Hour', 'Rental_Station_Number'])['Rental_Station_Number'].transform('count')

    grouped_data = data.groupby(['Hour', 'Rental_Station_Number', 'Dew_point_temperature_C',
                                 'Humidity', 'Rainfall', 'Snowfall', 'Solar_Radiation',
                                 'Temperature_C', 'Visibility', 'Wind_speed', 'Seasons',
                                 'Holiday', 'Functioning_Day']).agg({
        'Rental_count': 'sum'
    }).reset_index()

    features = ['Dew_point_temperature_C', 'Humidity', 'Rainfall', 'Snowfall',
                'Solar_Radiation', 'Temperature_C', 'Visibility', 'Wind_speed',
                'Seasons', 'Holiday', 'Functioning_Day']

    best_stations_linucb, best_station_demands = find_best_stations_linucb(grouped_data, features, alpha=1.0)
    print("Best stations per hour (LINUCB):", best_stations_linucb)

    hours = list(best_stations_linucb.keys())
    stations = list(best_stations_linucb.values())
    demands = list(best_station_demands.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(hours, demands, color='skyblue', edgecolor='k')
    plt.title('Best Rental Station per Hour (LINUCB) Based on Predicted Demand')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Predicted Rental Demand')
    plt.xticks(hours)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar, station in zip(bars, stations):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{station}\n{yval:.2f}', ha='center', va='bottom')

    plt.show()

    hourly_rental_counts = data.groupby('Hour')['Rental_count'].sum()

    plt.figure(figsize=(12, 6))
    plt.plot(hourly_rental_counts.index, hourly_rental_counts.values, marker='o', color='b', linestyle='-')
    plt.title('Rental Counts per Hour')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Rental Counts')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(hourly_rental_counts.index)
    plt.show()
