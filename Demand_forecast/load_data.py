# load_data.py
import pandas as pd


def load_data(file_path):
    data_expanded = pd.read_csv(file_path, encoding='latin1')
    data_expanded.columns = [
        'Date', 'Hour', 'Temperature_C', 'Humidity', 'Wind_speed', 'Visibility',
        'Dew_point_temperature_C', 'Solar_Radiation', 'Rainfall', 'Snowfall',
        'Seasons', 'Holiday', 'Functioning_Day', 'Rental_Station_Number'
    ]
    data_expanded['Rental_count'] = data_expanded.groupby(['Hour', 'Rental_Station_Number'])[
        'Rental_Station_Number'].transform('count')
    grouped_data = data_expanded.groupby(['Hour', 'Rental_Station_Number', 'Dew_point_temperature_C',
                                          'Humidity', 'Rainfall', 'Snowfall', 'Solar_Radiation',
                                          'Temperature_C', 'Visibility', 'Wind_speed', 'Seasons',
                                          'Holiday', 'Functioning_Day']).agg({
        'Rental_count': 'sum'
    }).reset_index()
    return grouped_data


if __name__ == "__main__":
    file_path = "C:\\Dataset\\transformed_bike_data.csv"
    # 전체 컬럼 값을 다 보이게 설정
    pd.set_option('display.max_columns', None)

    data = load_data(file_path)
    print(data.head())
