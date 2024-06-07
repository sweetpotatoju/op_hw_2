import pandas as pd

# 데이터 읽기
bike_data_file = 'C:\\Dataset\\Merged_SeoulBike_WeatherData_2018.csv'
data = pd.read_csv(bike_data_file, encoding='utf-8')

# 불필요한 컬럼 제거
columns_to_drop = ['대여구분코드', '성별', '연령대코드','운동량', '탄소량', '이동거리', '이동시간']
data = data.drop(columns=columns_to_drop)

# 이용건수만큼 행을 확장하는 함수
def expand_rows(df, count_col):
    repeat_df = df.loc[df.index.repeat(df[count_col])].reset_index(drop=True)
    repeat_df[count_col] = 1
    return repeat_df

# 이용건수에 따라 데이터 확장
data_expanded = expand_rows(data, '이용건수')

# 모든 컬럼 출력 설정
pd.set_option('display.max_columns', None)

# 데이터프레임 확인
print(data_expanded.head(10))
print(data_expanded.columns)

# 확장된 데이터 저장
expanded_data_file = 'C:\\Dataset\\Expanded_SeoulBike_WeatherData_2018.csv'
data_expanded.to_csv(expanded_data_file, index=False, encoding='utf-8-sig')

print(f"확장된 데이터를 {expanded_data_file} 파일로 저장했습니다.")
