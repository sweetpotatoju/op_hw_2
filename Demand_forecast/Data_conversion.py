import pandas as pd
import glob

# 서울특별시 공공자전거 이용정보 파일 경로 패턴 설정
bike_data_files_pattern = 'C:\\Dataset\\서울특별시 공공자전거 이용정보*.xlsx'
weather_data_file = 'C:\\Dataset\\Cleaned_SeoulBikeData_2018.csv'

# 모든 파일의 경로를 가져옵니다.
bike_data_files = glob.glob(bike_data_files_pattern)

# 빈 데이터프레임 생성
bike_data = pd.DataFrame()

# 모든 Excel 파일을 읽어서 하나의 데이터프레임으로 병합합니다.
for file in bike_data_files:
    temp_data = pd.read_excel(file)
    bike_data = pd.concat([bike_data, temp_data], ignore_index=True)

# 열 이름 수동 지정 (영어로 변경)
column_names = [
    'Rental Date', 'Rental Hour', 'Rental Station Number', 'Rental Station Name',
    'Rental Type', 'Gender', 'Age Group', 'Usage Count', 'Calorie',
    'Carbon', 'Distance', 'Duration'
]
bike_data.columns = column_names

# 특정 열 제거
bike_data.drop(columns=['Rental Type', 'Gender', 'Age Group', 'Calorie', 'Carbon', 'Distance', 'Duration'], inplace=True)

# 이용건수에 따라 행 확장 함수
def expand_rows(df):
    rows = []
    for _, row in df.iterrows():
        count = row['Usage Count']
        for _ in range(count):
            row_copy = row.copy()
            row_copy['Usage Count'] = 1
            rows.append(row_copy)
    return pd.DataFrame(rows)

# 이용건수에 따라 행 확장
bike_data_expanded = expand_rows(bike_data)

# 'Usage Count' 열 제거
bike_data_expanded.drop(columns=['Usage Count'], inplace=True)

# 날씨 데이터 읽기
weather_data = pd.read_csv(weather_data_file, encoding='latin1')

# 날짜 형식 맞추기
weather_data['Date'] = pd.to_datetime(weather_data['Date'])
bike_data_expanded['Rental Date'] = pd.to_datetime(bike_data_expanded['Rental Date'])

# 데이터프레임 병합 (Date와 Hour를 기준으로)
merged_data = pd.merge(weather_data, bike_data_expanded, left_on=['Date', 'Hour'], right_on=['Rental Date', 'Rental Hour'])

# 병합 후 'Rental Date' 열 제거
merged_data.drop(columns=['Rental Date'], inplace=True)

# 병합된 데이터 저장하기 (원하는 경로로 수정)
merged_data.to_csv('C:\\Dataset\\merged_bike_all.csv', index=False)

print("병합된 데이터가 성공적으로 저장되었습니다.")
