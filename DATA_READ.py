import pandas as pd

# CSV 파일 경로
file_path = "C:\\Dataset\\merged_bike_all_cleaned.csv"

# 전체 컬럼 값을 다 보이게 설정
pd.set_option('display.max_columns', None)

# CSV 파일 읽기
data = pd.read_csv(file_path)


# 데이터 출력
print(data)
