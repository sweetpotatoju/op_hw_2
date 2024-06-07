import pandas as pd

# 데이터 불러오기
data1 = pd.read_csv("C:\\Dataset\\transformed_bike_data.csv", encoding='latin1')


# 병합된 데이터프레임의 상위 5개 행을 출력합니다.
print(data1.head())

# 병합된 데이터프레임의 정보(열 타입, null 값 등)를 출력합니다.
print(data1.info())

