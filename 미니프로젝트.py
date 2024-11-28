import pandas as pd

# CSV 파일 경로
file_path = r'C:\Users\r2com\Desktop\프로젝트\환경부_전기자동차 급속충전기 보급 현황_20231231.csv'

# 데이터 불러오기
try:
    data = pd.read_csv(file_path, encoding='cp949')  # 또는 encoding='euc-kr'
    print(data.head())  # 데이터 상위 5개 출력
except Exception as e:
    print(f"오류 발생: {e}")
