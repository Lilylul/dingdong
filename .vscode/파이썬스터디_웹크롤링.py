'''
get() 함수 사용해서 네이버 주식사이트에서 HTML 코드 가져오기
그리고 상태 코드와 HTML 일부 출력하기
'''

import requests
import sys
sys.stdout.reconfigure(encoding = 'utf-8')

# 2단계: GET 요청 보내기
response = requests.get('https://finance.naver.com/')

print(f"응답 상태 코드: {response.status_code}")  # 상태 코드 확인

# 응답 성공 여부 확인
if response.status_code == 200:
     #응답 내용 확인
    print(response.text[:1000])  # HTML 일부 출력
else:
    print("요청 실패")

### 네이버 주식사이트에서 종목명 1개 가져오기 ###

# 2단계: 필요 라이브리러 설치
from bs4 import BeautifulSoup

if response.status_code == 200:
    # 3단계: BeautifulSoup 객체 생성
    soup = BeautifulSoup(response.text, 'html.parser')

    # 4단계: 원하는 정보 추출
    # id가 "_topItems1"인 태그의 첫 번째 <tr> 태그를 선택하여 "종목정보" 변수에 저장
    종목정보 = soup.select_one('#_topItems1 tr th a').text #id접근시 "#" 사용, class 접근시 "." 사용
    # "종목정보"에서 <th> 태그 내부 <a> 태그의 텍스트를 추출하여 "종목명" 변수에 저장
    # 종목명 = 종목정보.select_one('th a').text
    print(f"크롤링한 종목명: {종목정보}")
else:
    print(f"페이지 요청 실패: 상태 코드 {response.status_code}")
