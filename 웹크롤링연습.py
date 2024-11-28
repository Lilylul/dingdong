from bs4 import BeautifulSoup
import requests as req
url = 'https://search.naver.com/search.naver'
res = req.get(url, params={'query':'SQLD'})

if res.status_code == 200:
  soup = BeautifulSoup(res.text, 'html.parser')
  #'정보처리기사'가 포함된 텍스트를 모두 찾기
  target_lst = []
  for element in soup.find_all(text=True): #모든 텍스트를 가져옴.
    if 'SQLD' in element:
      target_lst.append(element.strip())

  print(target_lst)
else:
  print(f'요청 실패: {res.status_code}')

import requests as req
from bs4 import BeautifulSoup
import re


#url 설정
url = 'https://search.naver.com/search.naver'

#http요청보내기
try:
  res = req.get(url,params={'query':'정처기'}, timeout=5) #5초 타임아웃 설정
  res.raise_for_status() #응답코드가 200번대가 아닐경우 예외발생
except req.exceptions.RequestException as e:
  print(f'HTTP 요청 중 오류 발생: {e}')
else:
  #성공적으로 데이터를 받았다면, HTML 파싱
  res.encoding = 'utf-8' #한글 인코딩 설정
  soup = BeautifulSoup(res.text, 'html.parser')

  #정규 표현식을 사용하여 "정보처리기사"를 포함한 텍스트 찾기
  target_lst = []
  pattern = re.compile(r'.*정보처리기사.*') #'정보처리기사'가 포함된 텍스트 찾기 위한 정규식 패턴

  #페이지에서 모든 텍스트 추출
  for element in soup.find_all(text=True): #모든텍스트 요소 순회
    if pattern.match(element): #정보처리기사를 포함하는 텍스트만 필터링
       target_lst.append(element.strip()) #공백 제거 후 리스트에 추가
    #   target_lst+=element.strip()
  #결과 출력
  if target_lst:
    print("정보처리기사 관련 텍스트 목록: ")
    for item in target_lst:
      print(item)
  else:
    print("정보처리기사 관련 텍스트를 찾을 수 없습니다.")
