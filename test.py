#import pandas
#실행키는 f5(디버깅 실행): 전체
# ctrl+f5 실행(해당행 실행)

#answer = ''
#n = int(input('0보다 큰 숫자를 입력하세요'))
#while n > 0:
 #   num = n%2
  #  answer = str(num)+answer
   # n = n//2

#print(answer)

# import requests
# res = requests.get('http://naver.com')
# #resquests.get(url)/url에 할당된 웹페이지 정보 가져오기
# print('응답코드: ', res.status_code) #200이면 정상

# if res.status_code == requests.codes.ok:
#     print('정상입니다.')
# else:
#     print('문제가 생겼습니다. [에러코드',res.status_code,']')
# 
# import requests
# from bs4 import BeautifulSoup

# url = 'https://comic.naver.com/webtoon'
# res = requests.get(url)
# print('응답코드 : ', res.status_code)
# print(res.raise_for_status())#문제가 있는지 확인

# soup = BeautifulSoup(res.text,'html')
# print(soup)

# import requests

# res = requests.get('http://google.com')
# res.raise_for_status()

# with open('mygoogle.html','w',encoding='utf-8') as f:
#     f.write(res.text)

#미세먼지
# import requests # requests 라이브러리 설치 필요

# res = requests.get('http://openapi.seoul.go.kr:8088/6d4d776b466c656533356a4b4b5872/json/RealtimeCityAir/1/99')
# resj = res.json()

# print(resj['RealtimeCityAir']['row'][0]['NO2'])

#서울구 이름(msrste_nm)과 해당 구(idex_mvl)에 미세먼지 데이터만 가져와 출력해보자
# citys = resj['RealtimeCityAir']['row']
# for city in citys:
#     gu_name = city['MSRSTE_NM']
#     gu_mise = city['IDEX_MVL']
#     print(gu_name, gu_mise)

#정처기 검색
from bs4 import BeautifulSoup
import requests as req

url = 'http://search.naver.com/search.naver'
res = req.get(url, params={'query':'SQL'})
#get>response의 정보들을 가져온것/ params(파라미터)>쿼리 [....naver/정처기]

if res.status_code ==200:
    #정상적인 반응(200)
    soup = BeautifulSoup(res.text, 'html.parser')
    #response로 받아온 정보들을 html.parser라는것으로 자른다.
    #'정보처리기사' 포함된 텍스트를 모두 찾기.
    target_lst=[]
    #빈공간부터 만듬.
    for element in soup.find_all(text=True): #모든 텍스트를 가져옴.
        #beautifulsoup으로 자른것을 모두 찾아주는(find_all(text이면 모두다))
        #변수 element으로 받음.
        if 'SQL' in element:
            #만약 변수 element안에 '정보처리기사'라는 단어가 있다면,
            target_lst.append(element.strip())
            #target_lst에 쌓아줘.(쌓아주는것은 element.strip())
            # strip():문자열 양쪽에서 공백이나 특정 문자를 제거하는 데 사용
    print(target_lst)
else:
    print(f'요청 실패: {res.status_code}')
