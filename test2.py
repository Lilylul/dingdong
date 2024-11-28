# # 네이버 검색 API 예제 - 블로그 검색
# import os
# import sys
# import urllib.request
# client_id = "CM4HtuSunG5jDMvL898n"
# client_secret = "gctnPVNNLD"
# encText = urllib.parse.quote("경북궁")
# url = "https://openapi.naver.com/v1/search/blog?query=" + encText # JSON 결과
# # url = "https://openapi.naver.com/v1/search/blog.xml?query=" + encText # XML 결과
# request = urllib.request.Request(url)
# request.add_header("X-Naver-Client-Id",client_id)
# request.add_header("X-Naver-Client-Secret",client_secret)
# response = urllib.request.urlopen(request)
# rescode = response.getcode()
# if(rescode==200):
#     response_body = response.read()
#     print(response_body.decode('utf-8'))
# else:
#     print("Error Code:" + rescode)

# import os
# import sys
# import urllib.request
# import json
# import pandas as pd

# def getresult(client_id,client_secret,query,display=10,start=1,sort='sim'):
#     #sort : sim(정확도순으로 내림차순 정렬(기본값)), date(날짜순으로 내림차순 정렬)
#     encText = urllib.parse.quote(query)
#     url = "https://openapi.naver.com/v1/search/blog?query=" + encText + \
#     "&display=" + str(display) + "&start=" + str(start) + "&sort=" + sort

#     request = urllib.request.Request(url)
#     request.add_header("X-Naver-Client-Id",client_id)
#     request.add_header("X-Naver-Client-Secret",client_secret)
#     response = urllib.request.urlopen(request)
#     rescode = response.getcode()
#     if(rescode==200):
#         response_body = response.read()
#         response_json = json.loads(response_body)
#     else:
#         print("Error Code:" + rescode)

#     return pd.DataFrame(response_json['items'])

# import pandas as pd
# client_id = "siVxxON3yoYD5ROOJyrj"
# client_secret = "aprvODVlkZ"
# query = '경복궁'
# display=100
# start=1
# sort='sim'

# result_all=pd.DataFrame()
# for i in range(0,2):
#     start= 1 + 100*i
#     result= getresult(client_id,client_secret,query,display,start,sort)
#     result_all=pd.concat([result_all,result])
# result_all

#pip install regex
# '.' 은 한개의 임의의 문자를 나타냄.
# import re
# r = re.compile('a.c') #a 와 c 사이에 어떤 1개의 문자라도 올수있는것
# print(r.search('kkk'))#아무런 출력이 없음
# print(r.search('abc'))
# print(r.search('abbbbbbbbbbbbbbbc'))
# # span=(0,3): 찾고자 하는 문자열은 3글자

# #?: ?앞의 문자가 존재할 수도 있고, 존재하지 않을 수도 있는 경우를 나타냄.
# #ex) ab?c라고 한다면,  b는 있을수도 있고 없다고 취급할수도 있다.
# r = re.compile('ab?c')
# print(r.search('abbc')) #아무런 출력이 되지않음
# print(r.search('abc')) #b가 있는것을 판단하여 abc를 매치함
# print(r.search('ac'))

# #* 기호: 바로앞의 문자가 0개 이상인 경우
# r = re.compile('ab*c')
# print(r.search('a'))
# print(r.search('ac'))
# print(r.search('abbc'))

# #+기호: *와 유사함. 다른점: 앞의 문자가 최소 1개 이상의 경우
# r = re.compile('ab+c')
# print(r.search('ac')) #아무것도 출력안됨
# print(r.search('abc'))

# #^: 시작되는 글자를 지정
# r = re.compile('^a')
# print(r.search('bbc')) #아무것도 나오지않음
# print(r.search('ab'))

# [] 기호: 문자들을 넣으면 그문자들중 한개의 문자와 매치되는 의미를 가짐.
#[a-c] #[abc]
#[0-5] #[012345]
#[a-zA-Z] #모든 알파벳(28,26)
#[^0-9]숫자/전화 번호 찾을 수있는 정규표현식?
#[a-c] #[abc]
import re
text = '전화번호는 010-1234-5678 연락해'
pat = re.compile('\d{3}-\d{4}-\d{4}')
phone_list = pat.findall(text)
print(phone_list)

pat2 = re.compile('[0-9]{3}-[0-9]{4}-[0-9]{4}')
phone_list = pat2.findall(text)
print(phone_list)