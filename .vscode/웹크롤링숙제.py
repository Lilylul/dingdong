#1번
# 문제: Books to Scrape 사이트에서 책 제목과 가격 추출
# 목표: https://books.toscrape.com/ 웹사이트에서 상위 10개의 책 제목과 가격을 추출합니다.
# 각 책의 제목과 가격을 출력합니다.
# 요구 사항:
# 웹사이트: https://books.toscrape.com/
# 목표: 웹사이트에서 제공하는 책들의 제목과 가격을 추출하여 출력합니다.
# 제출 내용:
# o    Python 코드
# o    각 책의 제목과 가격을 출력한 결과

import requests
from bs4 import BeautifulSoup
import sys
sys.stdout.reconfigure(encoding='utf-8')

URL1 = 'https://books.toscrape.com/'
top_book_list = []

response = requests.get(URL1)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    book_list = soup.select('section ol li')
    
    # 책 제목과 가격 추출
    for book in book_list:
        # 'h3 a' 선택자에서 첫 번째 a 태그의 텍스트를 추출
        book_title = book.select('h3 a')[0].text.strip()  # 책 제목 추출
        
        # 가격을 추출 (정확한 가격 선택자를 사용)
        book_price = book.select('div p.price_color')[0].text.strip()  # 가격 추출
        
        # 책 제목과 가격을 리스트에 저장
        top_book_list.append({'책 제목': book_title,'책 가격': book_price})

    # 책 제목과 가격을 리스트로 출력
    for book in top_book_list[:10]:
        print(f'책 제목: {book["책 제목"]},책 가격: {book["책 가격"]} ')

else:
    print(f"페이지 요청 실패: 상태 코드 {response.status_code}")



# #2번
# 문제: 네이버 뉴스에서 "Python" 관련 최신 기사 제목과 링크 크롤링
# 목표:
# 네이버 뉴스에서 "Python" 관련 최신 기사를 크롤링하여, 각 기사의 제목과 링크를 추출합니다.
# 상위 10개의 기사를 추출하고, 제목과 링크를 출력합니다.
# 요구 사항:
# 웹사이트: https://search.naver.com/search.naver?&where=news&query=python
# 목표:
# 네이버 뉴스 검색 결과에서 "Python" 관련 기사를 검색하고, 상위 10개의 기사의 제목과 링크를 추출합니다.
# 제출 내용:
# o    Python 코드
# o    상위 10개의 기사의 제목과 링크 출력 결과