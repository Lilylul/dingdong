from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By #find_element 함수를 쉽게 쓰기위함.
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from selenium.common.exceptions import NoSuchElementException
import time
import sys
sys.stdout.reconfigure(encoding = 'utf-8')

chrome_options = Options()
chrome_options.add_experimental_option("detach",True)
chrome_options.add_argument("--lang=ko_KR")

driver = webdriver.Chrome(options= chrome_options)
driver.get('https://www.11st.co.kr/browsing/BestSeller.tmall?method=getBestSellerMain&xfrom=main^gnb')

SCROLL_PAUSE_SEC = 1
last_height = driver.execute_script('return document.body.scrollHeight')

while True:
    driver.execute_script('window.scrollTo(0,document.body.scrollHeight);')
    time.sleep(SCROLL_PAUSE_SEC)
    new_height = driver.execute_script('return document.body.scrollHeight')
    if new_height == last_height:
        break
    last_height = new_height

#데이터를 저장할 리스트 생성
all_data = []

#lists = driver.find_elements(By.NAME, 'viewtype')
try:
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'bestPrdList')))
    lists = driver.find_element(By.ID, 'bestPrdList').find_elements(By.CLASS_NAME,'viewtype')

    ##thisClick_7711985915 > div.box_pd.ranking_pd > a > div.pname > p
    for list in lists :
        bestlist = list.find_elements(By.TAG_NAME,'li')
        try:
            for item in bestlist:
                rank =item.find_element(By.CLASS_NAME, 'best').text #순번
                Product_name = item.find_element(By.CLASS_NAME,'pname').find_element(By.TAG_NAME,'p').text #제품명
                Price = item.find_element(By.CLASS_NAME,'sale_price').text
                product_url = item.find_element(By.CLASS_NAME,'img_plot').find_element(By.TAG_NAME,'a').get_attribute('href')
                image_url = item.find_element(By.CLASS_NAME,'img_plot').find_element(By.TAG_NAME,'img').get_attribute('src')
                
                #수집한 데이터를 딕셔너리 형태로 리스트에 추가
                all_data.append({
                    "Rank" : rank,
                    "Product name" :Product_name,
                    "Price": Price,
                    "Product URL": product_url,
                    "image_url": image_url
                })
        except NoSuchElementException:
            print('An element was not found in this item.')
except NoSuchElementException:
    print('The product list was not found')
df = pd.DataFrame(all_data)
print(df.head())

#thisClick_7752358014 > div.box_pd.ranking_pd > a > div.img_plot > img
#thisClick_4201058576 > div > a > div.img_plot > img
#thisClick_374861784 > div > a > div.img_plot > img