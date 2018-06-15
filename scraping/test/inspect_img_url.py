from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pyvirtualdisplay import Display
import os
import time
import traceback
import sys
from bs4 import BeautifulSoup
from urllib.request import urlretrieve

os.getcwd()
os.chdir("/home/hidehisa/common/TeamERepository/scraping/test")
disp = Display(visible=1, size=(800, 600))
disp.start()
driver_path = "../linux/chromedriver"
driver = webdriver.Chrome(executable_path=driver_path)
driver.implicitly_wait(2)
driver.get("https://www.google.co.jp/imghp?hl=ja")

input_query = driver.find_element_by_xpath('//*[@id="lst-ib"]')
input_query.send_keys("ワンピース 赤")
input_query.send_keys(Keys.ENTER)

driver.execute_script('scroll(0, document.body.scrollHeight)')
print('Waiting for contents to be loaded...', file=sys.stderr)
time.sleep(2)


imgs = driver.find_elements_by_tag_name("img")
driver.implicitly_wait(10)
print(imgs[24].get_attribute("src"))

src = driver.page_source
soup = BeautifulSoup(src, "html.parser")
imgs = soup.find_all("div", attrs={"jscontroller": "Q7Rsec"})
len(imgs)
len(imgs[0].find_all("img"))
imgs[387].find_all("img")[0].get("src")
