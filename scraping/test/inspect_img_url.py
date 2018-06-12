from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from pyvirtualdisplay import Display
import time
from urllib.request import urlretrieve
from bs4 import BeautifulSoup
import os

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

jscontroller = driver.find_elements_by_tag_name("")
imgs = driver.find_elements_by_tag_name("img")
driver.implicitly_wait(10)
print(imgs[59].get_attribute("src"))

src = driver.page_source
soup = BeautifulSoup(src, "html.parser")
imgs = soup.find_all("div", attrs={"jscontroller": "Q7Rsec"})
len(imgs)
imgs[0]
