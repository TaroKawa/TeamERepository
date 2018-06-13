import os
import shutil
import time
import sys
from select import webdriver
from selenium.webdriver.common.keys import Keys
from pyvirtualdisplay import Display
from bs4 import BeautifulSoup
from urllib.request import urlretrieve


def access():
    disp = Display(visible=0, size=(800, 600))
    disp.start()

    print("========================================")
    print("Start query")
    if os.name == "nt":
        driver_path = "../windows/chromedriver.exe"
    elif os.name == "posix":
        driver_path = "../linux/chromedriver"
    elif os.name == "darwin":
        driver_path = "../mac/chromedriver"

    driver = webdriver.Chrome(executable_path=driver_path)
    driver.implicitly_wait(10)
    driver.get("https://www.google.co.jp/imghp?hl=ja")
    return driver, disp


def dir_manage(tag, basecolor):
    file_path = f"../../data/{basecolor}/{tag}"
    if os.path.isdir(file_path):
        shutil.rmtree(file_path)

    os.mkdir(file_path)
    return file_path


def get_tag(driver, query):
    input_query = driver.find_element_by_xpath('//*[@id="lst-ib"]')
    input_query.send_keys(query)
    input_query.send_keys(Keys.ENTER)

    
