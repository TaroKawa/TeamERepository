import os
import shutil
import time
import sys
import yahoo_img_en as en
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from pyvirtualdisplay import Display
from bs4 import BeautifulSoup
from urllib.request import urlretrieve, urlopen


def access():
    disp = Display(visible=0, size=(1920, 1080))
    disp.start()

    print("==============================")
    print("Start query")
    if os.name == "nt":
        driver_path = "../windows/chromedriver.exe"
    elif os.name == "posix":
        driver_path = "../linux/chromedriver"
    elif os.name == "darwin":
        driver_path = "../mac/chromedriver"

    driver = webdriver.Chrome(executable_path=driver_path)
    driver.implicitly_wait(10)
    driver.get("https://images.search.yahoo.com/")
    input = driver.find_element_by_css_selector("#yschsp")
    input.send_keys("robe rouge")
    input.send_keys(Keys.ENTER)
    return driver, disp


def dir_manage(size, color):
    color_dic = {
        "Red": "赤",
        "Orange": "オレンジ",
        "Yellow": "黄色",
        "Green": "緑",
        "Teal": "青緑色",
        "Blue": "青",
        "Purple": "紫",
        "Pink": "桃色",
        "Brown": "茶色",
        "Black": "黒",
        "Gray": "灰色",
        "White": "白",
        "Black & White": "黒白"
    }
    color_jp = color_dic[color]
    file_path = f"../../data/fr/{color_jp}/{size}"
    if not os.path.isdir(f"../../data/fr/{color_jp}"):
        os.mkdir(f"../../data/fr/{color_jp}")
    if os.path.isdir(file_path):
        return False

    os.mkdir(file_path)
    print(f"Made directory {file_path}")
    return file_path


def retreive_img(li, size, color):
    file_path = dir_manage(size, color)
    if not file_path:
        return
    n = len(li)
    for i, l in enumerate(li):
        a_tag = l.find("a")
        src = a_tag.get("href")
        src = "https://images.search.yahoo.com" + src
        try:
            html = urlopen(src)
            soup = BeautifulSoup(html, "html.parser")
            li_inside = soup.find("li", attrs={"class": "initial"})
            img = li_inside.find("img")
            img_src = img.get("src")
            urlretrieve(img_src, f"{file_path}/{i}.jpg")
            time.sleep(3)
            percentage = ((i+1) / n) * 100
            sys.stdout.write(f"\r Finished {percentage:.2f} percent")
            sys.stdout.flush()
        except Exception:
            print(f"Failed to retreive: {src}")


def img_query():
    driver, disp = access()
    driver, color_tags = en.get_available_colors(driver)
    driver, size_tags = en.get_available_sizes(driver)

    n_colors = len(color_tags)
    n_sizes = len(size_tags)

    for i in range(1, n_colors):
        for j in range(1, n_sizes):
            driver, color, size = en.click_tag(driver, i, j)
            driver, li = en.scroll_and_collect(driver)
            retreive_img(li, size, color)


if __name__ == '__main__':
    img_query()
