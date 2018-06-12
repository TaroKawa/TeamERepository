from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from pyvirtualdisplay import Display
import os
import time
import traceback
from bs4 import BeautifulSoup
from urllib.request import urlretrieve


def query_on_google_img(query=""):
    disp = Display(visible=0, size=(800, 600))
    disp.start()

    print("=======================================")
    print("Start query.")
    if os.name == 'nt':
        driver_path = "../windows/chromedriver.exe"
    elif os.name == 'posix':
        driver_path = "../linux/chromedriver"
    elif os.name == "darwin":
        driver_path = "../mac/chromedriver"

    driver = webdriver.Chrome(executable_path=driver_path)
    driver.implicitly_wait(2)
    driver.get("https://www.google.co.jp/imghp?hl=ja")

    os.mkdir(f"../../data/{query.split()[1]}")
    file_path = f"../../data/{query.split()[1]}/"

    try:
        input_query = driver.find_element_by_xpath('//*[@id="lst-ib"]')
        input_query.send_keys(query)
        input_query.send_keys(Keys.ENTER)

        src = driver.page_source
        time.sleep(20)
        soup = BeautifulSoup(src, "html.parser")
        jscontroller = soup.find_all("div", attrs={"jscontroller": "Q7Rsec"})
        print(f"Found {len(jscontroller)} pictures.")
        print("===================================")
        for i, js in enumerate(jscontroller):
            img = js.find_all("img")[0]
            src = img.get("src")
            try:
                urlretrieve(src, f"{file_path}{i}.jpg")
                time.sleep(10)
                print(f"Successfully saved {i}.png")
            except Exception as e:
                print(f"Failed to retreive: {src}")
                traceback.format_exc()
    except Exception:
        print(traceback.format_exc())
        driver.quit()
        disp.stop()
        exit(0)

    driver.quit()
    disp.stop()


if __name__ == "__main__":
    colors = ["赤", "青", "緑", "黄色", "紫", "黒", "白", "オレンジ"]
    for c in colors:
        query_on_google_img(f"ワンピース {c}")
