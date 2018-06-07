from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from pyvirtualdisplay import Display
import os
import time
import traceback
from urllib.request import urlretrieve


def query_on_google_img(query=""):
    disp = Display(visible=1, size=(800, 600))
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

        imgs = driver.find_elements_by_tag_name("img")
        print(f"Found {len(imgs)} pictures.")
        print("===================================")
        for i, img_tag in enumerate(imgs):
            src = img_tag.get_attribute("src")
            try:
                urlretrieve(src, f"{file_path}{i}.jpeg")
                time.sleep(5)
            except Exception as e:
                traceback.format_exc()
    except Exception:
        print(traceback.format_exc())
        driver.quit()
        disp.stop()
        exit(0)

    driver.quit()
    disp.stop()


if __name__ == "__main__":
    query_on_google_img("ワンピース 赤")
