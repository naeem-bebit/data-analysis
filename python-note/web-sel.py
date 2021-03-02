from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

PATH = "/usr/local/bin/chromedriver"
driver = webdriver.Chrome(PATH)
driver.get("https://www.airasia.com/en/gb")

print(driver.title)

time.sleep(5)
driver.quit()
