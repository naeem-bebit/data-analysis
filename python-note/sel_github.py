from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import os

options = Options()
# options.add_argument("--headless")
options.add_argument("--window-size=1920, 1080")

driver = webdriver.Chrome(options=options)

params = {'behavior': 'allow', 'downloadPath': os.getcwd() }
driver.execute_cdp_cmd('Page.setDownloadBehavior', params)

driver.get('https://github.com/naeem-bebit')

driver.find_element_by_xpath('//span[@title="streamlit-data"]').click()

driver.find_element_by_xpath('//summary[@class="btn btn-primary"]').click()

driver.find_element_by_xpath('//a[@data-open-app="link"]').click()

driver.close()
