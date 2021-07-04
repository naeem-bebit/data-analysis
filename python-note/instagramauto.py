from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
import time
import os



# start = time.time()

# # Run the scrapping with headless
# # google "my user agent"
# user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
# # ua = UserAgent()
# # user_agent = ua.random
# options = webdriver.ChromeOptions()
# # options.headless = True
# options.add_argument(f'user-agent={user_agent}')
# options.add_argument("--window-size=1920,1080")
# options.add_argument('--ignore-certificate-errors')
# options.add_argument('--allow-running-insecure-content')
# options.add_argument("--disable-extensions")
# options.add_argument("--proxy-server='direct://'")
# options.add_argument("--proxy-bypass-list=*")
# options.add_argument("--start-maximized")
# options.add_argument('--disable-dev-shm-usage')
# options.add_argument('--no-sandbox')
# options.add_argument('disable-infobars')
# options.add_argument('--hide-scrollbars')
# options.add_argument('--enable-logging')

# PATH = "/usr/local/bin/chromedriver"
# driver = webdriver.Chrome(PATH, options=options)
# wait = WebDriverWait(driver, 10)
# # driver = webdriver.Chrome()

# driver.get("https://www.airasia.com/en/gb")


###

# driver = webdriver.Chrome('/Users/naeem/Downloads/chromedriver')
options = Options()
options.add_argument
options.add_argument("--headless")
options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(options=options)

params = {"behavior":'allow', 'downloadPath':os.getcwd()}
driver.execute_cdp_cmd('Page.setDownloadBehavior', params)

driver.get('https://dashboard.goquo.com/Account/Login')
wait = WebDriverWait(driver, 10)

username_input = '//*[@id="Email"]'
password_input = '//*[@id="Password"]'
driver.find_element_by_xpath(username_input).send_keys(username)
driver.find_element_by_xpath(password_input).send_keys(password)
log_in = '//*[@id="js-page-content"]/div/div/section/form/div[5]/button'
driver.find_element_by_xpath(log_in).click()
site_check = '//*[@id="js-nav-menu"]/li[3]/a'
driver.find_element_by_xpath(site_check).click()
site_click = '//*[@id="js-nav-menu"]/li[3]/ul/li[1]/a/span'
driver.find_element_by_xpath(site_click).click()
filter_date = '//*[@id="js-page-content"]/div[1]/div/button'
driver.find_element_by_xpath(filter_date).click()
time_range = '//*[@id="TimeRange"]'
driver.find_element_by_xpath(time_range).click()
select = Select(driver.find_element_by_id('TimeRange'))
select.select_by_visible_text('6 Months')
select.select_by_value('7')

submit_daterange = '//*[@id="filters-modal"]/div/div/div[2]/form/div[5]/div/div/button'
driver.find_element_by_xpath(submit_daterange).click()

# driver.implicitly_wait(200)

download_data = '//*[@id="topSuppliersGrid"]/div/div[4]/div/div/div[3]/div[1]/div/div'
driver.find_element_by_xpath(download_data).click()

print('ready')
driver.close()
