from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import time



start = time.time()

# Run the scrapping with headless
# google "my user agent"
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
# ua = UserAgent()
# user_agent = ua.random
options = webdriver.ChromeOptions()
# options.headless = True
options.add_argument(f'user-agent={user_agent}')
options.add_argument("--window-size=1920,1080")
options.add_argument('--ignore-certificate-errors')
options.add_argument('--allow-running-insecure-content')
options.add_argument("--disable-extensions")
options.add_argument("--proxy-server='direct://'")
options.add_argument("--proxy-bypass-list=*")
options.add_argument("--start-maximized")
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--no-sandbox')
options.add_argument('disable-infobars')
options.add_argument('--hide-scrollbars')
options.add_argument('--enable-logging')

PATH = "/usr/local/bin/chromedriver"
driver = webdriver.Chrome(PATH, options=options)
wait = WebDriverWait(driver, 10)
# driver = webdriver.Chrome()

driver.get("https://www.airasia.com/en/gb")