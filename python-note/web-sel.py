from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time



start = time.time()

# Run the scrapping with headless
# google "my user agent"
user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_2_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36"

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
# driver = webdriver.Chrome()
# driver.get("https://www.airasia.com/en/gb")
driver.get("https://www.techwithtim.net/")
print(driver.title)
# assert "airasia.com" in driver.title

# try:
#     element = WebDriverWait(driver, 15).until(
#         EC.presence_of_element_located((By.LINK_TEXT, "Flights"))
#     )
#     element.click()
# except:
#     driver.quit()

# content = driver.find_element_by_css_selector('p.Tile__InnerWrapper-sc-36ntfl-3 cQuWCb tile-inner-wrapper')
# print(link)
# link.click()

time.sleep(5)
driver.quit()

end = time.time()
print(end - start)