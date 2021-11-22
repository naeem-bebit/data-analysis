from bs4 import BeautifulSoup
import requests
import pandas as pd
url = 'https://www.booking.com/searchresults.en-gb.html?label=gen173nr-1FCAEoggI46AdIM1gEaKEBiAEBmAEJuAEXyAEM2AEB6AEB-AELiAIBqAIDuAK5sNOMBsACAdICJDA2NzVkZDA1LWZhYjktNDkyMS1hMWJmLTNkZjUxMTU4OTcwZdgCBuACAQ&sid=6902d1ae387cfec2ec3c1adf612ff996&aid=304142&sb_lp=1&error_url=https%3A%2F%2Fwww.booking.com%2Findex.en-gb.html%3Flabel%3Dgen173nr-1FCAEoggI46AdIM1gEaKEBiAEBmAEJuAEXyAEM2AEB6AEB-AELiAIBqAIDuAK5sNOMBsACAdICJDA2NzVkZDA1LWZhYjktNDkyMS1hMWJmLTNkZjUxMTU4OTcwZdgCBuACAQ%3Bsid%3D6902d1ae387cfec2ec3c1adf612ff996%3Bsb_price_type%3Dtotal%3Bsig%3Dv1gy-6SYTt%26%3B&ss=Bangkok%2C+Bangkok+Province%2C+Thailand&is_ski_area=0&ssne=Rio+de+Janeiro&ssne_untouched=Rio+de+Janeiro&checkin_year=&checkin_month=&checkout_year=&checkout_month=&group_adults=2&group_children=0&no_rooms=1&b_h4u_keep_filters=&from_sf=1&ss_raw=bangkok&ac_position=0&ac_langcode=en&ac_click_type=b&dest_id=-3414440&dest_type=city&iata=BKK&place_id_lat=13.755838&place_id_lon=100.505638&search_pageview_id=6594491d8812023b&search_selected=true&nflt=ht_id%3D204%3Bclass%3D4&shw_aparth=0&order=bayesian_review_score'

# write a loop to scrape from page 1 to the last page
first_page = 1
last_page = 2

for page in range(first_page,last_page):
    url_page = url+str(page)
    scrape = requests.get(url_page)
    soup = BeautifulSoup(scrape.text, 'html.parser')
    # soup = BeautifulSoup(scrape.content, 'lxml') #both can be utilized for scrapping
    link = soup.find_all('div',{'class':'item','class':'summary'})
    # print(link)
    length = len(link)

print(length)

import atexit

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait

URL = ("https://www.booking.com/searchresults.en-gb.html?"
       "label=gen173nr-1FCAEoggI46AdIM1gEaK4BiAEBmAEJuAEXyAEM2AEB6AEB-AELiAIBqAIDuALnhOzyBcACAQ"
       "&lang=en-gb&sid=422b3ff3c0e98b522259ad1cad2505ea&sb=1&src=searchresults&src_elem=sb"
       "&error_url=https%3A%2F%2Fwww.booking.com%2Fsearchresults.en-gb.html%3Flabel%3Dgen173nr-"
       "1FCAEoggI46AdIM1gEaK4BiAEBmAEJuAEXyAEM2AEB6AEB-AELiAIBqAIDuALnhOzyBcACAQ%3Bsid%3D422b3ff"
       "3c0e98b522259ad1cad2505ea%3Btmpl%3Dsearchresults%3Bclass_interval%3D1%3Bdest_id%3D-150690"
       "9%3Bdest_type%3Dcity%3Bdtdisc%3D0%3Bfrom_sf%3D1%3Bgroup_adults%3D2%3Bgroup_children%3D0%3"
       "Binac%3D0%3Bindex_postcard%3D0%3Blabel_click%3Dundef%3Bno_rooms%3D1%3Boffset%3D0%3Bpostcar"
       "d%3D0%3Braw_dest_type%3Dcity%3Broom1%3DA%252CA%3Bsb_price_type%3Dtotal%3Bshw_aparth%3D1%3Bs"
       "lp_r_match%3D0%3Bsrc%3Dindex%3Bsrc_elem%3Dsb%3Bsrpvid%3D912403b6d1220012%3Bss%3DAuckland%3B"
       "ss_all%3D0%3Bssb%3Dempty%3Bsshis%3D0%3Bssne%3DAuckland%3Bssne_untouched%3DAuckland%3Btop_ufi"
       "s%3D1%26%3B&sr_autoscroll=1&ss=Auckland&is_ski_area=0&ssne=Auckland&ssne_untouched=Auckland&ci"
       "ty=-1506909&checkin_year=2020&checkin_month=9&checkin_monthday=1&checkout_year=2020&checkout_m"
       "onth=9&checkout_monthday=2&group_adults=2&group_children=0&no_rooms=1&from_sf=1'")

print(URL)
class page_loaded:
    def __call__(self, driver):
        document_ready = driver.execute_script("return document.readyState;") == "complete"
        jquery_ready = driver.execute_script("return jQuery.active == 0;")
        print(f"document ready: [({type(document_ready).__name__}){document_ready}]")
        print(f"jquery  ready: [({type(jquery_ready).__name__}){jquery_ready}]")
        return document_ready and jquery_ready


def wait_for_page_to_load(driver, timeout_seconds=20):
    WebDriverWait(driver, timeout_seconds, 0.2).until(page_loaded(), f"Page could not load in {timeout_seconds} s.!")


def go_to_url(driver, url):
    driver.get(url)
    wait_for_page_to_load(driver)


def get_orange_prices(soup):
    return [price_label.get_text(strip=True)
            for price_label
            in soup.select("label.tpi_price_label.tpi_price_label__orange")]


def get_normal_prices(soup):
    return [price_label.get_text(strip=True)
            for price_label
            in soup.select("div[class*=fde444d7ef _c445487e2]")]


def start_driver():
    driver = webdriver.Chrome(executable_path="/usr/local/bin/chromedriver")
    atexit.register(driver.quit)
    driver.maximize_window()
    return driver


def main():
    driver = start_driver()
    go_to_url(driver, URL)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    orange_prices = get_orange_prices(soup)
    print(orange_prices)
    normal_prices = get_normal_prices(soup)
    print(normal_prices)


if __name__ == '__main__':
    main()
