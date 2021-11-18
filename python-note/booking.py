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
