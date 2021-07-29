from bs4 import BeautifulSoup
import requests
import pandas as pd

page_range = 2

# try:
#     this
# else:
#     exit
url = f'https://www.carlist.my/cars-for-sale/malaysia?inspected=true&page_number={page_range}&page_size=25'
# # write a loop to scrape from page 1 to the last page
# first_page = 1
# last_page = 2

for page in range(page_range):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    car = soup.find('article', class_ = 'listing')
    # print(car)
    print(car.get('data-title'))
#     epl_table = soup.find('table', class_ = 'standing-table__table')
    # epl_table = soup.find('table', {'class':'standing-table__table'})
    # print(url_page)
    # length = len(link)
