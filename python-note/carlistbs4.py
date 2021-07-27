from bs4 import BeautifulSoup
import requests
import pandas as pd
# url = 'https://www.carlist.my/cars-for-sale/malaysia?inspected=true'

# # write a loop to scrape from page 1 to the last page
# first_page = 1
# last_page = 2

# for page in range(first_page,last_page):
#     url_page = url+str(page)
#     scrape = requests.get(url_page)
#     soup = BeautifulSoup(scrape.content, 'lxml')
#     link = soup.find_all('div',{'class':'item','class':'summary'})
#     # print(link)
#     length = len(link)

url = 'https://www.skysports.com/premier-league-table'

r = requests.get(url)
print(r.status_code)
soup = BeautifulSoup(r.text, 'html.parser')

# epl_table = soup.find('table', class_ = 'standing-table__table')
epl_table = soup.find('table', {'class':'standing-table__table'})

print(epl_table)

