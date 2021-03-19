from bs4 import BeautifulSoup
import requests
import pandas as pd

first_page = 1
last_page = 2
origin = 'KUL'
destination = 'KCH'
depart_date = '2021-05-01'
return_date = '2021-05-03'
url = f'https://www.airasia.com/select/en/gb/{origin}/{destination}/{depart_date}/{return_date}/1/0/0/R/N/MYR/ST'
print(url)

scrape = requests.get(url)
soup = BeautifulSoup(scrape.content, 'lxml')
link = soup.find_all('div',{'class':'ng-star-inserted'})
print(link)
# price = link[i].a.b.get('data-price')
# print(price)
# for page in range(first_page,last_page):
#     url_page = url+str(page)
#     scrape = requests.get(url_page)
#     soup = BeautifulSoup(scrape.content, 'lxml')
#     link = soup.find_all('div',{'class':'item','class':'summary'})
#     length = len(link)

# product = pd.DataFrame()
# for i in range(0,length):
#     product = product.append(pd.DataFrame({'prodcut_price': link[i].a.b.get('data-price'), 
#                                         'product_title' : link[i].a.get('title')}, index=[0]), ignore_index=True)
# print(product)
# product.to_csv('scrapped_data.csv', index=False)
