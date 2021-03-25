from bs4 import BeautifulSoup
import requests
import pandas as pd
import string

url = 'https://en.wikipedia.org/wiki/List_of_airports_by_IATA_airport_code:_Z'
print(url)
# scrape = requests.get(url)
# soup = BeautifulSoup(scrape.content, 'lxml')
# link = soup.find_all('div',{'class':'price'})
# print(link)

# scrape = requests.get('https://www.lelong.com.my/catalog/all/list?TheKeyword=macbook+pro&D=1')
# soup = BeautifulSoup(scrape.content, 'lxml')
# # link = soup.find_all('div',{'class':'item','class':'summary', 'class':'price-section'})
# link = soup.find_all('div',{'class':'item','class':'summary'})

for i in string.ascii_uppercase:
    url = f'https://en.wikipedia.org/wiki/List_of_airports_by_IATA_airport_code:_{i}'
    # print(url)

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
