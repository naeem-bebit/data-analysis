from bs4 import BeautifulSoup
import requests
import pandas as pd
url = 'https://www.lelong.com.my/catalog/all/list?TheKeyword=macbook+pro&D='

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

# scrape = requests.get('https://www.lelong.com.my/catalog/all/list?TheKeyword=macbook+pro&D=1')
# soup = BeautifulSoup(scrape.content, 'lxml')
# # link = soup.find_all('div',{'class':'item','class':'summary', 'class':'price-section'})
# link = soup.find_all('div',{'class':'item','class':'summary'})

# print(link)
# length = len(link)
# print("LENGTH: ", length)
# product_title = link[1].a.b.get('data-price')
# # product_title = link[1].a.get('discountpercentage')
# print(product_title)

# append the title to the price
# data = []
product = pd.DataFrame()
for i in range(0,length):
    # product_price = link[i].a.b.get('data-price')
    # product_title = link[i].a.get('title')
    # print(product_title)
    # product.append(product_price)  'title': product_title
    product = product.append(pd.DataFrame({'prodcut_price': link[i].a.b.get('data-price'), 
                                        'product_title' : link[i].a.get('title')}, index=[0]), ignore_index=True)
    # product = product.append(pd.DataFrame({'data_price': product_price}, index=[0]))
    # product.append(pd.DataFrame({'data_price': product_title}, index=[0]))
    # data.append(product_price)
# print(data)
print(product)
# convert the list to a pandas dataframe
# df = pd.DataFrame({'product title':data})
# product.to_csv('scrapped_data.csv', index=False)
