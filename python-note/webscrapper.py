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
    soup = BeautifulSoup(scrape.content, 'lxml')
    link = soup.find_all('div',{'class':'item','class':'summary', 'class':'price-section'})
    print(link)
    length = len(link)

product_names=[]
for i in range(0,length):
    product_title = link[i].a.get('discountpercentage')
    product_names.append(product_title)

# write to csv
# convert the list to a pandas dataframe
df = pd.DataFrame({'name':product_names})
df.to_csv('scrapped_data.csv', index=False)
