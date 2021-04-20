from bs4 import BeautifulSoup
import requests
import pandas as pd

# url = 'https://en.wikipedia.org/wiki/List_of_airports_by_IATA_airport_code:_Z'
# print(url)
# scrape = requests.get(url)
# soup = BeautifulSoup(scrape.content, 'lxml')
# link = soup.find_all('div',{'class':'wikitable sortable jquery-tablesorter'})
# print(link)
# product_title = link[0].a.get('title')
# print(product_title)

# scrape = requests.get('https://www.lelong.com.my/catalog/all/list?TheKeyword=macbook+pro&D=1')
# soup = BeautifulSoup(scrape.content, 'lxml')
# # link = soup.find_all('div',{'class':'item','class':'summary', 'class':'price-section'})
# link = soup.find_all('div',{'class':'item','class':'summary'})

# for i in string.ascii_uppercase:
#     url = f'https://en.wikipedia.org/wiki/List_of_airports_by_IATA_airport_code:_{i}'
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

url = "https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population"
print(url)
s = requests.Session()
response = s.get(url, timeout=10)
soup = BeautifulSoup(response.content, 'html.parser')
pretty_soup = soup.prettify()
print(soup.title.string)
# all_tables=soup.find_all('table')
# print(all_tables)
wiki_table=soup.find('table', {"class":'wikitable'})
# print(wiki_table)

for row in wiki_table.findAll("tr"):
    cells = row.findAll('th')
    # print(cells)

# print(row[0])
print("COLUMNS: ", len(cells))

rows = wiki_table.findAll("tr")
print("ROWS: ", len(rows))
# print(rows[0])

header = [th.text.rstrip() for th in rows[0].find_all('th')][1:]
a0 = []
a1 = []
a2 = []
a3 = []
a4 = []
for d in wiki_table.findAll('tr'):
    # header = d.findAll('th')
    # print(header[0].text.rstrip()) #To extract Rank since it's in the "th"
    cells = d.findAll('td')
    if len(cells) == 5:
        # print(cells[1].text.rstrip()) #similar to find(text=True)
        a0.append(cells[0].find('a').text)   
        a1.append(cells[1].find(text=True))
        a2.append(cells[2].find(text=True))
        a3.append(cells[3].find(text=True))
        a4.append(cells[4].find(text=True))

df = pd.DataFrame(list(zip(a0,a1,a2,a3,a4)),columns = header)
print(df.head())
# df.to_csv('World population.csv')
