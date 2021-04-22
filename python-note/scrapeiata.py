from bs4 import BeautifulSoup
import requests
import pandas as pd

url = 'https://en.wikipedia.org/wiki/List_of_airports_by_IATA_airport_code:_A'
print(url)
s = requests.Session()
response = s.get(url, timeout=10)
soup = BeautifulSoup(response.content, 'html.parser')
pretty_soup = soup.prettify()
# print(pretty_soup)
print(soup.title.string)
# all_tables=soup.find_all('table')
# print(all_tables)
wiki_table=soup.find('table', {"class":'wikitable'})
# print(wiki_table)
iata_rowA = [] 
for row in wiki_table.findAll("tr")[1:]:
#     print("THIS IS ROW:",len(row))
#     print(row.text.rstrip()) #focus on rstrip
    if len(row) == 12:
        iata_rowA.append(row.text.rstrip())
    else:
        continue

df_iataA = pd.DataFrame(iata_rowA,columns = ['a'])
# iata_rowA
df_iataA = df_iataA['a'].str.split('\n',expand=True)
print(df_iataA.head())

#----

from bs4 import BeautifulSoup
import requests
import pandas as pd

url = 'https://www.tripadvisor.in/Hotels-g187147-Paris_Ile_de_France-Hotels.html'
print(url)
s = requests.Session()
response = s.get(url, timeout=10)
soup = BeautifulSoup(response.content, 'html.parser')

ratings = []
for rating in soup.find_all('a',{'class':'ui_bubble_rating'}):
    ratings.append(rating['alt'])

hotel = []
for name in soup.findAll('div',{'class':'listing_title'}):
    hotel.append(name.text.strip())

price = []

for p in soup.findAll('div',{'class':'price-wrap'}):
    price.append(p.text) 
price[:5]

pd.DataFrame(hotel)
