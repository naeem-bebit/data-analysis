from bs4 import BeautifulSoup
import requests
import pandas as pd

url = 'https://en.wikipedia.org/wiki/List_of_airports_by_IATA_airport_code:_A'
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
iata_rowA = [] 
for row in wiki_table.findAll("tr")[1:]:
#     print("THIS IS ROW:",len(row))
#     print(row.text.rstrip()) #focus on rstrip
    if len(row) == 12:
        iata_rowA.append(row.text.rstrip())
    elif len(row) == 2: 
        continue

df_iataA = pd.DataFrame(iata_rowA,columns = ['a'])
# iata_rowA
df_iataA = df_iataA['a'].str.split('\n',expand=True)
print(df_iataA.head()
