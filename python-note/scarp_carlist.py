import requests
from bs4 import BeautifulSoup as bs
import json 

url = 'https://www.carlist.my/used-cars-for-sale/malaysia?page_number='
first_page = '1'
last_page = '2'

url_inspected = 'https://www.carlist.my/cars-for-sale/malaysia?inspected=true'
full_url = url+first_page
print(full_url)
# changeType = lambda i: int(i) if i.isdigit() else float(i) if i.replace('.', '').isdigit() else i
# result = [(soup:=bs(requests.get(i).content, 'lxml')) and {'url': i,'name': soup.find(
#     'h1', class_='listing__title').text.strip(),'image': [i['data-src'] for i in soup.find('div', class_='gallery').findAll(
#         'img', class_='gallery__item')],'price': int(soup.find('div', class_='listing__price').text.strip().replace(
#             'RM', '').replace(',', '')),'installment': dict(zip(('price', 'per'), (r:=[i.strip() for i in soup.find(
#                 'div', class_='listing__installment').text.split('/')]) and [int(r[0].replace(
#                     'RM', '').replace(',', '')), r[1]])), 'sellerInfo': dict(zip(('type', 'location'), (
#                         r:=[i.contents for i in soup.find('div', class_='listing__seller__list').findAll(
#                             'div', class_='listing__seller__list__item')]) and [r[0][2].strip(), r[1][1].strip()[:-3].replace(
#                                 '\u00bb', '-')])), 'keyDetails': dict([(r:=[i.text.strip() for i in i.findAll(
#                                     'span', recursive=False)]) and [(s:=r[0].title().replace(
#                                         ' ', '')) and s[0].lower()+s[1:], changeType(r[1])] for i in soup.find(
#                                             'div', class_='listing__key-listing__list').findAll(
#                                                 'div', class_='list-item')]),'sellerComment': soup.find(
#                                                     'p', class_='listing__body').text.strip(),**dict(zip(
#                                                         ('specifications', 'equipment'), [(r:=i.findAll((
#                                                             'h3', 'div'), recursive=False)) and (s:=[r[i:i+2] for i in range(0, len(r), 2)]) and dict(
#                                                                 [(s:=i[0].text.strip().replace(' ', '')) and s[0].lower()+s[1:], dict((
#                                                                     r:=[i.text for i in i.findAll('span')]) and [(s:=r[0].title().replace(' ', '')) and s[0].lower()+s[1:], changeType(
#                                                                         r[1])] for i in i[1].findAll('div', class_='list-item'))] for i in s) for i in soup.find('div',class_='specifications').findAll('div', class_='cycle-slide')]))} for i in [i.find(
#                                                                             'a')['href'] for i in bs(requests.get('https://www.carlist.my/new-cars-for-sale/malaysia?page_size=50').content, 'lxml').findAll('h2', class_='listing__title')]]
# print(json.dump(result, open('result.json', 'w'), indent=4))
