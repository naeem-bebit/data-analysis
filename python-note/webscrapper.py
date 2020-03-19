url = 'https://www.lelong.com.my/catalog/all/list?TheKeyword=macbook+pro&D='

# write a loop to scrape from page 1 to the last page

product_name=[]
for page in range(1,19):
url_page = url+str(page)
scrape = requests.get(url_page)
soup = BeautifulSoup(scrape.content, 'lxml')
link = soup.find_all('div',{'class':'item','class':'summary'})
length = len(link)
for i in range(0,length):
name = link[i].a.get('title')
product_name.append(name)

# write to csv
# convert the list to a pandas dataframe

df = pd.DataFrame({'name':product_name})
df
df.to_csv('output.csv', index=False)