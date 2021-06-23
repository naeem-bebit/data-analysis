import sqlite3
import pandas as pd

db = sqlite3.connect('books.db')
df = pd.read_sql_query('SELECT * FROM books;',db)

new_data = {'id':'12','author': 'Naeem'}
df = df.append(new_data, ignore_index=True)

df.to_sql('books', db, if_exists = 'replace', index = False)
cur = db.cursor()

cur.execute('''()''')

cur.executemany()

cur.execute('SELECT * FROM books')

for new_line in cur.fetchall():
    print(new_line)

print(cur.fetchall)

db.commit()
db.close()
