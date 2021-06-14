import sqlite3

db = sqlite3.connect('book.db')

cur = db.cursor()

cur.execute('''()''')

cur.executemany()

cur.execute('SELECT * FROM books')

print(cur.fetchall)

db.commit()
db.close()
