# Public PostgreSQL https://rnacentral.org/help/public-database 
# Public MySQL https://docs.rfam.org/en/latest/database.html 

import psycopg2

try:
 connection = psycopg2.connect(user="reader",
                               password="NWDMCE5xdipIjRrp",
                               host="hh-pgsql-public.ebi.ac.uk",
                            #    port="5432", Optional
                               database="pfmegrnargs")

 cursor = connection.cursor()

 cursor.execute(f"SELECT * FROM rnc_database limit 10;")
 record = cursor.fetchall()
 print(record, "\n")

except (Exception, psycopg2.Error) as error:
 print("Error while connecting to PostgreSQL", error)
finally:
 try:
     if connection:
         cursor.close()
         connection.close()
         print("PostgreSQL connection is closed")
 except:
     print("Connection timeout")
