# Public PostgreSQL https://rnacentral.org/help/public-database 
# Public MySQL https://docs.rfam.org/en/latest/database.html 

import psycopg2

table_name = 'rnc_database'

try:
    connection = psycopg2.connect(user="reader",
                                  password="NWDMCE5xdipIjRrp",
                                  host="hh-pgsql-public.ebi.ac.uk",
#                                   port="5432", #Optional
                                  database="pfmegrnargs")

    cursor = connection.cursor()
    sqlquery = f"SELECT * FROM {table_name} limit 10;"
    cursor.execute(sqlquery)
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


import psycopg2
import pandas as pd

try:
    connection = psycopg2.connect(user="reader",
                                  password="NWDMCE5xdipIjRrp",
                                  host="hh-pgsql-public.ebi.ac.uk",
#                                   port="5432", #Optional
                                  database="pfmegrnargs")
    
    sqlquery = 'SELECT * FROM rnc_database;'
    df_sql = pd.read_sql(sqlquery, connection)

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

df_sql.shape
