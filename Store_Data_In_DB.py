import sqlite3
import pandas as pd

def connect_db(db_path='data.db'):   
    return sqlite3.connect(db_path)

def store_data(data, conn):   
    data.to_sql('reviews', conn, if_exists='replace', index=False)
    print("Data stored successfully.")
