from Preprocess_Data import load_data, preprocess_data
from Store_Data_In_DB import connect_db, store_data

def main(file_path):
    data = load_data(file_path)
    clean_data = preprocess_data(data)
    
    conn = connect_db()
    store_data(clean_data, conn)
    conn.close()
    print("Pipeline execution completed.")

if __name__ == '__main__':
    import sys
    main(sys.argv[1])
