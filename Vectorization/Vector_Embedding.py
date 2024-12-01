from annoy import AnnoyIndex
import spacy
import sqlite3

nlp = spacy.load('en_core_web_md')
print("Loaded spaCy model 'en_core_web_md'.")

embedding_size = nlp.vocab.vectors_length
t = AnnoyIndex(embedding_size, 'angular')
print("Initialized Annoy index with embedding size:", embedding_size)

def fetch_data(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT rowid, Review_Text FROM reviews")
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    print("Fetched data from 'reviews' table, total records:", len(data))
    return data

def process_in_batches(texts, batch_size):
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        for rowid, text in batch:
            doc = nlp(text)
            t.add_item(rowid - 1, doc.vector) 
        print(f"Processed batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}")
        t.build(10)  
        t.save('embeddings.ann')
        if input("Continue with next batch? (Y/N): ").strip().upper() != 'Y':
            break

def main():
    db_path = 'data.db'
    texts = fetch_data(db_path)
    batch_size = int(input("Enter batch size: "))
    process_in_batches(texts, batch_size)

main()
