from annoy import AnnoyIndex
import spacy
import sqlite3
import os
from groq import Groq

# Load the spaCy model
nlp = spacy.load('en_core_web_md')
print("spaCy model loaded.")

client = Groq(
    api_key="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",
)

embedding_size = nlp.vocab.vectors_length
t = AnnoyIndex(embedding_size, 'angular')
if t.load('embeddings.ann'):  
    print("Annoy index loaded from file.")
else:
    print("Failed to load Annoy index. Check the file path and permissions.")

def find_nearest_neighbor_index(query_text):
    query_vector = nlp(query_text).vector
    nearest_index = t.get_nns_by_vector(query_vector, 1, include_distances=False)[0]
    print(f"Nearest neighbor index for the provided query: {nearest_index}")
    return nearest_index

def fetch_and_print_record(db_path, index):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT Review_ID, Rating, Year_Month, Reviewer_Location, Review_Text, Branch FROM reviews WHERE rowid = ?", (index + 1,))
    record = cursor.fetchone()
    cursor.close()
    conn.close()

    if record:
        print(f"Record at index {index}:\n - Review ID: {record[0]}\n - Rating: {record[1]}\n - Year/Month: {record[2]}\n - Reviewer Location: {record[3]}\n - Review Text: {record[4]}\n - Branch: {record[5]}")
    else:
        print(f"No record found at index {index}")

def analyze_sentiment(text):
    sentiment_analysis = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Analyze the sentiment of this text: {text}",
            }
        ],
        model="llama3-8b-8192",
    )
    sentiment = sentiment_analysis.choices[0].message.content
    print(f"Sentiment analysis result: {sentiment}")
    return sentiment

def retrieve_and_generate(query):
    retriever_response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Retrieve relevant data for this query: {query}",
            }
        ],
        model="llama3-8b-8192",
    )
    retrieved_data = retriever_response.choices[0].message.content
    print(f"Retrieved data: {retrieved_data}")

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Use the retrieved data to generate a detailed response.",
            },
            {
                "role": "user",
                "content": retrieved_data,
            }
        ],
        model="llama3-8b-8192",
    )
    generated_response = response.choices[0].message.content
    print(f"Generated response: {generated_response}")
    return generated_response

# Get query from user input
def main():
    query_text = input("Please enter your query text: ")
    nearest_index = find_nearest_neighbor_index(query_text)
    fetch_and_print_record('data.db', nearest_index)

    analyze_first = input("Do you want to generate sentiment analysis first? (Y/N): ").strip().upper()
    if analyze_first == 'Y':
        sentiment = analyze_sentiment(query_text)

    print("\nPerforming Retriever-Augmented Generation...")
    retrieve_and_generate(query_text)

if __name__ == "__main__":
    main()