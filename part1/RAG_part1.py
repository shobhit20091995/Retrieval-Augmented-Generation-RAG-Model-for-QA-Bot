# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 20:50:45 2025

@author: HP
"""

import camelot
import pandas as pd
import openai
from pinecone import Pinecone, ServerlessSpec
import os


###############################################################################
# 1. Initialize OpenAI and Pinecone
###############################################################################
"""

Due to the limitations of the free account in Pinecone,
storing embeddings for multiple pages of a PDF can result in increased processing time and potential memory issues.
To optimize performance, we have currently limited the process to a single page of the PDF for generating results.
However, you are free to use as many pages as needed, keeping in mind that it may require additional processing time.


kindly paste your own api keys
"""

openai.api_key = "your api key"  # Replace with your OpenAI API key

# Replace "your_pinecone_api_key" with your Pinecone API key.
pc = Pinecone(
    api_key="your api key",
)

###############################################################################
# 2. Extract P&L Tables from Multiple Pages using Camelot
###############################################################################
def extract_tables_with_camelot(pdf_path, start_page, end_page):
    print(f"Extracting tables from pages {start_page} to {end_page} using Camelot...")
    all_tables = []
    for page_num in range(start_page, end_page + 1):
        print(f"Processing page {page_num}...")
        tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor='stream')
        if not tables:
            print(f"No tables found on page {page_num}. Skipping.")
        else:
            all_tables.extend(tables)  # Collect all tables from all pages
    if not all_tables:
        raise ValueError("No tables found in the specified page range.")
    return all_tables

###############################################################################
# 3. Preprocess the Table with Unique Column Names
###############################################################################
def make_unique_column_names(columns):
    """Return a list of column names ensuring uniqueness by appending a counter."""
    seen = {}
    unique_columns = []
    for col in columns:
        if col not in seen:
            seen[col] = 0
            unique_columns.append(col)
        else:
            seen[col] += 1
            new_col_name = f"{col}.{seen[col]}"
            unique_columns.append(new_col_name)
    return unique_columns

def preprocess_table_from_camelot(tables):
    print("Preprocessing extracted tables...")
    table = tables[0].df  # Use the first table extracted
    
    # First row becomes the header
    table.columns = table.iloc[0]  
    table = table[1:]  # Remove the old header row from data
    
    # Convert column index to a list and ensure they are unique
    new_col_names = make_unique_column_names(table.columns.tolist())
    table.columns = new_col_names

    # Reset the index
    table.reset_index(drop=True, inplace=True)
    return table

###############################################################################
# 4. Generate Embeddings
###############################################################################
def generate_embeddings(dataframe):
    print("Generating embeddings for the table...")
    embeddings = []
    for _, row in dataframe.iterrows():
        # Convert each row into a text string
        text = " ".join([f"{col}: {value}" for col, value in row.items()])
        # Use the OpenAI Embeddings API
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        embeddings.append({"text": text, "embedding": response["data"][0]["embedding"]})
    return embeddings

###############################################################################
# 5. Store Embeddings in Pinecone
###############################################################################
def store_embeddings_in_pinecone(embeddings, index_name="financial-data"):
    print("Storing embeddings in Pinecone...")

    # Check if the index exists; create it if not
    existing_indexes = pc.list_indexes().names()  # returns a list of existing index names
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=len(embeddings[0]["embedding"]),
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',       # Could be 'aws', 'gcp', or 'azure'
                region='us-east-1' # Replace with a region supported by your account
            )
        )

    # Connect to the index
    index = pc.Index(index_name)

    # Prepare vectors for upsert
    vectors_to_upsert = []
    for i, record in enumerate(embeddings):
        vectors_to_upsert.append(
            (
                str(i),               # unique ID for this vector
                record["embedding"],  # the vector itself
                {"text": record["text"]}  # metadata
            )
        )
    # Upsert the embeddings into the Pinecone index
    index.upsert(vectors_to_upsert)

###############################################################################
# 6. Query Pinecone and Generate Response using GPT-3.5-Turbo
###############################################################################
def query_pinecone_and_generate_response(query, index_name="financial-data"):
    print(f"Querying Pinecone for: {query}")

    # 1. Connect to the index
    index = pc.Index(index_name)

    # 2. Generate a query embedding
    query_response = openai.Embedding.create(input=query, model="text-embedding-ada-002")
    query_embedding = query_response["data"][0]["embedding"]

    # 3. Search in Pinecone
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    # 4. Extract relevant texts
    relevant_texts = [match["metadata"]["text"] for match in results["matches"]]
    context = " ".join(relevant_texts)

    # 5. Prepare the messages for ChatCompletion
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions based on provided financial data."
        },
        {
            "role": "user",
            "content": (
                f"Answer the following question based on this data:\n{context}"
                f"\n\nQuestion: {query}\nAnswer:"
            )
        }
    ]

    # 6. Make a ChatCompletion call with GPT-3.5-Turbo
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=200
    )

    # 7. Return the assistant's response
    return completion["choices"][0]["message"]["content"].strip()

###############################################################################
# 7. Main Pipeline
###############################################################################
def main_pipeline(pdf_path, start_page, end_page, query):
    # Step 1: Extract tables using Camelot for the specified page range
    tables = extract_tables_with_camelot(pdf_path, start_page, end_page)
    
    # Step 2: Preprocess the tables
    combined_df = pd.DataFrame()
    for i, table in enumerate(tables):
        print(f"Preprocessing table {i + 1}...")
        df = preprocess_table_from_camelot([table])  # Preprocess each table
        combined_df = pd.concat([combined_df, df], ignore_index=True)  # Combine all tables into one DataFrame

    print("--- Combined Preprocessed Table ---")
    print(combined_df)

    # Step 3: Generate embeddings
    embeddings = generate_embeddings(combined_df)

    # Step 4: Store embeddings in Pinecone
    store_embeddings_in_pinecone(embeddings)

    # Step 5: Query Pinecone and generate a response
    response = query_pinecone_and_generate_response(query)
    print(f"Response: {response}")

###############################################################################
# 8. Execute the Pipeline (Example Usage)
###############################################################################
if __name__ == "__main__":
    pdf_path = "Samplepd.pdf"  # Replace with your local PDF file path
    start_page = 2             # The starting page
    end_page =  2              # The ending page
    query = "What is the difference in total non-current assets between March 31, 2023, and March 31, 2024?"  # Example query

    main_pipeline(pdf_path, start_page, end_page, query)
