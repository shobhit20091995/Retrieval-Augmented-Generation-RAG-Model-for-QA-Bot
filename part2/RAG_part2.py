import streamlit as st
import pandas as pd
import openai
import camelot
import os

# Pinecone imports
from pinecone import Pinecone, ServerlessSpec

###############################################################################
# 1. Initialize OpenAI and Pinecone
###############################################################################

"""
kindly paste your own api keys

"""

openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the Pinecone API key from an environment variable
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone with the API key
pc = Pinecone(api_key=pinecone_api_key)

###############################################################################
# 2. Helper Functions (from your existing pipeline)
###############################################################################

def extract_tables_with_camelot(pdf_bytes, start_page, end_page):
    """
    Extract tables from the specified page range of an in-memory PDF (bytes).
    Uses Camelot's stream flavor.
    """
    import tempfile

    # Write PDF bytes to a temp file so Camelot can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    all_tables = []
    for page_num in range(start_page, end_page + 1):
        tables = camelot.read_pdf(tmp_path, pages=str(page_num), flavor='stream')
        all_tables.extend(tables)
    os.remove(tmp_path)  # Clean up the temp file
    return all_tables

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

def preprocess_table_from_camelot(table):
    """
    Preprocess a single table object from Camelot, ensuring:
    - First row becomes header
    - Duplicate column names are handled
    """
    df = table.df
    # Use first row as header
    df.columns = df.iloc[0]
    df = df[1:]
    # Make columns unique
    df.columns = make_unique_column_names(df.columns.tolist())
    df.reset_index(drop=True, inplace=True)
    return df

def generate_embeddings(dataframe):
    """
    Generate embeddings for each row in the DataFrame
    using OpenAI 'text-embedding-ada-002'
    """
    embeddings = []
    for _, row in dataframe.iterrows():
        # Convert each row into a text string
        text = " ".join([f"{col}: {value}" for col, value in row.items()])
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
        embeddings.append({"text": text, "embedding": response["data"][0]["embedding"]})
    return embeddings

def store_embeddings_in_pinecone(embeddings, index_name="financial-data"):
    """
    Store embeddings in Pinecone. Will create the index if not present.
    """
    # Check if the index exists; create it if not
    existing_indexes = pc.list_indexes().names()  # returns a list of existing index names
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=len(embeddings[0]["embedding"]),
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',       # Could be 'aws', 'gcp', or 'azure'
                region='us-east-1' # Region supported by your account
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

def query_pinecone_and_generate_response(query, index_name="financial-data"):
    """
    Query Pinecone with a user query, retrieve top matches,
    and generate a GPT-3.5-Turbo response using the relevant context.
    """
    # Connect to the index
    index = pc.Index(index_name)

    # Generate a query embedding
    query_response = openai.Embedding.create(input=query, model="text-embedding-ada-002")
    query_embedding = query_response["data"][0]["embedding"]

    # Search in Pinecone
    results = index.query(vector=query_embedding, top_k=3, include_metadata=True)

    # Extract relevant texts
    relevant_texts = [match["metadata"]["text"] for match in results["matches"]]
    context = " ".join(relevant_texts)

    # Prepare the messages for ChatCompletion
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

    # Generate a response using GPT-3.5-Turbo
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=200
    )

    answer = completion["choices"][0]["message"]["content"].strip()
    return answer, relevant_texts


###############################################################################
# 3. Streamlit Frontend
###############################################################################
def main():
    st.title("Financial Data QA Bot")

    st.write("Upload a PDF with P&L data, then ask your financial queries.")

    # PDF Upload
    uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])
    start_page = st.number_input("Start Page", min_value=1, value=1)
    end_page = st.number_input("End Page", min_value=1, value=1)

    if uploaded_file is not None:
        if st.button("Process PDF"):
            pdf_bytes = uploaded_file.read()
            st.write("Extracting tables... This may take a few moments.")

            try:
                tables = extract_tables_with_camelot(pdf_bytes, start_page, end_page)

                if len(tables) == 0:
                    st.error("No tables were found in the specified pages.")
                    return

                # Combine all tables
                combined_df = pd.DataFrame()
                for i, table in enumerate(tables):
                    st.write(f"Preprocessing Table {i+1}")
                    df = preprocess_table_from_camelot(table)
                    combined_df = pd.concat([combined_df, df], ignore_index=True)

                st.write("Combined Table Preview:")
                st.dataframe(combined_df.head())

                # Generate embeddings
                st.write("Generating embeddings for the table...")
                embeddings = generate_embeddings(combined_df)

                # Store embeddings in Pinecone
                st.write("Storing embeddings in Pinecone...")
                store_embeddings_in_pinecone(embeddings)

                st.success("PDF processed and embeddings stored. You can now ask questions!")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")

    # Query Input
    query = st.text_input("Enter your financial question:")
    if st.button("Get Answer"):
        if not query:
            st.warning("Please enter a query.")
        else:
            # Query Pinecone and generate a response
            with st.spinner("Getting answer..."):
                answer, relevant_texts = query_pinecone_and_generate_response(query)
            st.write("**Answer:** ", answer)
            with st.expander("Relevant Table Data"):
                for idx, snippet in enumerate(relevant_texts):
                    st.write(f"Match {idx+1}:", snippet)

if __name__ == "__main__":
    main()
