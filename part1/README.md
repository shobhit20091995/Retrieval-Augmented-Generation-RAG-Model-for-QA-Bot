# Retrieval-Augmented Generation (RAG) Model for QA Bot on P&L Data

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) model designed to create a QA bot for answering queries based on P&L (Profit and Loss) data extracted from PDF documents.

## Note 
Due to the limitations of the free account in Pinecone, storing embeddings for multiple pages of a PDF can result in increased processing time and potential memory issues. To optimize performance, we have currently limited the process to a single page of the PDF for generating results. However, you are free to use as many pages as needed, keeping in mind that it may require additional processing time.

## Kindly ensure you use your own API keys for both OpenAI and Pinecone.

## Model Architecture

### 1. Data Extraction
- **Tool Used**: Camelot
- **Process**:
  - Extract P&L tables from PDF documents using Camelot’s `stream` flavor.
  - Specify pages of interest for precise parsing.
  - Combine extracted tables into a unified DataFrame.

### 2. Preprocessing
- **Column Naming**: Ensure duplicate column names are unique using a helper function.
- **Header Adjustment**: Set the first row of the table as the header and remove redundant rows.
- **Index Reset**: Clean up the DataFrame for further processing.

### 3. Embedding Generation
- **Tool Used**: OpenAI’s `text-embedding-ada-002` model.
- **Process**:
  - Convert each row of the DataFrame into a structured text string (key-value pairs).
  - Use OpenAI's API to generate embeddings for these text strings.

### 4. Embedding Storage
- **Tool Used**: Pinecone
- **Process**:
  - Store embeddings in a vector database for efficient retrieval.
  - Use cosine similarity as the metric for indexing.
  - Store each embedding with metadata (original text) for context retrieval.

### 5. Query Processing and Response Generation
- **Embedding Query**:
  - Generate an embedding for the user’s query using OpenAI.
  - Use Pinecone to search for the top-k similar embeddings.
- **Response Generation**:
  - Retrieve text data is passed to OpenAI’s `gpt-3.5-turbo` for generating responses.
  - The model generates a detailed and contextually accurate response based on the retrieved data.

---

## Approach to Data Extraction and Preprocessing

### Data Extraction
- Parse tables using Camelot’s `read_pdf` function.
- Process multiple pages sequentially and combine tables into a unified dataset.

### Preprocessing
- Ensure all column names in the DataFrame are unique.
- Convert the DataFrame into a clean and structured format suitable for embedding generation.
- Handle edge cases such as empty or improperly extracted tables.

---

## Generative Response Workflow

1. **Query Embedding**: Embed the user’s query using OpenAI’s embedding model.
2. **Embedding Retrieval**: Use Pinecone to retrieve the top-k similar embeddings based on cosine similarity.
3. **Context Creation**: Convert retrieved embeddings back into a context string.
4. **Response Generation**: Use OpenAI’s ChatCompletion API with the query and context to generate a coherent response.

---

## Challenges and Solutions

1. **Inconsistent Table Extraction**:
   - **Solution**: Applied preprocessing techniques to handle misaligned or duplicate headers.

2. **Embedding Size Management**:
   - **Solution**: Limited the number of rows processed in a single batch to comply with API limitations.

3. **Ensuring Query Relevance**:
   - **Solution**: Used Pinecone’s ranking to focus on the top-k matches, ensuring relevance.

4. **Handling Empty Tables**:
   - **Solution**: Incorporated validation checks to skip pages without extractable tables.

---

## Tools and Technologies Used
- **Data Extraction**: Camelot
- **Embedding Generation**: OpenAI’s `text-embedding-ada-002` model
- **Embedding Storage and Retrieval**: Pinecone
- **Response Generation**: OpenAI’s `gpt-3.5-turbo`

---

## How to Use
1. Extract P&L tables from PDF documents using Camelot.
2. Preprocess the extracted data for consistency and structure.
3. Generate embeddings for the preprocessed data using OpenAI’s API.
4. Store embeddings in Pinecone for efficient retrieval.
5. Process user queries and generate responses using the outlined workflow.

---

Feel free to fork, modify, and extend this repository for your specific needs.
