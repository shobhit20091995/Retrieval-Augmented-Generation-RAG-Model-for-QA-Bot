# Interactive QA Bot for Financial Data

Welcome to the **QA Bot for Financial Data**! This interactive platform enables users to upload financial documents (such as Profit & Loss statements) in PDF format and ask specific financial questions. The bot processes the uploaded documents, extracts relevant data, and uses advanced AI models to generate accurate, contextually relevant responses.

---

## Note 
Due to the limitations of the free account in Pinecone, storing embeddings for multiple pages of a PDF can result in increased processing time and potential memory issues. To optimize performance, we have currently limited the process to a single page of the PDF for generating results. However, you are free to use as many pages as needed, keeping in mind that it may require additional processing time.

## Kindly ensure you use your own API keys for both OpenAI and Pinecone.

## Features

- **PDF Upload**: Upload financial documents for analysis.
- **Real-Time Querying**: Ask financial questions and get instant answers.
- **Relevant Data Display**: View table segments retrieved from the document alongside the generated answers.
- **User-Friendly Interface**: Built using Streamlit for ease of use.

---

## How to Use the QA Bot

### 1. Uploading Financial Documents
1. Open the QA Bot interface.
2. Click on the **Upload your PDF file** button.
3. Select a PDF document containing P&L tables from your computer.
4. Specify the start and end pages for processing using the input fields below the upload button.
5. Click the **Process PDF** button to extract tables from the document.

**Notes**:
- Ensure the document contains clearly formatted tables for accurate extraction.
- The bot uses Camelot to extract table data, so tables should not be heavily stylized or contain merged cells.

---

### 2. Asking Financial Questions
1. After processing the PDF, a preview of the extracted table will be displayed.
2. Enter your financial question in the text input box labeled **Enter your financial question**.
3. Click the **Get Answer** button to retrieve the answer.

**Example Queries**:
- What are the total expenses for Q2 2023?
- Show the operating margin for the past 6 months.
- What is the total revenue for the year?

**Query Output**:
- The bot will display the generated answer in **bold** under the query input box.
- Relevant table data will be shown in an expandable section labeled **Relevant Table Data**.

---

## Example Queries and Outputs

### Query 1: Total Value of Non-Current Assets (March 31, 2024)
**Query**: What is the total value of non-current assets as of March 31, 2024?  
**Response**:  
- To calculate the total value of non-current assets as of March 31, 2024, sum the total non-current assets and other non-current assets for that date.  
  - **Total Non-Current Assets**:
    - $48,382 (from 1.1)
    - $54,935 (from 1.2)
    - **Sum of Non-Current Assets**: $48,382 + $54,935 = $103,317
  - **Other Non-Current Assets**:
    - $2,121 (from 2.9.1)
    - $2,318 (from 2.9.2)
    - **Sum of Other Non-Current Assets**: $2,121 + $2,318 = $4,439
  - **Total Value of Non-Current Assets**: $103,317 + $4,439 = $107,756

### Query 2: Non-Current Assets (March 31, 2023)
**Query**: What are the non-current assets for March 31, 2023?  
**Response**:  
- The non-current assets for March 31, 2023, are $54,935 million.

### Query 3: Difference in Non-Current Assets (March 31, 2023, vs. March 31, 2024)
**Query**: What is the difference in total non-current assets between March 31, 2023, and March 31, 2024?  
**Response**:  
- **To calculate the difference in total non-current assets**:
  - **Total Non-Current Assets (March 31, 2024)**: $54,935
  - **Total Non-Current Assets (March 31, 2023)**: $48,382
  - **Difference**: $54,935 - $48,382 = $6,553
- **Final Answer**: The difference in total non-current assets between March 31, 2023, and March 31, 2024, is $6,553.

---

## Troubleshooting

### Common Issues
- **No Tables Found**:
  - Ensure the specified page range contains structured tables.
  - Use PDFs with text-based tables rather than scanned images.
- **Inaccurate Data Extraction**:
  - Tables with merged or multi-row headers may cause issues.
  - Verify table formatting before uploading.
- **Query Errors**:
  - Use specific and clear financial terms in queries.
  - Ensure the uploaded document contains data relevant to the question.

---

## Contact Support

For further assistance, contact our support team at **shobhit20091995@gmail.com**.

---

## Technical Notes
- **Backend**: Uses OpenAI’s `GPT-3.5-Turbo` for generating responses and Pinecone for vector database storage.
- **Table Extraction**: Camelot’s `stream` flavor extracts tables from PDFs.
- **Embeddings**: Generated using OpenAI’s `text-embedding-ada-002` model.

---

Thank you for using the QA Bot for Financial Data!
