# RAG Chatbot User Manual

## 1. Introduction
This document serves as a user manual for the **Retrieval Augmented Generation (RAG) Chatbot** system. The chatbot answers questions using information from provided documents (PDF and Excel). In future versions, it may include internet search capabilities. This version demonstrates core RAG principles like document ingestion, chunking, embedding, and retrieval.

## 2. System Architecture Overview
The RAG chatbot combines a **Language Model (LLM)** with relevant information retrieved from a knowledge base to provide more accurate, context-specific answers.

### Core Components:
- **Document Loader:** Ingests and extracts content from PDF and Excel files.
- **Text Splitter:** Breaks documents into smaller chunks for efficient retrieval.
- **Embedding Model:** Converts text into vector representations.
- **Vector Store:** Stores embeddings for fast similarity searches.
- **Retrieval-Augmented QA Chain:** Combines document chunks with user queries to generate coherent answers.
- **Language Model (LLM):** Generates responses based on the retrieved context.

## 3. Technology Stack
The chatbot uses the following technologies:
- **LangChain:** Framework for language model applications.
- **PyMuPDF (fitz):** For extracting text from PDFs.
- **pandas:** For handling Excel files.
- **ChromaDB:** Stores and retrieves document embeddings.
- **HuggingFace Transformers:** Provides pre-trained models for NLP tasks.
- **python-dotenv:** Manages environment variables (e.g., for future API key integration).

## 4. Setup and Installation

### 4.1. Prerequisites
Ensure you have:
- **Python 3.8+**
- **pip** (Python package installer)

### 4.2. Install Dependencies
1. Clone the repository or create your project directory.
2. Navigate to the project directory.
3. Create a `requirements.txt` file with the following:

    ```
    langchain
    langchain-community
    pymupdf
    pandas
    openpyxl
    chromadb
    python-dotenv
    sentence-transformers
    transformers
    torch
    ```

4. Install the dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

    *Note: If `torch` causes issues, try installing the CPU version:*

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```

### 4.3. Environment Variables for API keys
Set up a `.env` file for LLM GPT4 model integration:
OPENAI_API_KEY='YOUR_OPENAI_API_KEY'
SERPER_API_KEY='YOUR_SERPER_API_KEY'

## 5. Running the Chatbot

### 5.1. Ensure your `OPENAI_API_KEY` environment variable is set.
 Obtain an API key from OpenAI and add it to `.env`.

### 5.2. How to Train the Model with Your Documents
In RAG systems, documents are "ingested" into the system rather than training the LLM in a traditional sense.

To ingest documents:
1. Place your PDF and Excel files in the project directory.
2. Modify `ingest_documents` in **`chatbot.py`** to point to your files:

    ```python
    my_documents = [
        "./my_data/sample.pdf",
        "./my_data/sample.xlsx"
    ]
    chatbot.ingest_documents(my_documents)
    ```

3. Run the script:

    ```bash
    python3 chatbot.py
    ```

The chatbot will process the documents and store the embeddings in `chroma_db`.

### Supported Document Formats:
- **PDF (.pdf)**: Processed by `PyMuPDF`.
- **Excel (.xlsx, .xls)**: Processed by `pandas`.

To support other formats (like `.docx`, `.txt`), modify the document loading logic.

## 6. Future Enhancements

### 6.1. Integrating Internet Search (e.g., Serper.dev)
To enable the chatbot to search the internet:
1. Obtain an API key from a search provider like Serper.dev.
2. Modify **`chatbot.py`** to include the search tool and setup an agent to use both document retrieval and internet search.

## 7. Conclusion
This RAG chatbot system provides a solid foundation for document-based Q&A. You can extend it by integrating other commercial LLMs or adding internet search capabilities for more advanced functionality.


