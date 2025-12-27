
import os
import pandas as pd
import fitz  # PyMuPDF
from dotenv import load_dotenv

from openai  import OpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from transformers import pipeline
from langchain_core.language_models import LLM

# Load environment variables
load_dotenv()
client = OpenAI()

class Gpt4LLM(LLM):
    def _call(self, prompt: srt, stop=None):
        response = client.chat.completions.create(
            model = "gpt-4.1",
            max_tokens=500,
            temperature=0.5,
            messages=[
                {"role":"system","content":"You are an exoert AI tutor explaining complex technical concepts in a clear, detailed and helpful way."},
                {"role":"user","content":prompt}
            ]
        )
        return response.choices[0].message.content


# OpenAI Embeddings class
class OpenAIEmbeddings:
    def __init__(self, client, model="text-embedding-3-large"):
        self.client = client
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return response.data[0].embedding


class RAGChatbot:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.llm = Gpt4LLM()
        self.embeddings = OpenAIEmbeddings()

    def _load_documents(self, file_paths):
        all_docs = []
        for file_path in file_paths:
            if file_path.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path)
                all_docs.extend(loader.load())
            elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                df = pd.read_excel(file_path)
                text_content = df.to_string()
                all_docs.append(Document(page_content=text_content, metadata={"source": file_path, "type": "excel"}))
            else:
                print(f"Unsupported file type: {file_path}")
                continue
        return all_docs

    def _split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(documents)

    def ingest_documents(self, file_paths):
        print("Loading documents...")
        documents = self._load_documents(file_paths)
        print(f"Loaded {len(documents)} documents.")

        print("Splitting documents into chunks...")
        chunks = self._split_documents(documents)
        print(f"Created {len(chunks)} chunks.")

        print("Creating vector store and embeddings...")
        self.vectorstore = Chroma.from_documents(chunks, self.embeddings, persist_directory=self.persist_directory)
        self.vectorstore.persist()
        print("Vector store created and persisted.")
        self.retriever = self.vectorstore.as_retriever()

    def setup_qa_chain(self):
        if not self.retriever:
            raise ValueError("Documents must be ingested first. Call ingest_documents().")

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )
        print("QA chain set up. Note: Using a dummy embedding and small LLM for demonstration. Actual answers will be limited.")

    def ask_question(self, question):
        if not self.qa_chain:
            raise ValueError("QA chain must be set up first. Call setup_qa_chain().")

        print(f"\nAsking: {question}")
        response = self.qa_chain.invoke({"query": question})
        return response["result"]

if __name__ == "__main__":

    chatbot = RAGChatbot()

    # Ingest documents Enter name of documets store in project folder
    chatbot.ingest_documents(["sample.pdf", "sample.xlsx"])

    # Setup QA chain
    chatbot.setup_qa_chain()

    # Ask questions
    print("\nChatbot is ready! Type 'exit' pr 'quit' to end the conservation")
    while True:
        user_query = input("\nYour question: ")
        if user_query.lower() in ['exit','quit']
        print("Exiting chatbot. Goodbye!")
        break

    answer = chatbot.ask_question(user_query)
    print(f"Chatbot: {answer}")
