# vectorstore_utils.py

import os
import pandas as pd
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# function to load CSV as documents
def load_csv_as_documents(file_path: str):
    """Convert CSV rows into Document objects for embedding."""
    df = pd.read_csv(file_path)
    docs = []
    for _, row in df.iterrows():
        row_text = " | ".join(f"{col}: {val}" for col, val in row.items())
        docs.append(Document(page_content=row_text))
    return docs

# function to load TXT as documents
def load_text_as_documents(file_path: str):
    """Load .txt file into Document objects."""
    loader = TextLoader(file_path)
    return loader.load()

# function to build vectorstore
def build_vectorstore(file_path: str):
    """Create or update a Chroma retriever from .txt or .csv."""
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".csv":
        docs = load_csv_as_documents(file_path)
    elif ext == ".txt":
        docs = load_text_as_documents(file_path)
    else:
        raise ValueError("Unsupported file type. Please use .txt or .csv")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    persist_dir = "./chroma_db"
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)

    return vectorstore.as_retriever(search_kwargs={"k": 3})
