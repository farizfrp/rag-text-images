# indexer.py
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

def index_documents(documents):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore