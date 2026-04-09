import os
import faiss
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_PATH = "data/faiss_index"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def save_vector_store(vector_store, path=INDEX_PATH):
    if not os.path.exists(path):
        os.makedirs(path)
    vector_store.save_local(path)

def load_vector_store(path=INDEX_PATH):
    if os.path.exists(os.path.join(path, "index.faiss")):
        return FAISS.load_local(path, get_embeddings(), allow_dangerous_deserialization=True)
    return None
