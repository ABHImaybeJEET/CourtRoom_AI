import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from rag.vector_store import get_embeddings, save_vector_store, load_vector_store

def ingest_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    return ingest_text(text)

def ingest_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = get_embeddings()
    
    vector_store = load_vector_store()
    if vector_store:
        vector_store.add_texts(chunks)
    else:
        vector_store = FAISS.from_texts(chunks, embeddings)
    
    save_vector_store(vector_store)
    return chunks
