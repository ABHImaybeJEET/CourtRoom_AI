from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from rag.vector_store import get_embeddings, save_vector_store, load_vector_store

DATA_PATH = Path("data")


def ingest_pdf(file_path: str, session_id: str | None = None):
    from langchain_community.document_loaders import PyMuPDFLoader

    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    parsed_text = "\n\n".join(doc.page_content for doc in docs if doc.page_content)
    return ingest_text(parsed_text, session_id=session_id)


def ingest_text(text: str, session_id: str | None = None):
    if not text or not text.strip():
        return []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    if not chunks:
        return []

    embeddings = get_embeddings()
    vector_store = load_vector_store(session_id=session_id)

    if vector_store:
        vector_store.add_texts(chunks)
    else:
        vector_store = FAISS.from_texts(chunks, embeddings)

    save_vector_store(vector_store, session_id=session_id)
    return chunks
