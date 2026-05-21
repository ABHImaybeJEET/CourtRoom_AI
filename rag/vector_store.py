from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

INDEX_PATH = Path("data/faiss_index")


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def save_vector_store(vector_store, path: Path = INDEX_PATH):
    path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(path))


def load_vector_store(path: Path = INDEX_PATH):
    index_file = path / "index.faiss"
    if index_file.exists():
        return FAISS.load_local(str(path), get_embeddings(), allow_dangerous_deserialization=True)
    return None
