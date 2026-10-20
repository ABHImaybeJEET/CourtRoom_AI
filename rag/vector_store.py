from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DEFAULT_INDEX_PATH = Path("data/faiss_index")
_EMBEDDINGS_INSTANCE = None


def get_embeddings():
    global _EMBEDDINGS_INSTANCE
    if _EMBEDDINGS_INSTANCE is None:
        _EMBEDDINGS_INSTANCE = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _EMBEDDINGS_INSTANCE


def get_session_index_path(session_id: str | None = None) -> Path:
    if session_id:
        # Sanitize session_id to alphanumeric/dash
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return Path(f"data/sessions/{safe_id}/faiss_index")
    return DEFAULT_INDEX_PATH


def save_vector_store(vector_store, session_id: str | None = None, path: Path | None = None):
    target_path = path or get_session_index_path(session_id)
    target_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(target_path))


def load_vector_store(session_id: str | None = None, path: Path | None = None):
    target_path = path or get_session_index_path(session_id)
    index_file = target_path / "index.faiss"
    if index_file.exists():
        try:
            return FAISS.load_local(
                str(target_path),
                get_embeddings(),
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            print(f"Warning: Failed to load FAISS index from {target_path}: {exc}")
            return None
    return None
