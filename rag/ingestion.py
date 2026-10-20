from pathlib import Path
from langchain_community.vectorstores import FAISS
from rag.vector_store import get_embeddings, save_vector_store, load_vector_store

DATA_PATH = Path("data")


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    """Pure-Python recursive chunking that doesn't trigger spacy/pydantic-v1 issues on Python 3.14."""
    if not text or not text.strip():
        return []

    # Try importing RecursiveCharacterTextSplitter if possible, fallback to clean recursive splitting
    try:
        import sys
        # Prevent spacy from loading if present in modules
        sys.modules.setdefault("spacy", None)
        from langchain_text_splitters.character import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)
    except Exception:
        pass

    # Pure Python recursive text splitter fallback
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_len = 0

    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue
        p_len = len(p_clean)

        if current_len + p_len + 2 <= chunk_size:
            current_chunk.append(p_clean)
            current_len += p_len + 2
        else:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            # If paragraph itself is larger than chunk_size, split by sentences
            if p_len > chunk_size:
                sentences = p_clean.split(". ")
                sub_chunk = []
                sub_len = 0
                for s in sentences:
                    s_clean = s.strip()
                    if sub_len + len(s_clean) + 2 <= chunk_size:
                        sub_chunk.append(s_clean)
                        sub_len += len(s_clean) + 2
                    else:
                        if sub_chunk:
                            chunks.append(". ".join(sub_chunk) + ".")
                        sub_chunk = [s_clean]
                        sub_len = len(s_clean)
                if sub_chunk:
                    chunks.append(". ".join(sub_chunk))
                current_chunk = []
                current_len = 0
            else:
                current_chunk = [p_clean]
                current_len = p_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def ingest_pdf(file_path: str, session_id: str | None = None) -> list[str]:
    from langchain_community.document_loaders import PyMuPDFLoader

    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    parsed_text = "\n\n".join(doc.page_content for doc in docs if doc.page_content)
    return ingest_text(parsed_text, session_id=session_id)


def ingest_text(text: str, session_id: str | None = None) -> list[str]:
    if not text or not text.strip():
        return []

    chunks = split_text(text, chunk_size=800, chunk_overlap=100)
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
