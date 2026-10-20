import pytest
from pathlib import Path
from rag.vector_store import get_session_index_path, load_vector_store
from rag.retriever import retrieve_context


def test_session_index_path_isolation():
    path_a = get_session_index_path("trial-123")
    path_b = get_session_index_path("trial-456")
    default_path = get_session_index_path(None)

    assert path_a != path_b
    assert "trial-123" in str(path_a)
    assert "trial-456" in str(path_b)
    assert default_path == Path("data/faiss_index")


def test_retrieve_empty_context():
    # Calling retrieve_context on non-existent session returns empty string without error
    result = retrieve_context("What happened?", k=3, session_id="non-existent-session-xyz")
    assert result == ""
