from rag.vector_store import load_vector_store


def retrieve_context(query: str, k: int = 5, session_id: str | None = None) -> str:
    if not query or not query.strip():
        return ""

    vector_store = load_vector_store(session_id=session_id)
    if not vector_store:
        return ""

    try:
        # Use MMR for diversity in evidence retrieval
        docs = vector_store.max_marginal_relevance_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs if doc.page_content])
    except Exception as exc:
        print(f"Warning: Vector retrieval failed: {exc}")
        return ""
