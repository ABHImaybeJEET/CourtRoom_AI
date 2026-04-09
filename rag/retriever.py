from rag.vector_store import load_vector_store

def retrieve_context(query, k=5):
    vector_store = load_vector_store()
    if not vector_store:
        return ""
    
    # Use MMR for diversity as requested
    docs = vector_store.max_marginal_relevance_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])
