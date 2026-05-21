import os


def web_search(query: str) -> str:
    """Search for case law, precedents, or verify citations using Tavily."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Tavily API key not configured."

    try:
        from tavily import TavilyClient
    except ImportError as exc:
        return f"Tavily client unavailable: {exc}"

    try:
        client = TavilyClient(api_key=api_key)
        return client.get_search_context(query=query, search_depth="advanced")
    except Exception as exc:
        return f"Error during web search: {exc}"
