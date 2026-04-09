import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

def web_search(query: str):
    """
    Search for case law, precedents, or verify citations using Tavily.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Tavily API key not found."
    
    tavily = TavilyClient(api_key=api_key)
    try:
        # Using search_context as requested
        context = tavily.get_search_context(query=query, search_depth="advanced")
        return context
    except Exception as e:
        return f"Error during web search: {str(e)}"
