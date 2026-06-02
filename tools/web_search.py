import json
import os
from pathlib import Path

CACHE_PATH = Path("data/api_cache.json")


def _ensure_cache_file() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE_PATH.exists():
        CACHE_PATH.write_text(json.dumps({}, indent=2), encoding="utf-8")


def _load_cache() -> dict:
    _ensure_cache_file()
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_cache(cache: dict) -> None:
    _ensure_cache_file()
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _cache_key(query: str) -> str:
    return query.strip().lower()


def web_search(query: str) -> str:
    """Search for case law, precedents, or verify citations using Tavily."""
    if not query or not query.strip():
        return ""

    cache = _load_cache()
    key = _cache_key(query)
    if key in cache:
        return cache[key]

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Tavily API key not configured."

    try:
        from tavily import TavilyClient
    except ImportError as exc:
        return f"Tavily client unavailable: {exc}"

    try:
        client = TavilyClient(api_key=api_key)
        result = client.get_search_context(query=query, search_depth="advanced")
        cache[key] = result
        _save_cache(cache)
        return result
    except Exception as exc:
        return f"Error during web search: {exc}"
