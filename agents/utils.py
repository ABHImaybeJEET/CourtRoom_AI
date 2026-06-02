import re
import json
import asyncio
from typing import Any


def parse_json_from_llm(content: str) -> dict[str, Any]:
    """Extract JSON from LLM responses even when wrapped in markdown or code fences."""
    candidate = content.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', candidate, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    object_match = re.search(r'(\{.*\})', candidate, re.DOTALL)
    if object_match:
        try:
            return json.loads(object_match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError("Could not parse JSON from response")


def _get_streamlit_api_stats() -> dict[str, Any] | None:
    try:
        import streamlit as st
    except ImportError:
        return None

    if "api_stats" not in st.session_state:
        st.session_state.api_stats = {"total_calls": 0, "fallback_count": 0, "model_usage": {}}
    return st.session_state.api_stats


async def safe_ainvoke(
    messages: list[tuple[str, str]],
    temperature: float = 0.2,
    model_names: list[str] | None = None,
    max_retries: int = 2,
):
    from langchain_groq import ChatGroq
    import random

    api_stats = _get_streamlit_api_stats()
    if model_names is None:
        model_names = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "llama3-8b-8192",
        ]

    last_error = None
    for model_index, model_name in enumerate(model_names):
        for attempt in range(max_retries):
            try:
                llm = ChatGroq(model_name=model_name, temperature=temperature)
                await asyncio.sleep(random.uniform(0.1, 0.4))
                response = await llm.ainvoke(messages)

                if api_stats is not None:
                    api_stats["total_calls"] += 1
                    if model_index > 0:
                        api_stats["fallback_count"] += 1
                    api_stats["model_usage"][model_name] = api_stats["model_usage"].get(model_name, 0) + 1

                return response
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                if "429" in message or "rate limit" in message:
                    wait_time = (attempt + 1) * 1.5 + random.uniform(0.5, 1.0)
                    await asyncio.sleep(wait_time)
                    continue
                break

    raise last_error
