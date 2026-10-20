import re
import json
import asyncio
import os
from typing import Any

GLOBAL_API_STATS = {
    "total_calls": 0,
    "fallback_count": 0,
    "model_usage": {},
}


def get_api_stats() -> dict[str, Any]:
    return dict(GLOBAL_API_STATS)


def reset_api_stats() -> None:
    GLOBAL_API_STATS["total_calls"] = 0
    GLOBAL_API_STATS["fallback_count"] = 0
    GLOBAL_API_STATS["model_usage"] = {}


def parse_json_from_llm(content: str) -> dict[str, Any]:
    """Extract JSON from LLM responses even when wrapped in markdown, code fences, or surrounding text."""
    candidate = content.strip()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Extract markdown code blocks
    code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', candidate, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        block_text = code_block_match.group(1).strip()
        try:
            return json.loads(block_text)
        except json.JSONDecodeError:
            pass

    # Extract outermost JSON object { ... }
    first_brace = candidate.find('{')
    last_brace = candidate.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        slice_text = candidate[first_brace : last_brace + 1].strip()
        try:
            return json.loads(slice_text)
        except json.JSONDecodeError:
            pass

    # Extract outermost JSON array [ ... ]
    first_bracket = candidate.find('[')
    last_bracket = candidate.rfind(']')
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        slice_text = candidate[first_bracket : last_bracket + 1].strip()
        try:
            parsed = json.loads(slice_text)
            if isinstance(parsed, list):
                return {"items": parsed}
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse valid JSON from response content: {candidate[:200]}...")


async def safe_ainvoke(
    messages: list[tuple[str, str]],
    temperature: float = 0.2,
    model_names: list[str] | None = None,
    max_retries: int = 2,
):
    from langchain_groq import ChatGroq
    import random

    if model_names is None:
        model_names = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
        ]

    api_key = os.getenv("GROQ_API_KEY", "")
    last_error: Exception | None = None

    for model_index, model_name in enumerate(model_names):
        for attempt in range(max_retries):
            try:
                llm = ChatGroq(
                    model_name=model_name,
                    temperature=temperature,
                    api_key=api_key or None,
                    timeout=45.0,
                    max_retries=1,
                )
                await asyncio.sleep(random.uniform(0.05, 0.2))
                response = await llm.ainvoke(messages)

                GLOBAL_API_STATS["total_calls"] += 1
                if model_index > 0:
                    GLOBAL_API_STATS["fallback_count"] += 1
                GLOBAL_API_STATS["model_usage"][model_name] = (
                    GLOBAL_API_STATS["model_usage"].get(model_name, 0) + 1
                )

                return response
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                # Rate limit retry
                if "429" in message or "rate limit" in message:
                    wait_time = (attempt + 1) * 1.5 + random.uniform(0.5, 1.0)
                    await asyncio.sleep(wait_time)
                    continue
                # If model is not found or decommissioned, break immediately to next model
                if "404" in message or "not found" in message or "deprecated" in message:
                    break
                # Other transient errors
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0)
                    continue
                break

    if last_error:
        raise last_error
    raise RuntimeError("No model in the fallback chain succeeded.")
