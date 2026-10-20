import re
import json
import asyncio
import os
import logging
from typing import Any, Callable

# Initialize structured logger for CourtRoom AI
logger = logging.getLogger("courtroom.agents")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Graceful LangSmith traceable decorator fallback
try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func: Callable):
            return func
        if args and callable(args[0]):
            return args[0]
        return decorator


class AgentExecutionError(RuntimeError):
    """Raised when an agent fails to execute its legal reasoning or schema generation."""
    pass


class ModelProviderError(AgentExecutionError):
    """Raised when an LLM provider call fails due to credentials, rate limits, or network timeouts."""
    pass


# Active Groq models tested and supported for legal reasoning
DEFAULT_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
]

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


def get_langsmith_config(trial_id: str | None = None, stage: str | None = None, extra_meta: dict | None = None) -> dict:
    """
    Construct safe LangSmith run configuration and metadata.
    Guarantees secrets and huge case bodies are never leaked into tracing metadata.
    """
    is_tracing = os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1")
    has_api_key = bool(os.getenv("LANGCHAIN_API_KEY", "").strip())
    
    metadata = {
        "trial_id": trial_id or "anonymous",
        "stage": stage or "general",
        "tracing_active": is_tracing and has_api_key,
    }
    if extra_meta:
        # Only copy non-sensitive primitive metadata
        for k, v in extra_meta.items():
            if any(s in k.lower() for s in ["key", "secret", "token", "password", "credential"]):
                continue
            if isinstance(v, (str, int, float, bool)):
                metadata[k] = v if not isinstance(v, str) or len(v) < 250 else f"{v[:250]}..."

    tags = ["courtroom-ai"]
    if stage:
        tags.append(stage.lower().replace("_", "-"))

    return {
        "run_name": f"CourtRoom [{stage or 'Execution'}]",
        "tags": tags,
        "metadata": metadata,
    }


async def safe_ainvoke(
    messages: list[tuple[str, str]],
    temperature: float = 0.2,
    model_names: list[str] | None = None,
    max_retries: int = 2,
    timeout_seconds: float = 45.0,
    tracing_metadata: dict | None = None,
):
    """
    Invoke Groq LLM with retries, fallback chain, and clean tracing/logging.
    Fails explicitly if GROQ_API_KEY is missing or if all models fail.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        logger.error("LLM call aborted: GROQ_API_KEY is not configured.")
        raise ModelProviderError(
            "GROQ_API_KEY is not configured on the server. Please set GROQ_API_KEY in your environment or .env file."
        )

    from langchain_groq import ChatGroq
    import random

    if model_names is None:
        model_names = list(DEFAULT_GROQ_MODELS)

    last_error: Exception | None = None

    for model_index, model_name in enumerate(model_names):
        for attempt in range(max_retries):
            logger.info(
                "Invoking LLM: model=%s, attempt=%d/%d, temp=%.2f",
                model_name,
                attempt + 1,
                max_retries,
                temperature,
            )
            try:
                llm = ChatGroq(
                    model_name=model_name,
                    temperature=temperature,
                    api_key=api_key,
                    timeout=timeout_seconds,
                    max_retries=1,
                )
                await asyncio.sleep(random.uniform(0.05, 0.15))

                # Safe execution with timeout safeguard
                response = await asyncio.wait_for(
                    llm.ainvoke(messages),
                    timeout=timeout_seconds,
                )

                GLOBAL_API_STATS["total_calls"] += 1
                if model_index > 0:
                    GLOBAL_API_STATS["fallback_count"] += 1
                    logger.warning("Fallback model '%s' succeeded (primary model was unavailable).", model_name)
                GLOBAL_API_STATS["model_usage"][model_name] = (
                    GLOBAL_API_STATS["model_usage"].get(model_name, 0) + 1
                )

                logger.info("LLM invocation succeeded: model=%s", model_name)
                return response

            except asyncio.TimeoutError:
                last_error = ModelProviderError(
                    f"Model '{model_name}' timed out after {timeout_seconds}s (attempt {attempt + 1}/{max_retries})."
                )
                logger.warning("Model call to '%s' timed out (attempt %d/%d).", model_name, attempt + 1, max_retries)
                continue

            except Exception as exc:
                last_error = exc
                message = str(exc).lower()

                # Authentication error -> immediately fail, retrying won't help
                if "401" in message or "invalid_api_key" in message or "authentication" in message:
                    logger.error("Groq API authentication failed: %s", exc)
                    raise ModelProviderError(f"Groq API authentication failed: {exc}") from exc

                # Rate limit (429) -> exponential backoff with jitter
                if "429" in message or "rate limit" in message:
                    wait_time = (attempt + 1) * 2.0 + random.uniform(0.5, 1.5)
                    logger.warning(
                        "Rate limit on '%s' (attempt %d/%d). Sleeping %.2fs before retry...",
                        model_name,
                        attempt + 1,
                        max_retries,
                        wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue

                # Deprecated / Model not found -> immediately move to next fallback model
                if "404" in message or "not found" in message or "deprecated" in message:
                    logger.warning(
                        "Model '%s' is deprecated or unavailable on Groq. Moving to fallback...",
                        model_name,
                    )
                    break

                # Transient network error
                logger.warning(
                    "Transient error on '%s' (attempt %d/%d): %s",
                    model_name,
                    attempt + 1,
                    max_retries,
                    exc,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0)
                    continue
                break

    if isinstance(last_error, AgentExecutionError):
        raise last_error
    raise ModelProviderError(f"All model providers in the fallback chain failed. Last error: {last_error}")
