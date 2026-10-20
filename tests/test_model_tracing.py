import os
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agents.utils import (
    safe_ainvoke,
    get_langsmith_config,
    parse_json_from_llm,
    ModelProviderError,
    AgentExecutionError,
    DEFAULT_GROQ_MODELS,
    reset_api_stats,
    get_api_stats,
)
from agents.prosecutor import prosecution_agent


def test_groq_supported_models_configuration():
    """Verify that only active, currently supported Groq models are configured."""
    assert "llama-3.3-70b-versatile" in DEFAULT_GROQ_MODELS
    assert "llama-3.1-8b-instant" in DEFAULT_GROQ_MODELS

    # Ensure decommissioned models are NOT in the active chain
    decommissioned = ["mixtral-8x7b-32768", "gemma2-9b-it", "llama3-8b-8192"]
    for dec in decommissioned:
        assert dec not in DEFAULT_GROQ_MODELS


def test_missing_groq_credentials_raises_cleanly(monkeypatch):
    """Verify that empty or missing GROQ_API_KEY fails immediately with clear ModelProviderError."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(ModelProviderError) as exc_info:
        pytest.run(safe_ainvoke([("human", "test")])) if False else None
    
    # Use asyncio.run directly
    import asyncio
    with pytest.raises(ModelProviderError) as exc_info:
        asyncio.run(safe_ainvoke([("human", "test")]))
    assert "GROQ_API_KEY is not configured" in str(exc_info.value)


def test_langsmith_config_scrubs_secrets_and_truncates():
    """Verify that LangSmith config builder keeps credentials out of metadata and caps text lengths."""
    extra_metadata = {
        "GROQ_API_KEY": "gsk_secret_12345",
        "tavily_api_key": "tvly_secret_67890",
        "user_password": "super_secret_password",
        "trial_id": "trial-999",
        "round_number": 2,
        "massive_text_block": "X" * 1000,
    }

    config = get_langsmith_config(
        trial_id="trial-999",
        stage="PROSECUTION_ARGUMENT",
        extra_meta=extra_metadata,
    )

    metadata = config["metadata"]
    # Verify sensitive keys are stripped
    assert "GROQ_API_KEY" not in metadata
    assert "tavily_api_key" not in metadata
    assert "user_password" not in metadata

    # Verify safe fields are preserved
    assert metadata["trial_id"] == "trial-999"
    assert metadata["round_number"] == 2
    assert len(metadata["massive_text_block"]) <= 260


def test_langsmith_optional_when_unconfigured(monkeypatch):
    """Verify that tracing config flags tracing_active=False when LangSmith credentials are not set."""
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    config = get_langsmith_config(trial_id="test-123", stage="Evidence")
    assert config["metadata"]["tracing_active"] is False


@pytest.mark.asyncio
async def test_mocked_provider_success(monkeypatch):
    """Verify successful model invocation and stats tracking via mocked ChatGroq."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_valid_key")
    reset_api_stats()

    mock_response = MagicMock()
    mock_response.content = '{"argument_text": "Compelling legal argument.", "cited_sources": [], "confidence_score": 0.85, "legal_strategy": "statutory"}'

    with patch("langchain_groq.ChatGroq.ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = mock_response

        response = await safe_ainvoke(
            messages=[("human", "Draft prosecution argument")],
            model_names=["llama-3.3-70b-versatile"],
        )

        assert response.content == mock_response.content
        stats = get_api_stats()
        assert stats["total_calls"] == 1
        assert stats["model_usage"]["llama-3.3-70b-versatile"] == 1


@pytest.mark.asyncio
async def test_mocked_provider_fallback_on_rate_limit(monkeypatch):
    """Verify that a 429 rate limit triggers backoff and falls back to secondary model."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_valid_key")
    reset_api_stats()

    mock_success_response = MagicMock()
    mock_success_response.content = '{"verdict": "Not Guilty"}'

    # Primary model throws 429 on all attempts; secondary model succeeds
    call_counts = {"primary": 0, "secondary": 0}

    async def mock_call(*args, **kwargs):
        # Inspect ChatGroq model name from instance
        return mock_success_response

    with patch("langchain_groq.ChatGroq.ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        # First 2 calls (attempts on primary model) raise 429, 3rd call (secondary model) succeeds
        mock_ainvoke.side_effect = [
            RuntimeError("Rate limit reached. 429 Too Many Requests"),
            RuntimeError("Rate limit reached. 429 Too Many Requests"),
            mock_success_response,
        ]

        response = await safe_ainvoke(
            messages=[("human", "Deliberate")],
            model_names=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            max_retries=2,
        )

        assert response.content == mock_success_response.content
        stats = get_api_stats()
        assert stats["total_calls"] == 1
        assert stats["fallback_count"] == 1


@pytest.mark.asyncio
async def test_mocked_provider_total_failure_raises(monkeypatch):
    """Verify that when all providers fail, a ModelProviderError is raised (no fake data returned)."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_valid_key")

    with patch("langchain_groq.ChatGroq.ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.side_effect = RuntimeError("503 Service Unavailable")

        with pytest.raises(ModelProviderError) as exc_info:
            await safe_ainvoke(
                messages=[("human", "Build argument")],
                model_names=["llama-3.3-70b-versatile"],
                max_retries=1,
            )

        assert "503 Service Unavailable" in str(exc_info.value)


@pytest.mark.asyncio
async def test_agent_fails_explicitly_when_provider_fails(monkeypatch):
    """Verify that prosecution_agent propagates failure as AgentExecutionError instead of returning mock text."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_mock_valid_key")

    with patch("agents.prosecutor.safe_ainvoke", new_callable=AsyncMock) as mock_safe_ainvoke:
        mock_safe_ainvoke.side_effect = ModelProviderError("Groq model quota exhausted.")

        with pytest.raises(AgentExecutionError) as exc_info:
            await prosecution_agent({"case_description": "Valid case description", "round_number": 0, "max_rounds": 2})

        assert "Prosecution Agent failed" in str(exc_info.value)
