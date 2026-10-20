import logging
from agents.utils import parse_json_from_llm, safe_ainvoke, AgentExecutionError, traceable

logger = logging.getLogger("courtroom.prosecutor")

PROSECUTOR_SYSTEM_PROMPT = """
You are a lead prosecuting attorney representing the state or plaintiff in a formal legal proceeding.
Deliver a compelling, rigorous legal argument based on the case facts, evidence, and applicable statutory or precedent authority.
Write with professional legal precision and gravitas.

Your response must be a valid JSON object with the following keys:
- argument_text: str (full legal argument, 300-500 words, clearly organized)
- cited_sources: list[str] (names of specific statutes, regulations, or case precedents cited)
- confidence_score: float (0.0 to 1.0, self-assessed strength of argument)
- legal_strategy: str (one of: "statutory", "precedent-based", "circumstantial", "forensic", "character")
"""


@traceable(name="Prosecution_Agent", run_type="chain")
async def prosecution_agent(state: dict) -> dict:
    round_number = state.get("round_number", 0)
    max_rounds = state.get("max_rounds", 2)
    trial_id = state.get("trial_id", "anonymous")
    
    stage_name = "Opening Statement" if round_number == 0 else (
        "Rebuttal Argument" if round_number < max_rounds - 1 else "Closing Argument"
    )

    logger.info("Prosecution agent starting: stage='%s', trial_id='%s'", stage_name, trial_id)

    prompt = f"""
    TRIAL STAGE: {stage_name} (Round {round_number + 1} of {max_rounds})

    CASE SUMMARY:
    {state.get('case_description', '')}

    RETRIEVED EVIDENCE & CONTEXT:
    {state.get('retrieved_context', 'None submitted.')}

    BACKGROUND PRECEDENT / EXTERNAL SEARCH:
    {state.get('web_search_results', 'None available.')}

    PRIOR DEBATE HISTORY:
    {state.get('debate_history', [])}

    Deliver your {stage_name}. Synthesize the facts, highlight the opposing weaknesses, and build an airtight argument for liability or guilt.
    Output strictly in the specified JSON format.
    """

    messages = [
        ("system", PROSECUTOR_SYSTEM_PROMPT.strip()),
        ("human", prompt.strip()),
    ]

    try:
        response = await safe_ainvoke(
            messages,
            temperature=0.2,
            model_names=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            tracing_metadata={"trial_id": trial_id, "stage": stage_name, "role": "prosecution"},
        )
        result = parse_json_from_llm(response.content)
        if not isinstance(result, dict) or "argument_text" not in result:
            raise ValueError("Parsed JSON does not contain 'argument_text'")
        logger.info(
            "Prosecution argument delivered: strategy='%s', confidence=%.2f, citations=%d",
            result.get("legal_strategy"),
            result.get("confidence_score", 0.0),
            len(result.get("cited_sources", [])),
        )
        return {"prosecution_argument": result}
    except Exception as exc:
        logger.error("Prosecution agent failed during %s: %s", stage_name, exc)
        raise AgentExecutionError(f"Prosecution Agent failed during {stage_name}: {exc}") from exc
