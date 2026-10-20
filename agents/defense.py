from langsmith import traceable
from agents.utils import parse_json_from_llm, safe_ainvoke, AgentExecutionError

DEFENSE_SYSTEM_PROMPT = """
You are a senior defense attorney representing the accused or respondent in a formal courtroom proceeding.
Vigorously defend your client by highlighting factual reasonable doubt, procedural gaps, unproven assumptions, and statutory protections or safe harbors.
Maintain professional legal decorum while passionately defending your client.

Your response must be a valid JSON object with the following keys:
- argument_text: str (full legal defense argument, 300-500 words, clearly organized)
- cited_sources: list[str] (names of specific statutes, safe harbors, or precedents cited)
- confidence_score: float (0.0 to 1.0, self-assessed strength of defense)
- legal_strategy: str (one of: "statutory", "precedent-based", "circumstantial", "forensic", "character")
"""


@traceable
async def defense_agent(state: dict) -> dict:
    round_number = state.get("round_number", 0)
    max_rounds = state.get("max_rounds", 2)
    
    stage_name = "Defense Opening" if round_number == 0 else (
        "Rebuttal Defense" if round_number < max_rounds - 1 else "Closing Defense Argument"
    )
    
    current_prosecution = state.get("prosecution_argument", {}).get("argument_text", "")

    prompt = f"""
    TRIAL STAGE: {stage_name} (Round {round_number + 1} of {max_rounds})

    CASE SUMMARY:
    {state.get('case_description', '')}

    OPPOSING PROSECUTION ARGUMENT:
    {current_prosecution}

    RETRIEVED EVIDENCE & CONTEXT:
    {state.get('retrieved_context', 'None submitted.')}

    BACKGROUND PRECEDENT / EXTERNAL SEARCH:
    {state.get('web_search_results', 'None available.')}

    PRIOR DEBATE HISTORY:
    {state.get('debate_history', [])}

    Deliver your {stage_name}. Directly rebut the prosecution's claims, expose evidentiary leaps, and protect your client.
    Output strictly in the specified JSON format.
    """

    messages = [
        ("system", DEFENSE_SYSTEM_PROMPT.strip()),
        ("human", prompt.strip()),
    ]

    try:
        response = await safe_ainvoke(
            messages,
            temperature=0.2,
            model_names=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        )
        result = parse_json_from_llm(response.content)
        if not isinstance(result, dict) or "argument_text" not in result:
            raise ValueError("Parsed JSON does not contain 'argument_text'")
        return {"defense_argument": result}
    except Exception as exc:
        raise AgentExecutionError(f"Defense Agent failed during {stage_name}: {exc}") from exc
