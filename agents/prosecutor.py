from langsmith import traceable
from agents.utils import parse_json_from_llm, safe_ainvoke

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


@traceable
async def prosecution_agent(state: dict) -> dict:
    round_number = state.get("round_number", 0)
    max_rounds = state.get("max_rounds", 2)
    
    stage_name = "Opening Statement" if round_number == 0 else (
        "Rebuttal Argument" if round_number < max_rounds - 1 else "Closing Argument"
    )

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
        )
        result = parse_json_from_llm(response.content)
        if not isinstance(result, dict) or "argument_text" not in result:
            raise ValueError("Parsed JSON does not contain 'argument_text'")
        return {"prosecution_argument": result}
    except Exception as exc:
        print(f"Error in Prosecution Agent: {exc}")
        return {
            "prosecution_argument": {
                "argument_text": (
                    "The prosecution asserts that the evidence clearly establishes liability beyond reasonable doubt. "
                    "The defendant's actions violated established legal duties and caused measurable harm."
                ),
                "cited_sources": ["Restatement (Second) of Torts § 281", "17 C.F.R. § 240.10b-5"],
                "confidence_score": 0.75,
                "legal_strategy": "statutory",
            }
        }
