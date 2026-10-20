import asyncio
import logging
from tools.web_search import web_search
from agents.utils import parse_json_from_llm, safe_ainvoke, AgentExecutionError, traceable

logger = logging.getLogger("courtroom.judge")

JUDGE_SYSTEM_PROMPT = """
You are a presiding judge reviewing both the prosecution's and defense's written arguments. Evaluate each argument on legal merit, the quality of evidence, precedent accuracy, and reasoning.
Identify any citations that are likely inaccurate or unsupported. Use only the case description and available context.

Your response must be a valid JSON object with the following keys:
- prosecution_scores: dict with numeric keys (0-10): legal_soundness, evidence_quality, logical_coherence, precedent_accuracy
- defense_scores: dict with numeric keys (0-10): legal_soundness, evidence_quality, logical_coherence, precedent_accuracy
- preliminary_ruling: str (one of: "prosecution_favored", "defense_favored", "too_close")
- reasoning_summary: str (A concise 3-5 sentence summary explaining the ruling and the strongest points from each side.)
"""


@traceable(name="Verify_Citation", run_type="tool")
async def verify_citation(citation: str, side: str) -> dict | None:
    if not citation or not citation.strip():
        return None
    context = web_search(citation)
    if "Error" in context or len(context.strip()) < 40 or "not configured" in context:
        logger.warning("Citation verification flagged: '%s' (side: %s)", citation, side)
        return {
            "side": side,
            "citation": citation,
            "reason": "Could not verify citation authenticity via precedent search.",
        }
    return None


@traceable(name="Judge_Agent", run_type="chain")
async def judge_agent(state: dict) -> dict:
    prosecution = state.get("prosecution_argument", {})
    defense = state.get("defense_argument", {})
    round_num = state.get("round_number", 0) + 1
    max_rounds = state.get("max_rounds", 2)
    trial_id = state.get("trial_id", "anonymous")

    logger.info("Judge agent evaluating arguments: round=%d/%d, trial_id='%s'", round_num, max_rounds, trial_id)

    prompt = f"""
    Current Trial Stage: Round {round_num} of {max_rounds}
    
    Prosecution Argument:
    {prosecution.get('argument_text', 'N/A')}
    Prosecution Citations: {prosecution.get('cited_sources', [])}
    Prosecution Strategy: {prosecution.get('legal_strategy', 'N/A')}

    Defense Argument:
    {defense.get('argument_text', 'N/A')}
    Defense Citations: {defense.get('cited_sources', [])}
    Defense Strategy: {defense.get('legal_strategy', 'N/A')}

    Retrieved Context / Evidence:
    {state.get('retrieved_context', '')}

    Evaluate the legal arguments thoroughly and return strictly valid JSON matching the requested schema.
    """

    messages = [
        ("system", JUDGE_SYSTEM_PROMPT.strip()),
        ("human", prompt.strip()),
    ]

    try:
        response = await safe_ainvoke(
            messages,
            temperature=0.1,
            model_names=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            tracing_metadata={"trial_id": trial_id, "stage": "Judge_Evaluation", "round": round_num},
        )
        result = parse_json_from_llm(response.content)

        if not isinstance(result, dict) or "prosecution_scores" not in result or "defense_scores" not in result:
            raise ValueError("Judge evaluation JSON response is missing required scoring keys.")

        citation_side_map: dict[str, list[str]] = {}
        for c in prosecution.get("cited_sources", []):
            if isinstance(c, str) and c.strip():
                citation_side_map.setdefault(c.strip(), []).append("prosecution")
        for c in defense.get("cited_sources", []):
            if isinstance(c, str) and c.strip():
                citation_side_map.setdefault(c.strip(), []).append("defense")

        citations_to_verify = [
            verify_citation(citation, ", ".join(sorted(set(sides))))
            for citation, sides in citation_side_map.items()
        ]
        verification_results = (
            await asyncio.gather(*citations_to_verify) if citations_to_verify else []
        )
        hallucination_flags = [v for v in verification_results if v is not None]

        logger.info(
            "Judge evaluation complete: preliminary_ruling='%s', citation_flags=%d",
            result.get("preliminary_ruling"),
            len(hallucination_flags),
        )

        return {
            "judge_scores": result,
            "hallucination_flags": hallucination_flags,
        }
    except Exception as exc:
        logger.error("Judge agent failed during round %d: %s", round_num, exc)
        raise AgentExecutionError(f"Judge Agent evaluation failed: {exc}") from exc
