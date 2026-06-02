import asyncio
from langsmith import traceable
from tools.web_search import web_search
from agents.utils import parse_json_from_llm, safe_ainvoke


JUDGE_SYSTEM_PROMPT = """
You are a judge reviewing both the prosecution's and defense's written arguments. Evaluate each argument on legal merit, the quality of evidence, and reasoning.
Identify any citations that are likely inaccurate or unsupported. Use only the case description and available context.

Your response must be a JSON object with the following keys:
- prosecution_scores: dict with keys legal_soundness (0-10), evidence_quality (0-10), logical_coherence (0-10), precedent_accuracy (0-10)
- defense_scores: dict with keys legal_soundness (0-10), evidence_quality (0-10), logical_coherence (0-10), precedent_accuracy (0-10)
- preliminary_ruling: str (one of: "prosecution_favored", "defense_favored", "too_close")
- reasoning_summary: str (A concise 3-5 sentence summary explaining the ruling and the strongest points from each side.)

async def verify_citation(citation: str, side: str):
    context = web_search(citation)
    if "Error" in context or len(context) < 50:
        return {"side": side, "citation": citation, "reason": "Could not verify citation existence via web search."}
    return None


@traceable
async def judge_agent(state: dict):
    prosecution = state.get('prosecution_argument', {})
    defense = state.get('defense_argument', {})

    prompt = f"""
    Prosecution Argument: {prosecution.get('argument_text', '')}
    Prosecution Citations: {prosecution.get('cited_sources', [])}

    Defense Argument: {defense.get('argument_text', '')}
    Defense Citations: {defense.get('cited_sources', [])}

    Retrieved Context: {state.get('retrieved_context', '')}

    Evaluate the legal arguments.
    """

    messages = [
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human", prompt)
    ]

    try:
        response = await safe_ainvoke(
            messages,
            temperature=0.1,
            model_names=["llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"],
        )
        result = parse_json_from_llm(response.content)

        citation_side_map = {}
        for c in prosecution.get('cited_sources', []):
            citation_side_map.setdefault(c, []).append('prosecution')
        for c in defense.get('cited_sources', []):
            citation_side_map.setdefault(c, []).append('defense')

        citations_to_verify = [verify_citation(citation, ', '.join(sorted(set(sides))))
                               for citation, sides in citation_side_map.items()]
        verification_results = await asyncio.gather(*citations_to_verify) if citations_to_verify else []
        hallucination_flags = [v for v in verification_results if v is not None]

        return {
            "judge_scores": result,
            "hallucination_flags": hallucination_flags
        }
    except Exception as exc:
        print(f"Error in Judge Agent: {exc}")
        return {
            "judge_scores": {
                "prosecution_scores": {},
                "defense_scores": {},
                "preliminary_ruling": "N/A",
                "reasoning_summary": "Evaluation failed.",
            },
            "hallucination_flags": [
                {
                    "side": "System",
                    "citation": "N/A",
                    "reason": f"Evaluation parsing failed: {str(exc)}",
                }
            ],
        }
