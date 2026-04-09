import os
import json
import asyncio
from langchain_groq import ChatGroq
from langsmith import traceable
from tools.web_search import web_search
from tools.case_law_parser import extract_citations
from dotenv import load_dotenv

load_dotenv()

JUDGE_SYSTEM_PROMPT = """
You are an impartial federal judge. Evaluate both the prosecution and defense arguments on legal merit alone. Score each argument. Flag any citations that appear fabricated or inaccurate. Provide a preliminary ruling based on legal soundness only.

Your response must be a JSON object with the following keys:
- prosecution_scores: dict with keys legal_soundness (0-10), evidence_quality (0-10), logical_coherence (0-10), precedent_accuracy (0-10)
- defense_scores: dict with keys legal_soundness (0-10), evidence_quality (0-10), logical_coherence (0-10), precedent_accuracy (0-10)
- preliminary_ruling: str (one of: "prosecution_favored", "defense_favored", "too_close")
- reasoning_summary: str (2-3 sentences)
"""

async def verify_citation(citation: str, side: str):
    # Pass citation as query, check if results confirm it exists
    context = web_search(citation)
    # Basic logic: if context is very short or contains error, it might be hallucinated
    # In a real scenario, we'd use LLM to verify search results
    if "Error" in context or len(context) < 50:
        return {"side": side, "citation": citation, "reason": "Could not verify citation existence via web search."}
    return None

from agents.utils import parse_json_from_llm, safe_ainvoke

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
        response = await safe_ainvoke(messages, temperature=0.1)
        result = parse_json_from_llm(response.content)
        
        # Hallucination detection: verify citations
        citations_to_verify = []
        for c in prosecution.get('cited_sources', []):
            citations_to_verify.append(verify_citation(c, "prosecution"))
        for c in defense.get('cited_sources', []):
            citations_to_verify.append(verify_citation(c, "defense"))
            
        verification_results = await asyncio.gather(*citations_to_verify)
        hallucination_flags = [v for v in verification_results if v is not None]
        
        return {
            "judge_scores": result,
            "hallucination_flags": hallucination_flags
        }
    except Exception as e:
        print(f"Error in Judge Agent: {e}")
        return {
            "judge_scores": {"prosecution_scores": {}, "defense_scores": {}, "preliminary_ruling": "N/A", "reasoning_summary": "Evaluation failed."},
            "hallucination_flags": [{"side": "System", "citation": "N/A", "reason": f"Evaluation parsing failed: {str(e)}"}]
        }
