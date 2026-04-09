import os
import json
from langchain_groq import ChatGroq
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

DEFENSE_SYSTEM_PROMPT = """
You are a brilliant defense attorney. Your sole goal is to dismantle the prosecution's argument, challenge evidence admissibility, introduce reasonable doubt, and protect your client. Find loopholes. Be persuasive. Never fold under pressure.

Your response must be a JSON object with the following keys:
- argument_text: str (full legal argument, 300-500 words)
- cited_sources: list[str] (case names and statutes cited)
- confidence_score: float (0.0 to 1.0, self-assessed)
- legal_strategy: str (one of: "circumstantial", "statutory", "precedent-based", "forensic", "character")
"""

from agents.utils import parse_json_from_llm, safe_ainvoke

@traceable
async def defense_agent(state: dict):
    current_prosecution = state.get('prosecution_argument', {}).get('argument_text', "")
    
    prompt = f"""
    Case Description: {state.get('case_description', '')}
    Retrieved Context: {state.get('retrieved_context', '')}
    Web Search Results: {state.get('web_search_results', '')}
    Debate History: {state.get('debate_history', [])}
    Current Prosecution Argument: {current_prosecution}
    
    Dismantle the case and defend your client.
    """
    
    messages = [
        ("system", DEFENSE_SYSTEM_PROMPT),
        ("human", prompt)
    ]
    
    try:
        response = await safe_ainvoke(messages, temperature=0.2)
        result = parse_json_from_llm(response.content)
        return {"defense_argument": result}
    except Exception as e:
        print(f"Error in Defense Agent: {e}\nResponse was: {response.content if 'response' in locals() else 'None'}")
        return {"defense_argument": {"argument_text": "Error generating argument. The model may have returned invalid formatting.", "cited_sources": [], "confidence_score": 0.0, "legal_strategy": "N/A"}}
