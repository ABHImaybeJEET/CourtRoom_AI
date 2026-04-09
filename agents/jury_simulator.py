import os
import json
import asyncio
import random
from langchain_groq import ChatGroq
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

JUROR_PROFILES = [
    {"age": 34, "gender": "Female", "occupation": "Teacher", "political_lean": "Liberal", "education_level": "Master's", "prior_jury_experience": False, "empathy_score": 8, "skepticism_score": 4, "media_bias": 6},
    {"age": 52, "gender": "Male", "occupation": "Engineer", "political_lean": "Conservative", "education_level": "Bachelor's", "prior_jury_experience": True, "empathy_score": 4, "skepticism_score": 9, "media_bias": 3},
    {"age": 28, "gender": "Non-binary", "occupation": "Nurse", "political_lean": "Moderate", "education_level": "Associate", "prior_jury_experience": False, "empathy_score": 9, "skepticism_score": 5, "media_bias": 7},
    {"age": 65, "gender": "Male", "occupation": "Retired Veteran", "political_lean": "Conservative", "education_level": "High School", "prior_jury_experience": True, "empathy_score": 5, "skepticism_score": 8, "media_bias": 4},
    {"age": 41, "gender": "Female", "occupation": "Small Business Owner", "political_lean": "Moderate", "education_level": "Bachelor's", "prior_jury_experience": False, "empathy_score": 6, "skepticism_score": 7, "media_bias": 5},
    {"age": 39, "gender": "Female", "occupation": "Social Worker", "political_lean": "Liberal", "education_level": "Master's", "prior_jury_experience": True, "empathy_score": 10, "skepticism_score": 3, "media_bias": 8},
    {"age": 45, "gender": "Male", "occupation": "Accountant", "political_lean": "Conservative", "education_level": "Bachelor's", "prior_jury_experience": False, "empathy_score": 3, "skepticism_score": 9, "media_bias": 2},
    {"age": 31, "gender": "Male", "occupation": "Construction Worker", "political_lean": "Moderate", "education_level": "High School", "prior_jury_experience": False, "empathy_score": 5, "skepticism_score": 6, "media_bias": 5},
    {"age": 58, "gender": "Female", "occupation": "Professor", "political_lean": "Liberal", "education_level": "PhD", "prior_jury_experience": True, "empathy_score": 7, "skepticism_score": 8, "media_bias": 6},
    {"age": 48, "gender": "Female", "occupation": "Homemaker", "political_lean": "Conservative", "education_level": "Some College", "prior_jury_experience": False, "empathy_score": 8, "skepticism_score": 5, "media_bias": 4},
    {"age": 37, "gender": "Male", "occupation": "Sales Manager", "political_lean": "Moderate", "education_level": "Bachelor's", "prior_jury_experience": False, "empathy_score": 4, "skepticism_score": 7, "media_bias": 7},
    {"age": 26, "gender": "Female", "occupation": "Journalist", "political_lean": "Liberal", "education_level": "Bachelor's", "prior_jury_experience": False, "empathy_score": 7, "skepticism_score": 6, "media_bias": 9},
]

from agents.utils import parse_json_from_llm, safe_ainvoke

@traceable
async def run_single_juror(n, profile, prosecution, defense, judge_summary, judge_scores):
    system_prompt = f"""
    You are Juror #{n}. Profile: Age {profile['age']}, {profile['occupation']}, Bias: {profile['political_lean']}, Empathy: {profile['empathy_score']}/10. 
    Review the arguments. Render a personal verdict.
    CRITICAL: Provide exactly 2 robust sentences explaining your logical reasoning, tying it explicitly to your occupation/age. Do not be vague.

    Return JSON:
    - verdict: "Guilty" | "Not Guilty"
    - confidence: float 0.0-1.0
    - reasoning: 2 substantive sentences.
    - swayed_by: "prosecution" | "defense" | "neither"
    """
    
    human_prompt = f"""
    Prosecution's Final Argument: {prosecution}
    Defense's Final Argument: {defense}
    Judge's Scoring & Summary: {judge_summary}
    Judge's Scores: {json.dumps(judge_scores)}
    
    What is your verdict?
    """
    
    messages = [
        ("system", system_prompt),
        ("human", human_prompt)
    ]
    
    try:
        response = await safe_ainvoke(messages, temperature=0.9)
        result = parse_json_from_llm(response.content)
        result['juror_id'] = n
        result['profile'] = profile
        return result
    except Exception as e:
        print(f"Error for Juror {n}: {e}")
        return {
            "juror_id": n,
            "profile": profile,
            "verdict": "Not Guilty", # Default on error
            "confidence": 0.5,
            "reasoning": "I am unsure due to conflicting or poorly formatted information.",
            "swayed_by": "neither"
        }

@traceable
async def jury_simulator_node(state: dict):
    prosecution = state.get('prosecution_argument', {}).get('argument_text', "")
    defense = state.get('defense_argument', {}).get('argument_text', "")
    judge_summary = state.get('judge_scores', {}).get('reasoning_summary', "")
    judge_scores = state.get('judge_scores', {})
    
    # Use a semaphore to limit concurrency. 
    # Calling 12 LLM instances at once often triggers hard rate limits.
    # Processing 3 at a time is much safer and more reliable.
    sem = asyncio.Semaphore(3)
    
    async def safe_run_juror(n, profile, p, d, js, jsc):
        async with sem:
            return await run_single_juror(n, profile, p, d, js, jsc)

    tasks = []
    for i, profile in enumerate(JUROR_PROFILES):
        tasks.append(safe_run_juror(i+1, profile, prosecution, defense, judge_summary, judge_scores))
        
    verdicts = await asyncio.gather(*tasks)
    
    guilty_count = sum(1 for v in verdicts if v['verdict'] == "Guilty")
    not_guilty_count = sum(1 for v in verdicts if v['verdict'] == "Not Guilty")
    
    final_verdict = "Hung Jury"
    if guilty_count > not_guilty_count:
        final_verdict = "Guilty"
    elif not_guilty_count > guilty_count:
        final_verdict = "Not Guilty"
        
    # Simple demographic analysis (Internal logic or another LLM call)
    # Let's use a quick LLM call for analysis
    analysis_prompt = f"Analyze these jury verdicts and explain which demographic segments voted which way and why based on their profiles: {json.dumps(verdicts)}"
    try:
        analysis_response = await safe_ainvoke([("system", "You are a legal demographic analyst."), ("human", analysis_prompt)], temperature=0.3)
        analysis_content = analysis_response.content
    except Exception as e:
        analysis_content = f"Demographic analysis unavailable due to API rate limits: {e}"
    
    return {
        "jury_verdicts": verdicts,
        "vote_count": {"guilty": guilty_count, "not_guilty": not_guilty_count},
        "final_verdict": final_verdict,
        "demographic_analysis": analysis_content
    }
