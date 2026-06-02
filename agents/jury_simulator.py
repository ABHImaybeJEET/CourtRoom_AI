import json
import asyncio
from langsmith import traceable

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


async def run_jury_panel(prosecution, defense, judge_summary, judge_scores, case_desc, profiles):
    system_prompt = f"""
    You are responsible for generating a jury verdict summary for a panel of jurors.
    Review the case description, the prosecution's final argument, the defense's final argument, and the judge's scoring summary.
    There are {len(profiles)} jurors, each with a detailed profile. For each juror, provide a JSON object with:
    - juror_id: int
    - verdict: "Guilty" | "Not Guilty" | "Liable" | "Not Liable"
    - confidence: float between 0.0 and 1.0
    - reasoning: 2-3 sentences relating the verdict back to the juror's profile
    - swayed_by: "prosecution" | "defense" | "neither"

    Return a JSON object with one key: "jury_verdicts" containing the list of juror objects.
    """

    human_prompt = f"""
    Case Description: {case_desc}

    Prosecution Argument: {prosecution}

    Defense Argument: {defense}

    Judge Summary: {judge_summary}
    Judge Scores: {json.dumps(judge_scores)}

    Juror Profiles: {json.dumps(profiles)}

    Provide the jury verdicts in strict JSON.
    """

    messages = [
        ("system", system_prompt),
        ("human", human_prompt)
    ]

    response = await safe_ainvoke(
        messages,
        temperature=0.8,
        model_names=["llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"],
    )

    result = parse_json_from_llm(response.content)
    if isinstance(result, dict) and "jury_verdicts" in result:
        return result["jury_verdicts"]
    if isinstance(result, list):
        return result
    raise ValueError("Invalid jury panel response format")


@traceable
async def jury_simulator_node(state: dict):
    prosecution = state.get('prosecution_argument', {}).get('argument_text', "")
    defense = state.get('defense_argument', {}).get('argument_text', "")
    judge_summary = state.get('judge_scores', {}).get('reasoning_summary', "")
    judge_scores = state.get('judge_scores', {})
    case_desc = state.get('case_description', '')
    profiles = JUROR_PROFILES[:6] if state.get('fast_mode') else JUROR_PROFILES

    try:
        verdicts = await run_jury_panel(prosecution, defense, judge_summary, judge_scores, case_desc, profiles)
    except Exception as exc:
        print(f"Jury panel fallback due to: {exc}")
        sem = asyncio.Semaphore(3)

        async def safe_run_juror(n, profile, p, d, js, jsc, cd):
            async with sem:
                return await run_single_juror(n, profile, p, d, js, jsc, cd)

        tasks = []
        for i, profile in enumerate(profiles):
            tasks.append(safe_run_juror(i + 1, profile, prosecution, defense, judge_summary, judge_scores, case_desc))
        verdicts = await asyncio.gather(*tasks)

    guilty_count = sum(1 for v in verdicts if v.get('verdict') in ["Guilty", "Liable"])
    not_guilty_count = sum(1 for v in verdicts if v.get('verdict') in ["Not Guilty", "Not Liable"])

    final_verdict = "Hung Jury"
    if guilty_count > not_guilty_count:
        final_verdict = "Guilty / Liable"
    elif not_guilty_count > guilty_count:
        final_verdict = "Not Guilty / Not Liable"

    analysis_prompt = f"Analyze these jury verdicts and explain which demographic segments voted which way and why based on their profiles: {json.dumps(verdicts)}"
    try:
        analysis_response = await safe_ainvoke(
            [("system", "You are a legal demographic analyst."), ("human", analysis_prompt)],
            temperature=0.3,
            model_names=["llama-3.1-8b-instant", "gemma2-9b-it"],
        )
        analysis_content = analysis_response.content
    except Exception as e:
        analysis_content = f"Demographic analysis unavailable due to API rate limits: {e}"

    return {
        "jury_verdicts": verdicts,
        "vote_count": {"guilty": guilty_count, "not_guilty": not_guilty_count},
        "final_verdict": final_verdict,
        "demographic_analysis": analysis_content
    }
