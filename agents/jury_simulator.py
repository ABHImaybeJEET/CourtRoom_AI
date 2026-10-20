import json
import asyncio
import logging
from agents.utils import parse_json_from_llm, safe_ainvoke, AgentExecutionError, traceable

logger = logging.getLogger("courtroom.jury")

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


@traceable(name="Single_Juror_Vote", run_type="chain")
async def run_single_juror(
    juror_id: int,
    profile: dict,
    prosecution: str,
    defense: str,
    judge_summary: str,
    judge_scores: dict,
    case_desc: str,
    trial_id: str = "anonymous",
) -> dict:
    prompt = f"""
    You are Juror #{juror_id}. Here is your individual profile:
    {json.dumps(profile)}

    Case Summary: {case_desc}
    Prosecution Argument: {prosecution}
    Defense Argument: {defense}
    Judge Assessment: {judge_summary}

    Deliberate according to your personality traits (empathy, skepticism, background).
    Provide strict JSON:
    {{
        "juror_id": {juror_id},
        "verdict": "Guilty" or "Not Guilty",
        "confidence": float between 0.0 and 1.0,
        "reasoning": "2-3 sentences explaining your vote based on your traits and evidence",
        "swayed_by": "prosecution" or "defense" or "neither"
    }}
    """
    messages = [
        ("system", "You are an individual juror in a criminal/civil trial simulation."),
        ("human", prompt.strip()),
    ]
    response = await safe_ainvoke(
        messages,
        temperature=0.7,
        model_names=["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        tracing_metadata={"trial_id": trial_id, "stage": "Single_Juror", "juror_id": juror_id},
    )
    parsed = parse_json_from_llm(response.content)
    parsed["juror_id"] = juror_id
    parsed["profile"] = profile
    if parsed.get("verdict") not in ["Guilty", "Not Guilty", "Liable", "Not Liable"]:
        parsed["verdict"] = "Not Guilty"
    return parsed


@traceable(name="Jury_Panel_Deliberation", run_type="chain")
async def run_jury_panel(prosecution, defense, judge_summary, judge_scores, case_desc, profiles, trial_id: str = "anonymous"):
    system_prompt = f"""
    You are simulating the collective jury deliberation for a panel of {len(profiles)} jurors.
    Review the case description, the prosecution's final argument, the defense's final argument, and the judge's scoring summary.
    For each juror in the provided profile list, determine their vote based on their profile traits (empathy score, skepticism, occupation, political lean).

    Return a JSON object with one key: "jury_verdicts" containing an array of juror objects:
    - juror_id: int (1 to {len(profiles)})
    - verdict: "Guilty" or "Not Guilty"
    - confidence: float between 0.0 and 1.0
    - reasoning: 2-3 sentences relating the verdict directly back to their background and traits
    - swayed_by: "prosecution" | "defense" | "neither"
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
        ("system", system_prompt.strip()),
        ("human", human_prompt.strip()),
    ]

    response = await safe_ainvoke(
        messages,
        temperature=0.7,
        model_names=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        tracing_metadata={"trial_id": trial_id, "stage": "Jury_Panel", "pool_size": len(profiles)},
    )

    result = parse_json_from_llm(response.content)
    verdicts = None
    if isinstance(result, dict) and "jury_verdicts" in result:
        verdicts = result["jury_verdicts"]
    elif isinstance(result, list):
        verdicts = result
    elif isinstance(result, dict) and "items" in result:
        verdicts = result["items"]

    if not verdicts or not isinstance(verdicts, list):
        raise ValueError("Invalid jury panel response format")

    for i, juror in enumerate(verdicts):
        if not isinstance(juror, dict):
            continue
        juror_idx = juror.get("juror_id", i + 1) - 1
        if 0 <= juror_idx < len(profiles):
            juror["profile"] = profiles[juror_idx]
        else:
            juror["profile"] = profiles[min(i, len(profiles) - 1)]
        if juror.get("verdict") not in ["Guilty", "Not Guilty", "Liable", "Not Liable"]:
            juror["verdict"] = "Not Guilty"

    return verdicts


@traceable(name="Jury_Simulation_Stage", run_type="chain")
async def jury_simulator_node(state: dict):
    trial_id = state.get("trial_id", "anonymous")
    prosecution = state.get("prosecution_argument", {}).get("argument_text", "")
    defense = state.get("defense_argument", {}).get("argument_text", "")
    judge_summary = state.get("judge_scores", {}).get("reasoning_summary", "")
    judge_scores = state.get("judge_scores", {})
    case_desc = state.get("case_description", "")
    profiles = JUROR_PROFILES[:6] if state.get("fast_mode") else JUROR_PROFILES

    logger.info("Jury deliberation starting: pool_size=%d, trial_id='%s'", len(profiles), trial_id)

    try:
        verdicts = await run_jury_panel(prosecution, defense, judge_summary, judge_scores, case_desc, profiles, trial_id=trial_id)
    except Exception as exc:
        logger.warning("Jury panel full-batch failed (%s), running individual juror deliberation...", exc)
        sem = asyncio.Semaphore(3)

        async def safe_run(n, profile):
            async with sem:
                return await run_single_juror(n, profile, prosecution, defense, judge_summary, judge_scores, case_desc, trial_id=trial_id)

        try:
            tasks = [safe_run(i + 1, profile) for i, profile in enumerate(profiles)]
            verdicts = await asyncio.gather(*tasks)
        except Exception as fallback_exc:
            logger.error("Jury simulation deliberation completely failed: %s", fallback_exc)
            raise AgentExecutionError(f"Jury simulation deliberation completely failed: {fallback_exc}") from fallback_exc

    # Ensure profile is present in each verdict
    for i, v in enumerate(verdicts):
        if "profile" not in v and i < len(profiles):
            v["profile"] = profiles[i]

    guilty_count = sum(1 for v in verdicts if v.get("verdict") in ["Guilty", "Liable"])
    not_guilty_count = sum(1 for v in verdicts if v.get("verdict") in ["Not Guilty", "Not Liable"])

    if guilty_count > not_guilty_count:
        final_verdict = "Guilty"
    elif not_guilty_count > guilty_count:
        final_verdict = "Not Guilty"
    else:
        final_verdict = "Hung Jury"

    logger.info("Jury vote tally: %d Guilty, %d Not Guilty -> Verdict: %s", guilty_count, not_guilty_count, final_verdict)

    analysis_prompt = (
        f"Analyze these jury verdicts and explain which demographic segments voted which way and why based on their profiles: "
        f"{json.dumps(verdicts)}"
    )
    try:
        analysis_response = await safe_ainvoke(
            [("system", "You are an expert legal demographic analyst and trial consultant."), ("human", analysis_prompt)],
            temperature=0.3,
            model_names=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
            tracing_metadata={"trial_id": trial_id, "stage": "Demographic_Analysis"},
        )
        analysis_content = analysis_response.content
    except Exception as e:
        logger.warning("Demographic explanation call failed: %s", e)
        analysis_content = (
            f"Deliberation concluded with {guilty_count} Guilty votes and {not_guilty_count} Not Guilty votes. "
            f"Demographic explanation could not be generated: {e}"
        )

    return {
        "jury_profiles": profiles,
        "jury_verdicts": verdicts,
        "vote_count": {"guilty": guilty_count, "not_guilty": not_guilty_count},
        "final_verdict": final_verdict,
        "demographic_analysis": analysis_content,
    }
