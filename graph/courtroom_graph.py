from typing import TypedDict, Any
from langgraph.graph import StateGraph, END, START
from agents.prosecutor import prosecution_agent
from agents.defense import defense_agent
from agents.judge import judge_agent
from agents.jury_simulator import jury_simulator_node
from rag.retriever import retrieve_context
from tools.web_search import web_search


class CourtRoomState(TypedDict, total=False):
    trial_id: str
    case_description: str
    case_documents: list[str]
    retrieved_context: str
    web_search_results: str
    prosecution_argument: dict[str, Any]
    defense_argument: dict[str, Any]
    judge_scores: dict[str, Any]
    hallucination_flags: list[dict[str, Any]]
    jury_profiles: list[dict[str, Any]]
    jury_verdicts: list[dict[str, Any]]
    final_verdict: str
    round_number: int
    max_rounds: int
    debate_history: list[dict[str, Any]]
    vote_count: dict[str, int]
    demographic_analysis: str
    fast_mode: bool
    status: str


async def ingest_documents_node(state: CourtRoomState) -> dict:
    return {"status": "documents_ingested"}


async def retrieve_context_node(state: CourtRoomState) -> dict:
    session_id = state.get("trial_id")
    case_desc = state.get("case_description", "")
    context = retrieve_context(case_desc, k=5, session_id=session_id)
    return {"retrieved_context": context, "status": "context_retrieved"}


async def web_search_node(state: CourtRoomState) -> dict:
    case_desc = state.get("case_description", "")
    # Use key legal query line or first 200 chars for a cleaner search
    query_line = case_desc.split("\n")[0].strip() if case_desc else ""
    if len(query_line) > 200:
        query_line = query_line[:200]
    
    search_query = f"{query_line} legal case law precedent" if query_line else "legal precedent case law"
    search_results = web_search(search_query)
    return {"web_search_results": search_results, "status": "web_search_completed"}


async def prosecution_node(state: CourtRoomState) -> dict:
    result = await prosecution_agent(dict(state))
    result["status"] = "prosecution_completed"
    return result


async def defense_node(state: CourtRoomState) -> dict:
    result = await defense_agent(dict(state))
    result["status"] = "defense_completed"
    return result


async def judge_node(state: CourtRoomState) -> dict:
    result = await judge_agent(dict(state))
    result["status"] = "judge_completed"
    return result


async def increment_round(state: CourtRoomState) -> dict:
    history = list(state.get("debate_history", []))
    history.append({
        "round": state.get("round_number", 0) + 1,
        "prosecution": state.get("prosecution_argument", {}),
        "defense": state.get("defense_argument", {}),
        "judge_scores": state.get("judge_scores", {}),
    })
    return {
        "round_number": state.get("round_number", 0) + 1,
        "debate_history": history,
        "status": f"round_{state.get('round_number', 0) + 1}_completed",
    }


def should_continue(state: CourtRoomState) -> str:
    current_round = state.get("round_number", 0)
    max_rounds = state.get("max_rounds", 2)
    if current_round < max_rounds - 1:
        return "increment_round"
    return "jury_simulation"


async def jury_node(state: CourtRoomState) -> dict:
    result = await jury_simulator_node(dict(state))
    result["status"] = "jury_deliberation_completed"
    return result


async def final_verdict_node(state: CourtRoomState) -> dict:
    return {"status": "trial_concluded"}


def create_courtroom_graph():
    workflow = StateGraph(CourtRoomState)

    # Nodes
    workflow.add_node("ingest_documents", ingest_documents_node)
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("prosecution_round", prosecution_node)
    workflow.add_node("defense_round", defense_node)
    workflow.add_node("judge_evaluation", judge_node)
    workflow.add_node("increment_round", increment_round)
    workflow.add_node("jury_simulation", jury_node)
    workflow.add_node("final_verdict_node", final_verdict_node)

    # Edges
    workflow.add_edge(START, "ingest_documents")
    workflow.add_edge("ingest_documents", "retrieve_context")
    workflow.add_edge("retrieve_context", "web_search_node")
    workflow.add_edge("web_search_node", "prosecution_round")
    workflow.add_edge("prosecution_round", "defense_round")
    workflow.add_edge("defense_round", "judge_evaluation")

    workflow.add_conditional_edges(
        "judge_evaluation",
        should_continue,
        {
            "increment_round": "increment_round",
            "jury_simulation": "jury_simulation",
        },
    )

    workflow.add_edge("increment_round", "prosecution_round")
    workflow.add_edge("jury_simulation", "final_verdict_node")
    workflow.add_edge("final_verdict_node", END)

    return workflow.compile()
