import functools
from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, END, START
from agents.prosecutor import prosecution_agent
from agents.defense import defense_agent
from agents.judge import judge_agent
from agents.jury_simulator import jury_simulator_node
from rag.retriever import retrieve_context
from tools.web_search import web_search

class CourtRoomState(TypedDict):
    case_description: str
    case_documents: list[str]
    retrieved_context: str
    web_search_results: str
    prosecution_argument: dict
    defense_argument: dict
    judge_scores: dict
    hallucination_flags: list
    jury_profiles: list[dict]
    jury_verdicts: list[dict]
    final_verdict: str
    round_number: int
    max_rounds: int
    debate_history: list[dict]
    vote_count: dict
    demographic_analysis: str

async def ingest_documents_node(state: CourtRoomState):
    # This node is a placeholder as ingestion usually happens in the UI or setup
    # But we ensure context is updated if any docs are in state
    return state

async def retrieve_context_node(state: CourtRoomState):
    context = retrieve_context(state['case_description'])
    return {"retrieved_context": context}

async def web_search_node(state: CourtRoomState):
    # Perform web search for the case
    search_results = web_search(state['case_description'])
    return {"web_search_results": search_results}

async def final_verdict_node(state: CourtRoomState):
    return state

async def prosecution_node(state: CourtRoomState):
    result = await prosecution_agent(state)
    return result

async def defense_node(state: CourtRoomState):
    result = await defense_agent(state)
    return result

async def judge_node(state: CourtRoomState):
    result = await judge_agent(state)
    return result

async def increment_round(state: CourtRoomState):
    # Store arguments in debate history
    history = state.get("debate_history", [])
    history.append({
        "round": state["round_number"],
        "prosecution": state["prosecution_argument"],
        "defense": state["defense_argument"]
    })
    return {"round_number": state['round_number'] + 1, "debate_history": history}

def should_continue(state: CourtRoomState):
    if state['round_number'] < state['max_rounds'] - 1:
        return "increment_round"
    return "jury_simulation"

def create_courtroom_graph():
    workflow = StateGraph(CourtRoomState)

    # Add nodes
    workflow.add_node("ingest_documents", ingest_documents_node)
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("prosecution_round", prosecution_node)
    workflow.add_node("defense_round", defense_node)
    workflow.add_node("judge_evaluation", judge_node)
    workflow.add_node("increment_round", increment_round)
    workflow.add_node("jury_simulation", jury_simulator_node)
    workflow.add_node("final_verdict_node", final_verdict_node)

    # Define edges
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
            "jury_simulation": "jury_simulation"
        }
    )
    
    workflow.add_edge("increment_round", "prosecution_round")
    workflow.add_edge("jury_simulation", "final_verdict_node")
    workflow.add_edge("final_verdict_node", END)

    return workflow.compile()
