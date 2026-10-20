import asyncio
import json
import uuid
from typing import AsyncGenerator, Any
from graph.courtroom_graph import create_courtroom_graph, CourtRoomState
from rag.ingestion import ingest_text, ingest_pdf


class TrialManager:
    def __init__(self):
        self.trials: dict[str, dict[str, Any]] = {}
        self.event_queues: dict[str, list[asyncio.Queue]] = {}

    def create_trial(
        self,
        case_description: str,
        max_rounds: int = 2,
        fast_mode: bool = False,
        file_paths: list[str] | None = None,
    ) -> str:
        trial_id = f"trial-{uuid.uuid4().hex[:10]}"
        
        # Ingest text or documents under this trial's isolated session
        if file_paths:
            for p in file_paths:
                ingest_pdf(p, session_id=trial_id)
        elif case_description:
            ingest_text(case_description, session_id=trial_id)

        initial_state: CourtRoomState = {
            "trial_id": trial_id,
            "case_description": case_description,
            "case_documents": file_paths or [],
            "retrieved_context": "",
            "web_search_results": "",
            "prosecution_argument": {},
            "defense_argument": {},
            "judge_scores": {},
            "hallucination_flags": [],
            "jury_profiles": [],
            "jury_verdicts": [],
            "final_verdict": "",
            "round_number": 0,
            "max_rounds": max_rounds,
            "debate_history": [],
            "vote_count": {"guilty": 0, "not_guilty": 0},
            "demographic_analysis": "",
            "fast_mode": fast_mode,
            "status": "initialized",
        }

        self.trials[trial_id] = {
            "id": trial_id,
            "state": initial_state,
            "status": "pending",
            "error": None,
        }
        self.event_queues[trial_id] = []
        return trial_id

    def register_queue(self, trial_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        if trial_id not in self.event_queues:
            self.event_queues[trial_id] = []
        self.event_queues[trial_id].append(q)
        return q

    def remove_queue(self, trial_id: str, q: asyncio.Queue) -> None:
        if trial_id in self.event_queues and q in self.event_queues[trial_id]:
            self.event_queues[trial_id].remove(q)

    async def broadcast(self, trial_id: str, event_type: str, data: dict[str, Any]) -> None:
        if trial_id not in self.event_queues:
            return
        msg = {"event": event_type, "data": data}
        for q in list(self.event_queues[trial_id]):
            await q.put(msg)

    async def run_trial_pipeline(self, trial_id: str) -> None:
        if trial_id not in self.trials:
            return

        trial_info = self.trials[trial_id]
        state = trial_info["state"]
        trial_info["status"] = "running"

        await self.broadcast(
            trial_id,
            "stage_change",
            {"stage": "INITIALIZATION", "message": "Trial session started. Graph compiled."},
        )

        try:
            graph = create_courtroom_graph()

            # Stream nodes from LangGraph
            async for output in graph.astream(state, stream_mode="updates"):
                for node_name, node_update in output.items():
                    # Update local state
                    state.update(node_update)

                    if node_name == "retrieve_context":
                        await self.broadcast(
                            trial_id,
                            "context_retrieved",
                            {
                                "stage": "EVIDENCE_RETRIEVAL",
                                "context": state.get("retrieved_context", "")[:400] + "...",
                            },
                        )

                    elif node_name == "web_search_node":
                        await self.broadcast(
                            trial_id,
                            "web_search_completed",
                            {
                                "stage": "CITATION_PREPARATION",
                                "message": "External case law precedent search completed.",
                            },
                        )

                    elif node_name == "prosecution_round":
                        await self.broadcast(
                            trial_id,
                            "prosecution_argument",
                            {
                                "stage": "PROSECUTION_ARGUMENT",
                                "argument": state.get("prosecution_argument", {}),
                                "round_number": state.get("round_number", 0) + 1,
                            },
                        )

                    elif node_name == "defense_round":
                        await self.broadcast(
                            trial_id,
                            "defense_argument",
                            {
                                "stage": "DEFENSE_ARGUMENT",
                                "argument": state.get("defense_argument", {}),
                                "round_number": state.get("round_number", 0) + 1,
                            },
                        )

                    elif node_name == "judge_evaluation":
                        await self.broadcast(
                            trial_id,
                            "judge_evaluation",
                            {
                                "stage": "JUDICIAL_REVIEW",
                                "scores": state.get("judge_scores", {}),
                                "hallucination_flags": state.get("hallucination_flags", []),
                            },
                        )

                    elif node_name == "increment_round":
                        await self.broadcast(
                            trial_id,
                            "round_advanced",
                            {
                                "stage": "ROUND_ADVANCEMENT",
                                "next_round": state.get("round_number", 0) + 1,
                            },
                        )

                    elif node_name == "jury_simulation":
                        await self.broadcast(
                            trial_id,
                            "jury_deliberation",
                            {
                                "stage": "JURY_DELIBERATION",
                                "jury_verdicts": state.get("jury_verdicts", []),
                                "vote_count": state.get("vote_count", {}),
                                "demographic_analysis": state.get("demographic_analysis", ""),
                            },
                        )

            trial_info["status"] = "completed"
            await self.broadcast(
                trial_id,
                "trial_completed",
                {
                    "stage": "CONCLUDED",
                    "final_verdict": state.get("final_verdict", "N/A"),
                    "state": state,
                },
            )

        except Exception as exc:
            trial_info["status"] = "failed"
            trial_info["error"] = str(exc)
            await self.broadcast(
                trial_id,
                "error",
                {"message": f"Simulation interrupted: {str(exc)}"},
            )

    def load_cached_trial(self, filename: str) -> dict[str, Any] | None:
        try:
            with open(f"data/cached_results/{filename}", "r", encoding="utf-8") as f:
                data = json.load(f)
                trial_id = f"demo-{uuid.uuid4().hex[:8]}"
                data["trial_id"] = trial_id
                self.trials[trial_id] = {
                    "id": trial_id,
                    "state": data,
                    "status": "completed",
                    "error": None,
                }
                return data
        except Exception:
            return None


manager = TrialManager()
