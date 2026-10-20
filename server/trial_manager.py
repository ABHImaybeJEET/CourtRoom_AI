import asyncio
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Any
from graph.courtroom_graph import create_courtroom_graph, CourtRoomState
from rag.ingestion import ingest_text, ingest_pdf
from agents.utils import get_langsmith_config

logger = logging.getLogger("courtroom.manager")


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
        now_iso = datetime.now(timezone.utc).isoformat()

        logger.info("Creating trial: id='%s', max_rounds=%d, fast_mode=%s", trial_id, max_rounds, fast_mode)

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
            "status": "ready",
        }

        self.trials[trial_id] = {
            "id": trial_id,
            "state": initial_state,
            "status": "ready",
            "current_stage": "INITIALIZATION",
            "created_at": now_iso,
            "started_at": None,
            "completed_at": None,
            "error": None,
        }
        self.event_queues[trial_id] = []
        return trial_id

    def get_trial(self, trial_id: str) -> dict[str, Any] | None:
        return self.trials.get(trial_id)

    def get_trial_status(self, trial_id: str) -> dict[str, Any] | None:
        trial = self.trials.get(trial_id)
        if not trial:
            return None
        state = trial["state"]
        return {
            "trial_id": trial_id,
            "status": trial["status"],
            "current_stage": trial["current_stage"],
            "round_number": state.get("round_number", 0) + 1,
            "max_rounds": state.get("max_rounds", 2),
            "created_at": trial["created_at"],
            "started_at": trial["started_at"],
            "completed_at": trial["completed_at"],
            "error": trial["error"],
            "has_verdict": bool(state.get("final_verdict")),
        }

    def get_trial_stages(self, trial_id: str) -> dict[str, Any] | None:
        trial = self.trials.get(trial_id)
        if not trial:
            return None
        state = trial["state"]
        return {
            "trial_id": trial_id,
            "status": trial["status"],
            "current_stage": trial["current_stage"],
            "intake": {
                "case_description": state.get("case_description", ""),
                "documents": state.get("case_documents", []),
                "max_rounds": state.get("max_rounds", 2),
                "fast_mode": state.get("fast_mode", False),
            },
            "evidence": {
                "retrieved_context": state.get("retrieved_context", ""),
                "web_search_results": state.get("web_search_results", ""),
            },
            "arguments": {
                "current_prosecution": state.get("prosecution_argument", {}),
                "current_defense": state.get("defense_argument", {}),
                "debate_history": state.get("debate_history", []),
            },
            "judicial_review": {
                "scores": state.get("judge_scores", {}),
                "hallucination_flags": state.get("hallucination_flags", []),
            },
            "jury": {
                "profiles": state.get("jury_profiles", []),
                "verdicts": state.get("jury_verdicts", []),
                "vote_count": state.get("vote_count", {"guilty": 0, "not_guilty": 0}),
                "demographic_analysis": state.get("demographic_analysis", ""),
            },
            "verdict": {
                "final_verdict": state.get("final_verdict", ""),
            },
            "error": trial["error"],
        }

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
        if trial_info["status"] == "running":
            return

        state = trial_info["state"]
        trial_info["status"] = "running"
        trial_info["started_at"] = datetime.now(timezone.utc).isoformat()
        trial_info["current_stage"] = "INITIALIZATION"

        logger.info("Executing trial pipeline for '%s'", trial_id)

        # Build clean LangSmith run config without sensitive data
        run_config = {
            "run_name": f"CourtRoom Trial [{trial_id}]",
            "tags": [
                "courtroom-ai",
                f"rounds-{state.get('max_rounds', 2)}",
                "fast-mode" if state.get("fast_mode") else "standard-mode",
            ],
            "metadata": {
                "trial_id": trial_id,
                "max_rounds": state.get("max_rounds", 2),
                "fast_mode": state.get("fast_mode", False),
                "has_uploaded_docs": bool(state.get("case_documents")),
            },
        }

        await self.broadcast(
            trial_id,
            "stage_change",
            {"stage": "INITIALIZATION", "message": "Trial session started. Graph compiled."},
        )

        try:
            graph = create_courtroom_graph()

            # Stream nodes from LangGraph with run_config attached for LangSmith
            async for output in graph.astream(state, config=run_config, stream_mode="updates"):
                for node_name, node_update in output.items():
                    state.update(node_update)

                    if node_name == "retrieve_context":
                        trial_info["current_stage"] = "EVIDENCE_RETRIEVAL"
                        await self.broadcast(
                            trial_id,
                            "context_retrieved",
                            {
                                "stage": "EVIDENCE_RETRIEVAL",
                                "context": state.get("retrieved_context", "")[:400] + "...",
                            },
                        )

                    elif node_name == "web_search_node":
                        trial_info["current_stage"] = "EVIDENCE_RETRIEVAL"
                        await self.broadcast(
                            trial_id,
                            "web_search_completed",
                            {
                                "stage": "CITATION_PREPARATION",
                                "message": "External case law precedent search completed.",
                            },
                        )

                    elif node_name == "prosecution_round":
                        trial_info["current_stage"] = "PROSECUTION_ARGUMENT"
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
                        trial_info["current_stage"] = "DEFENSE_ARGUMENT"
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
                        trial_info["current_stage"] = "JUDICIAL_REVIEW"
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
                        trial_info["current_stage"] = "ROUND_ADVANCEMENT"
                        await self.broadcast(
                            trial_id,
                            "round_advanced",
                            {
                                "stage": "ROUND_ADVANCEMENT",
                                "next_round": state.get("round_number", 0) + 1,
                            },
                        )

                    elif node_name == "jury_simulation":
                        trial_info["current_stage"] = "JURY_DELIBERATION"
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
            trial_info["current_stage"] = "CONCLUDED"
            trial_info["completed_at"] = datetime.now(timezone.utc).isoformat()
            logger.info("Trial pipeline completed successfully for '%s'", trial_id)

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
            trial_info["completed_at"] = datetime.now(timezone.utc).isoformat()
            logger.error("Trial pipeline failed for '%s': %s", trial_id, exc, exc_info=True)

            await self.broadcast(
                trial_id,
                "error",
                {
                    "stage": trial_info["current_stage"],
                    "message": str(exc),
                },
            )

    def load_cached_trial(self, filename: str) -> dict[str, Any] | None:
        try:
            with open(f"data/cached_results/{filename}", "r", encoding="utf-8") as f:
                data = json.load(f)
                trial_id = f"demo-{uuid.uuid4().hex[:8]}"
                data["trial_id"] = trial_id
                now_iso = datetime.now(timezone.utc).isoformat()
                self.trials[trial_id] = {
                    "id": trial_id,
                    "state": data,
                    "status": "completed",
                    "current_stage": "CONCLUDED",
                    "created_at": now_iso,
                    "started_at": now_iso,
                    "completed_at": now_iso,
                    "error": None,
                }
                return data
        except Exception:
            return None


manager = TrialManager()
