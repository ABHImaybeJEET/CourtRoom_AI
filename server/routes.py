import os
import json
import asyncio
from pathlib import Path
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, Response
from server.trial_manager import manager
from server.transcript import build_trial_pdf

router = APIRouter(prefix="/api")

SAMPLE_CASES = {
    "insider_trading": {
        "id": "insider_trading",
        "title": "Insider Trading (Wall Street)",
        "subtitle": "48-hour pre-FDA disclosure liquidation",
        "file": "data/sample_cases/insider_trading.txt",
    },
    "india_fraud": {
        "id": "india_fraud",
        "title": "Satyam Scam (India Financial Fraud)",
        "subtitle": "Falsified fixed deposits and SEBI compliance breach",
        "file": "data/sample_cases/india_fraud.txt",
    },
    "corporate_fraud": {
        "id": "corporate_fraud",
        "title": "Corporate Fraud (Biotech Diagnostics)",
        "subtitle": "Defective proprietary blood testing device claims",
        "file": "data/sample_cases/corporate_fraud.txt",
    },
    "ip_infringement": {
        "id": "ip_infringement",
        "title": "IP Theft (Autonomous Vehicle LiDAR)",
        "subtitle": "Exfiltration of trade secret source code and schematics",
        "file": "data/sample_cases/ip_infringement.txt",
    },
}


@router.get("/health")
async def health_check():
    groq_configured = bool(os.getenv("GROQ_API_KEY"))
    tavily_configured = bool(os.getenv("TAVILY_API_KEY"))
    langsmith_configured = bool(os.getenv("LANGCHAIN_API_KEY"))

    return {
        "status": "online",
        "engine": "CourtRoom AI Core v2.0",
        "keys": {
            "groq": groq_configured,
            "tavily": tavily_configured,
            "langsmith": langsmith_configured,
        },
    }


@router.get("/cases/samples")
async def get_sample_cases():
    results = []
    for case_id, info in SAMPLE_CASES.items():
        content = ""
        path = Path(info["file"])
        if path.exists():
            content = path.read_text(encoding="utf-8")
        results.append({
            "id": case_id,
            "title": info["title"],
            "subtitle": info["subtitle"],
            "description": content,
        })
    return results


@router.get("/cases/cached")
async def get_cached_cases():
    cached = [
        {"id": "insider_trading.json", "name": "Insider Trading (Instant Demo)"},
        {"id": "india_fraud.json", "name": "Satyam Scam (Instant Demo)"},
    ]
    return cached


@router.post("/cases/load-cached/{filename}")
async def load_cached_simulation(filename: str):
    trial_data = manager.load_cached_trial(filename)
    if not trial_data:
        raise HTTPException(status_code=404, detail="Cached case not found")
    return {"trial_id": trial_data["trial_id"], "state": trial_data}


@router.post("/trials")
async def create_trial_endpoint(
    background_tasks: BackgroundTasks,
    case_description: str = Form(...),
    max_rounds: int = Form(2),
    fast_mode: bool = Form(False),
    files: list[UploadFile] = File(default=[]),
):
    saved_paths = []
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    for upload in files:
        if upload.filename:
            safe_name = Path(upload.filename).name
            target_file = upload_dir / safe_name
            content = await upload.read()
            target_file.write_bytes(content)
            saved_paths.append(str(target_file))

    trial_id = manager.create_trial(
        case_description=case_description,
        max_rounds=max_rounds,
        fast_mode=fast_mode,
        file_paths=saved_paths,
    )

    # Launch pipeline in background task
    background_tasks.add_task(manager.run_trial_pipeline, trial_id)

    return {
        "trial_id": trial_id,
        "status": "pending",
        "stream_url": f"/api/trials/{trial_id}/stream",
    }


@router.get("/trials/{trial_id}")
async def get_trial_state(trial_id: str):
    if trial_id not in manager.trials:
        raise HTTPException(status_code=404, detail="Trial not found")
    return manager.trials[trial_id]


@router.get("/trials/{trial_id}/stream")
async def stream_trial_events(trial_id: str):
    if trial_id not in manager.trials:
        raise HTTPException(status_code=404, detail="Trial not found")

    q = manager.register_queue(trial_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        # Send initial status
        current_status = manager.trials[trial_id]["status"]
        yield f"event: ping\ndata: {json.dumps({'trial_id': trial_id, 'status': current_status})}\n\n"

        # If already completed or failed, send final event immediately
        if current_status == "completed":
            final_state = manager.trials[trial_id]["state"]
            yield f"event: trial_completed\ndata: {json.dumps({'stage': 'CONCLUDED', 'final_verdict': final_state.get('final_verdict'), 'state': final_state})}\n\n"
            manager.remove_queue(trial_id, q)
            return

        try:
            while True:
                msg = await q.get()
                event_name = msg.get("event", "message")
                payload = json.dumps(msg.get("data", {}))
                yield f"event: {event_name}\ndata: {payload}\n\n"

                if event_name in ["trial_completed", "error"]:
                    break
        finally:
            manager.remove_queue(trial_id, q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/trials/{trial_id}/export/pdf")
async def export_trial_pdf(trial_id: str):
    if trial_id not in manager.trials:
        raise HTTPException(status_code=404, detail="Trial not found")

    trial = manager.trials[trial_id]
    state = trial["state"]
    pdf_buffer = build_trial_pdf(state)

    filename = f"CourtRoom_AI_Transcript_{trial_id}.pdf"
    return Response(
        content=pdf_buffer.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/trials/{trial_id}/export/json")
async def export_trial_json(trial_id: str):
    if trial_id not in manager.trials:
        raise HTTPException(status_code=404, detail="Trial not found")

    trial = manager.trials[trial_id]
    filename = f"CourtRoom_AI_Transcript_{trial_id}.json"
    return JSONResponse(
        content=trial["state"],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
