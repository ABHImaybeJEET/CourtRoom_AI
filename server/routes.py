import os
import json
import asyncio
from pathlib import Path
from typing import AsyncGenerator
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, status
from fastapi.responses import StreamingResponse, JSONResponse, Response
from server.trial_manager import manager
from server.transcript import build_trial_pdf

router = APIRouter(prefix="/api")

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB
MIN_CASE_DESC_LEN = 20
MAX_CASE_DESC_LEN = 100_000

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
    """Health status and configuration validation for model and search providers."""
    groq_configured = bool(os.getenv("GROQ_API_KEY", "").strip())
    tavily_configured = bool(os.getenv("TAVILY_API_KEY", "").strip())
    langsmith_configured = bool(os.getenv("LANGCHAIN_API_KEY", "").strip())

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
    """Retrieve list of pre-configured benchmark case briefs."""
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
    """Retrieve pre-recorded demo simulations available for instant playback."""
    return [
        {"id": "insider_trading.json", "name": "Insider Trading (Instant Demo)"},
        {"id": "india_fraud.json", "name": "Satyam Scam (Instant Demo)"},
    ]


@router.post("/cases/load-cached/{filename}")
async def load_cached_simulation(filename: str):
    """Load an instant benchmark trial playback from disk cache."""
    # Validate filename to prevent path traversal
    safe_name = Path(filename).name
    if not safe_name.endswith(".json"):
        raise HTTPException(status_code=400, detail="Invalid cache file format.")

    trial_data = manager.load_cached_trial(safe_name)
    if not trial_data:
        raise HTTPException(status_code=404, detail=f"Cached simulation '{safe_name}' not found.")
    return {"trial_id": trial_data["trial_id"], "state": trial_data}


@router.post("/trials", status_code=status.HTTP_201_CREATED)
async def create_trial_endpoint(
    background_tasks: BackgroundTasks,
    case_description: str = Form(...),
    max_rounds: int = Form(2),
    fast_mode: bool = Form(False),
    auto_start: bool = Form(True),
    files: list[UploadFile] = File(default=[]),
):
    """
    Create a new legal trial simulation from case text and optional uploaded PDF evidence.
    Performs comprehensive input validation and file checking.
    """
    clean_desc = case_description.strip()
    if len(clean_desc) < MIN_CASE_DESC_LEN:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Case description must contain at least {MIN_CASE_DESC_LEN} characters.",
        )
    if len(clean_desc) > MAX_CASE_DESC_LEN:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Case description exceeds maximum allowed length of {MAX_CASE_DESC_LEN:,} characters.",
        )

    if max_rounds < 1 or max_rounds > 5:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Argument rounds must be an integer between 1 and 5.",
        )

    # Validate uploaded files
    saved_paths: list[str] = []
    temp_id = f"prep-{Path(os.urandom(4).hex()).name}"
    upload_dir = Path(f"data/sessions/{temp_id}/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    for upload in files:
        if not upload.filename:
            continue
        safe_name = Path(upload.filename).name
        ext = Path(safe_name).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{safe_name}' has an unsupported extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        content = await upload.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{safe_name}' is empty (0 bytes).",
            )
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{safe_name}' ({len(content)/(1024*1024):.1f}MB) exceeds the 15MB size limit.",
            )

        target_file = upload_dir / safe_name
        target_file.write_bytes(content)
        saved_paths.append(str(target_file))

    trial_id = manager.create_trial(
        case_description=clean_desc,
        max_rounds=max_rounds,
        fast_mode=fast_mode,
        file_paths=saved_paths,
    )

    if auto_start:
        background_tasks.add_task(manager.run_trial_pipeline, trial_id)

    return {
        "trial_id": trial_id,
        "status": "running" if auto_start else "ready",
        "stream_url": f"/api/trials/{trial_id}/stream",
        "status_url": f"/api/trials/{trial_id}/status",
        "stages_url": f"/api/trials/{trial_id}/stages",
    }


@router.post("/trials/{trial_id}/start")
async def start_trial_run(trial_id: str, background_tasks: BackgroundTasks):
    """Explicitly start execution of a created trial."""
    trial = manager.get_trial(trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    if trial["status"] == "running":
        return {"trial_id": trial_id, "status": "running", "message": "Trial is already in progress."}

    if trial["status"] == "completed":
        raise HTTPException(status_code=400, detail="Trial has already concluded.")

    background_tasks.add_task(manager.run_trial_pipeline, trial_id)
    return {"trial_id": trial_id, "status": "running", "message": "Trial started successfully."}


@router.get("/trials/{trial_id}/status")
async def get_trial_monitoring_status(trial_id: str):
    """Retrieve lightweight status and telemetry for monitoring a trial run."""
    status_info = manager.get_trial_status(trial_id)
    if not status_info:
        raise HTTPException(status_code=404, detail="Trial not found")
    return status_info


@router.get("/trials/{trial_id}/stages")
async def get_trial_stage_outputs(trial_id: str):
    """Retrieve structured outputs across all trial stages."""
    stages_info = manager.get_trial_stages(trial_id)
    if not stages_info:
        raise HTTPException(status_code=404, detail="Trial not found")
    return stages_info


@router.get("/trials/{trial_id}")
async def get_full_trial(trial_id: str):
    """Retrieve complete trial record and state."""
    trial = manager.get_trial(trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")
    return trial


@router.get("/trials/{trial_id}/stream")
async def stream_trial_events(trial_id: str):
    """Server-Sent Events (SSE) stream delivering real-time deliberation events."""
    trial = manager.get_trial(trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    q = manager.register_queue(trial_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        current_status = trial["status"]
        yield f"event: ping\ndata: {json.dumps({'trial_id': trial_id, 'status': current_status, 'stage': trial['current_stage']})}\n\n"

        if current_status == "completed":
            final_state = trial["state"]
            yield f"event: trial_completed\ndata: {json.dumps({'stage': 'CONCLUDED', 'final_verdict': final_state.get('final_verdict'), 'state': final_state})}\n\n"
            manager.remove_queue(trial_id, q)
            return

        if current_status == "failed":
            yield f"event: error\ndata: {json.dumps({'stage': trial['current_stage'], 'message': trial['error']})}\n\n"
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
    """Download certified PDF trial transcript. Rejects incomplete or failed runs."""
    trial = manager.get_trial(trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    if trial["status"] == "failed":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot export transcript for a failed trial. Error: {trial.get('error')}",
        )

    if trial["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Trial is not yet concluded (current status: {trial['status']}).",
        )

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
    """Download full trial state JSON. Rejects incomplete or failed runs."""
    trial = manager.get_trial(trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail="Trial not found")

    if trial["status"] == "failed":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot export transcript for a failed trial. Error: {trial.get('error')}",
        )

    if trial["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Trial is not yet concluded (current status: {trial['status']}).",
        )

    filename = f"CourtRoom_AI_Transcript_{trial_id}.json"
    return JSONResponse(
        content=trial["state"],
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
