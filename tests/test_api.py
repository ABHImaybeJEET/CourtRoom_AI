import io
import pytest
from fastapi.testclient import TestClient
from server.app import app
from agents.utils import safe_ainvoke, ModelProviderError

client = TestClient(app)


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "keys" in data
    assert "groq" in data["keys"]
    assert "tavily" in data["keys"]
    # Secrets must not be leaked
    assert "GROQ_API_KEY" not in str(data)


def test_frontend_serving():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "") or "application/json" in response.headers.get("content-type", "")


def test_api_sample_cases():
    response = client.get("/api/cases/samples")
    assert response.status_code == 200
    samples = response.json()
    assert len(samples) >= 4
    case_ids = [s["id"] for s in samples]
    assert "insider_trading" in case_ids
    assert "india_fraud" in case_ids


def test_api_cached_cases_and_export():
    # 1. Load cached simulation
    load_resp = client.post("/api/cases/load-cached/insider_trading.json")
    assert load_resp.status_code == 200
    data = load_resp.json()
    trial_id = data["trial_id"]
    assert trial_id is not None

    # 2. Query trial state & stages
    state_resp = client.get(f"/api/trials/{trial_id}")
    assert state_resp.status_code == 200
    trial_state = state_resp.json()
    assert trial_state["status"] == "completed"
    assert trial_state["state"]["final_verdict"] == "Guilty"

    status_resp = client.get(f"/api/trials/{trial_id}/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["has_verdict"] is True

    stages_resp = client.get(f"/api/trials/{trial_id}/stages")
    assert stages_resp.status_code == 200
    stages_data = stages_resp.json()
    assert "intake" in stages_data
    assert "evidence" in stages_data
    assert "arguments" in stages_data
    assert "jury" in stages_data

    # 3. Test PDF export
    pdf_resp = client.get(f"/api/trials/{trial_id}/export/pdf")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["content-type"] == "application/pdf"
    assert len(pdf_resp.content) > 1000

    # 4. Test JSON export
    json_resp = client.get(f"/api/trials/{trial_id}/export/json")
    assert json_resp.status_code == 200
    assert json_resp.json()["final_verdict"] == "Guilty"


# --- CRITICAL ERROR PATH TESTS ---

def test_validation_rejects_short_description():
    response = client.post(
        "/api/trials",
        data={"case_description": "Too short", "max_rounds": "2"},
    )
    assert response.status_code == 422
    assert "at least 20 characters" in response.json()["detail"]


def test_validation_rejects_invalid_rounds():
    response = client.post(
        "/api/trials",
        data={
            "case_description": "Valid case description detailing securities fraud and insider trading breach.",
            "max_rounds": "10",
        },
    )
    assert response.status_code == 422
    assert "between 1 and 5" in response.json()["detail"]


def test_validation_rejects_unsupported_file_extension():
    fake_file = io.BytesIO(b"malicious executable or script")
    response = client.post(
        "/api/trials",
        data={
            "case_description": "Valid case description detailing corporate fraud and false representations.",
            "max_rounds": "2",
        },
        files={"files": ("exploit.exe", fake_file, "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "unsupported extension" in response.json()["detail"]


def test_validation_rejects_empty_uploaded_file():
    empty_file = io.BytesIO(b"")
    response = client.post(
        "/api/trials",
        data={
            "case_description": "Valid case description detailing corporate fraud and false representations.",
            "max_rounds": "2",
        },
        files={"files": ("empty_evidence.pdf", empty_file, "application/pdf")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_trial_lifecycle_manual_start():
    # Create with auto_start = False
    create_resp = client.post(
        "/api/trials",
        data={
            "case_description": "The defendant is charged with grand larceny and embezzling institutional funds.",
            "max_rounds": "2",
            "auto_start": "false",
        },
    )
    assert create_resp.status_code == 201
    trial_id = create_resp.json()["trial_id"]
    assert create_resp.json()["status"] == "ready"

    # Verify status is ready
    status_resp = client.get(f"/api/trials/{trial_id}/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "ready"

    # Start trial explicitly
    start_resp = client.post(f"/api/trials/{trial_id}/start")
    assert start_resp.status_code == 200
    assert start_resp.json()["status"] == "running"


def test_export_rejects_incomplete_trial():
    # Create trial without completing it
    create_resp = client.post(
        "/api/trials",
        data={
            "case_description": "The defendant is charged with theft of trade secrets regarding autonomous vehicles.",
            "max_rounds": "2",
            "auto_start": "false",
        },
    )
    trial_id = create_resp.json()["trial_id"]

    # Calling PDF export on an uncompleted trial must fail cleanly with 400
    pdf_resp = client.get(f"/api/trials/{trial_id}/export/pdf")
    assert pdf_resp.status_code == 400
    assert "not yet concluded" in pdf_resp.json()["detail"]

    # Calling JSON export on an uncompleted trial must fail cleanly with 400
    json_resp = client.get(f"/api/trials/{trial_id}/export/json")
    assert json_resp.status_code == 400
    assert "not yet concluded" in json_resp.json()["detail"]


@pytest.mark.asyncio
async def test_safe_ainvoke_fails_loudly_when_key_missing(monkeypatch):
    # Ensure missing GROQ_API_KEY does not fake success
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(ModelProviderError) as exc_info:
        await safe_ainvoke([("human", "Test prompt")])
    assert "GROQ_API_KEY is not configured" in str(exc_info.value)
