import pytest
from agents.utils import parse_json_from_llm
from tools.case_law_parser import extract_citations
from agents.jury_simulator import JUROR_PROFILES, run_single_juror


def test_parse_json_plain():
    raw = '{"key": "value", "score": 9.5}'
    parsed = parse_json_from_llm(raw)
    assert parsed == {"key": "value", "score": 9.5}


def test_parse_json_with_code_block():
    raw = """Here is the legal evaluation:
    ```json
    {
        "verdict": "Guilty",
        "confidence": 0.85
    }
    ```
    Hope this helps!"""
    parsed = parse_json_from_llm(raw)
    assert parsed["verdict"] == "Guilty"
    assert parsed["confidence"] == 0.85


def test_parse_json_with_surrounding_text():
    raw = 'Preliminary thoughts... {"prosecution_favored": true, "margin": 3} Final remarks.'
    parsed = parse_json_from_llm(raw)
    assert parsed["prosecution_favored"] is True
    assert parsed["margin"] == 3


def test_extract_citations():
    sample_text = (
        "Pursuant to 15 U.S.C. § 78j(b) and SEC Rule 10b-5, the defendant breached duty. "
        "See also Dirks v. SEC, 463 U.S. 646 (1983) and United States v. O'Hagan."
    )
    citations = extract_citations(sample_text)
    assert any("Dirks v. SEC" in c for c in citations)
    assert any("15 U.S.C." in c or "Rule 10b-5" in c for c in citations)


@pytest.mark.asyncio
async def test_juror_profile_integrity():
    assert len(JUROR_PROFILES) == 12
    for p in JUROR_PROFILES:
        assert "age" in p
        assert "occupation" in p
        assert "empathy_score" in p
