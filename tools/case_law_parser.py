import re


def extract_citations(text: str) -> list[str]:
    """Extract likely legal citations (case names, statutes, regulations) from text."""
    if not text:
        return []

    # Matches "Party A v. Party B" or "In re Party"
    case_pattern = r"\b(?:In re\s+[A-Z][A-Za-z0-9\s,\.]+|[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+v\.\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)(?:,\s*\d+\s+U\.S\.\s+\d+)?\b"
    
    # Matches statutory and regulatory citations like "17 C.F.R. § 240.10b-5", "15 U.S.C. § 78j(b)", "Fed. R. Evid. 403"
    statute_pattern = (
        r"\b(?:\d+\s+U\.S\.C\.\s*(?:§\s*\d+[A-Za-z0-9\-\(\)]*)|"
        r"\d+\s+C\.F\.R\.\s*(?:§\s*\d+[\.\d\w\-]*)|"
        r"SEC Rule\s+[\dA-Za-z\-]+|"
        r"Rule\s+10b-5|"
        r"Fed\.\s*R\.\s*(?:Evid|Civ|Crim)\.\s*(?:P\.\s*)?\d+|"
        r"§\s*\d+[A-Za-z0-9\-]*)\b"
    )

    found_cases = re.findall(case_pattern, text)
    found_statutes = re.findall(statute_pattern, text)
    
    combined = {c.strip().strip(",") for c in (found_cases + found_statutes) if len(c.strip()) > 3}
    return sorted(combined)
