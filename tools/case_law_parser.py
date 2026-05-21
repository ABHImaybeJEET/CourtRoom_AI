import re


def extract_citations(text: str) -> list[str]:
    """Extract likely legal citations from text using lightweight regex patterns."""
    case_pattern = r"\b[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)* v\. [A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)*\b"
    statute_pattern = r"\b(?:\d+\s+U\.S\.C\.\s*§\s*\d+[A-Za-z0-9\-]*|\d+\s+[A-Z][A-Za-z\.\s0-9\-]*(?:\d+)?|SEC Rule [\dA-Za-z\-]+|§\s*\d+[A-Za-z0-9\-]*)\b"

    found_cases = re.findall(case_pattern, text)
    found_statutes = re.findall(statute_pattern, text)
    return sorted({citation.strip() for citation in found_cases + found_statutes})
