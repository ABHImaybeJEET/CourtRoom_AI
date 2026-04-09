import re

def extract_citations(text: str) -> list[str]:
    """
    Extracts case names and statutes using regex.
    Common legal citation patterns:
    - Case names: Usually Title v. Title or In re Name
    - Statutes: e.g., 18 U.S.C. § 1343, SEC Rule 10b-5
    """
    # Simple regex for Case Names (Party v. Party)
    case_pattern = r'[A-Z][a-z]+ v\. [A-Z][a-z]+'
    # Pattern for Statutes / Rules
    statute_pattern = r'\d+ [A-Z\.]+(?: \.?\d+)*|SEC Rule [\da-z\-]+|§ \d+[a-z]?'
    
    cases = re.findall(case_pattern, text)
    statutes = re.findall(statute_pattern, text)
    
    return list(set(cases + statutes))
