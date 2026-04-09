import re
import json

def parse_json_from_llm(content: str) -> dict:
    """Safely extracts JSON from LLM response which might have markdown formatting."""
    content = content.strip()
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
        
    # Attempt to extract from markdown codeblock
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
            
    # Attempt to extract anything that looks like a JSON object / array
    match = re.search(r'(\{.*\})', content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
            
    # Return empty dict if all fails, which triggers specific fallback logic in agents
    raise ValueError("Could not parse JSON from response")

async def safe_ainvoke(messages, temperature=0.2):
    from langchain_groq import ChatGroq
    try:
        # We try the primary heavy model first
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=temperature)
        return await llm.ainvoke(messages)
    except Exception as e:
        print(f"Primary model failed (Error: {e}), falling back to llama-3.1-8b-instant...")
        try:
            # First fallback: llama-3.1-8b
            llm_fallback = ChatGroq(model_name="llama-3.1-8b-instant", temperature=temperature)
            return await llm_fallback.ainvoke(messages)
        except Exception as e2:
            print(f"Secondary model failed (Error: {e2}), falling back to fastest model miensrtal-sagemath...")
            # Last resort fallback: extremely lightweight
            llm_last = ChatGroq(model_name="mixtral-8x7b-32768", temperature=temperature)
            return await llm_last.ainvoke(messages)
