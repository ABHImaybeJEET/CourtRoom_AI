import re
import json
import asyncio

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
    import random
    import time
    
    # Ordered list of models to try from heavy to light/reliable
    models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "llama3-8b-8192"
    ]
    
    last_err = None
    
    # Try each model in sequence
    for model_name in models:
        # Retry each model up to 2 times with backoff if rate limited
        for attempt in range(2):
            try:
                llm = ChatGroq(model_name=model_name, temperature=temperature)
                # Add a tiny jitter to avoid hitting same-second limits
                await asyncio.sleep(random.uniform(0.1, 0.5))
                return await llm.ainvoke(messages)
            except Exception as e:
                last_err = e
                # If rate limited (429), sleep longer
                if "429" in str(e):
                    wait_time = (attempt + 1) * 2 + random.uniform(1, 3)
                    print(f"Rate limited on {model_name}. Waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    # Generic error (500 etc), skip to next model immediately
                    print(f"Error on {model_name}: {e}. Trying fallback...")
                    break
                    
    # If all models fail, raise the last encountered error
    raise last_err
