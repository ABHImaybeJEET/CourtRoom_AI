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
    
    # Store usage in streamlit session state for UI tracking
    import streamlit as st
    if "api_stats" not in st.session_state:
        st.session_state.api_stats = {"total_calls": 0, "fallback_count": 0, "model_usage": {}}

    models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
        "llama3-8b-8192"
    ]
    
    last_err = None
    for i, model_name in enumerate(models):
        for attempt in range(2):
            try:
                llm = ChatGroq(model_name=model_name, temperature=temperature)
                await asyncio.sleep(random.uniform(0.1, 0.4))
                response = await llm.ainvoke(messages)
                
                # Record success stats
                st.session_state.api_stats["total_calls"] += 1
                if i > 0: st.session_state.api_stats["fallback_count"] += 1
                st.session_state.api_stats["model_usage"][model_name] = st.session_state.api_stats["model_usage"].get(model_name, 0) + 1
                
                # Tag the response content with the model name for transparency
                # We won't modify the text, but the agent can see it if needed
                return response
            except Exception as e:
                last_err = e
                if "429" in str(e):
                    wait_time = (attempt + 1) * 2 + random.uniform(1, 2)
                    await asyncio.sleep(wait_time)
                else:
                    break
                    
    raise last_err
