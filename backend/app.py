from fastapi import FastAPI
from pydantic import BaseModel
from backend.retrieval import get_top_chunks, graph_search
import os, time
from huggingface_hub import InferenceClient
from scripts.sf_connect import get_conn
import json
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

# Import your agent class from wherever it is saved (e.g., agent.py)
from agent import ResearchAgent

app = FastAPI(title="CS 5542 Research Assistant API")

print("=" * 60)
print(" 1. Initializing Research Agent...")
print("    (Please check terminal for Snowflake MFA prompt)")
print("=" * 60)

# Global instantiation: This blocks server startup to grab the OTP and connect
agent = ResearchAgent()

print("=" * 60)
print(" 2. Pre-fetching vector chunks into memory...")
print("=" * 60)
# Cache chunks once so all subsequent queries are fast
agent._prefetch_chunks()

print("=" * 60)
print(" ✅ Server is ready to accept requests!")
print("=" * 60)

app = FastAPI(title="Research Assistant API")
@app.get("/")
def read_root():
    return {"message": "Welcome to the Research Assistant API"}


def save_to_history(query_text: str, answer: str, citations: list):
    """
    Saves the query details to /backend/history.json.
    """
    # 1. Define path and ensure directory exists
    history_path = Path("backend/history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Create the new history entry
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query_text,
        "answer": answer,
        "chunks": citations  # citations contains the chunk metadata from your return
    }

    # 3. Load existing data or initialize a new list
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            try:
                history_data = json.load(f)
            except json.JSONDecodeError:
                history_data = []
    else:
        history_data = []

    # 4. Append and save
    history_data.append(new_entry)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=4, ensure_ascii=False)



class QueryRequest(BaseModel):
    question: str
    top_k: int = 5  # Maintained for frontend compatibility

@app.post('/query')
def query(req: QueryRequest):
    """
    Main endpoint called by the Streamlit frontend.
    Passes the user's question to the agent and formats the output.
    """
    # 1. Run the agentic loop
    agent_result = agent.run(req.question)
    
    # 2. Extract agent outputs
    answer = agent_result.get("answer", "No answer generated.")
    citations = agent_result.get("citations", [])
    latency_ms = agent_result.get("latency_ms", 0)

    print(f"\n\n\n\n\nCITATIONS :  :  {citations}")
    
    # 3. Calculate confidence based on the top vector match score
    confidence = round(citations[0].get("score", 0.0), 3) if citations else 0.0
    print(f"\n\n\n\n\nCONFIDENCE :  :  {confidence}")

    # 4. Map to the JSON structure expected by the frontend
    result = {
        'answer': answer,
        'citations': citations, 
        'confidence': confidence,
        'retrieval_mode': 'agentic', 
        'latency_ms': latency_ms,
        'tools_used': agent_result.get("tools_used", []),
        'steps_taken': agent_result.get("steps", 0)
    }

    return result

@app.post('/reset')
def reset_history():
    """
    Optional utility endpoint to clear the agent's conversation history
    without needing to restart the server and re-enter the OTP.
    """
    agent.reset_history()
    return {"status": "success", "message": "Agent history cleared."}
    

@app.get('/papers')
def papers():
    conn = get_conn(input("OPT : ").strip())
    cur = conn.cursor()
    
    # 1. Setup session context
    cur.execute('USE WAREHOUSE ROHAN_BLAKE_KENNETH_WH')
    cur.execute('USE DATABASE CS5542_PROJECT_ROHAN_BLAKE_KENNETH')

    query = f"""
    SELECT * FROM RAW.PAPERS;
    """
    cur.execute(query)
    rows = cur.fetchall()
    return rows

@app.get('/health')
def health(): return {'status': 'ok'}

