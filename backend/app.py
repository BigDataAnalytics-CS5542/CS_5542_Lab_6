from fastapi import FastAPI
from pydantic import BaseModel
from backend.retrieval import get_top_chunks, graph_search
import os, time
from huggingface_hub import InferenceClient
from scripts.sf_connect import get_conn
import json
from datetime import datetime
from pathlib import Path

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
    top_k: int = 5

@app.post('/query')
def query(req: QueryRequest):
    start = time.time()
    conn = get_conn(passcode=input("OTP : ").strip())
    chunks = get_top_chunks(conn=conn, query_text=req.question, top_k=req.top_k)
    # 1. Format each chunk to include its index, title, section, and text
    formatted_chunks = []
    for i, chunk in enumerate(chunks, start=1):
        title = chunk[3]
        section = chunk[4]
        text = chunk[5]
        
        # Bundle the metadata with the text using a clear structure
        formatted_string = f"[{i}] Paper Title: {title}\nSection: {section}\nText: {text}"
        formatted_chunks.append(formatted_string)

    # 2. Join the structured chunks with newlines to separate them clearly
    context = '\n\n'.join(formatted_chunks)
    print(context)
   
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    client = InferenceClient(token=os.getenv('HF_TOKEN'))
    messages = [{"role": "user", "content": f"Answer: {req.question}\nYou absolutely need to cite the context. You must cite the text excerpts using their bracketed index (e.g., \"This method is highly scalable [1].\" You don't need to list references at the end). Never invent facts, and never invent citations. Context: {context}"}]
    
    response = client.chat_completion(
        model=model_id,
        messages=messages,
        max_tokens=500
    )
    print(f"\n\n\n\n\n\n\n{response.choices[0].message.content}\n\n")
    answer =response.choices[0].message.content
    
    result = {
        'answer': answer,
        'citations': [
            {'chunk_id': c[1], 'paper_id': c[2], 'title': c[3],
             'section': c[4], 'text': c[5][:200], 'score': c[0]}
            for c in chunks
        ],
        'confidence': round(chunks[0][0], 3) if chunks else 0,
        'retrieval_mode': 'vector',
        'latency_ms': int((time.time() - start) * 1000)
        }

    save_to_history(
        query_text=req.question, 
        answer=result['answer'], 
        citations=result['citations']
    )

    return {
        result
    }

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

