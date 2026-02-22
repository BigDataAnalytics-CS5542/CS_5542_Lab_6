from fastapi import FastAPI
from typing import List, Dict, Any
from pydantic import BaseModel
from backend.rag_pipeline import run_hybrid_query
from backend.json_to_csv import process_json_to_csv
import json

app = FastAPI(title="CS 5542 Demo API")


class EchoRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"status": "ok", "message": "API working"}


@app.post("/echo")
def echo(req: EchoRequest):
    return {"echo": req.text}