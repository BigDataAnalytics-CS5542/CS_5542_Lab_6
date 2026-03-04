"""
AI Agent for the CS 5542 Research Assistant.
"""

from __future__ import annotations
import json
import os
import time
from typing import Any
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from scripts.sf_connect import get_conn
from tool_schemas import TOOL_SCHEMAS, TOOL_FUNCTIONS

load_dotenv()

SYSTEM_PROMPT = """You are a Research Assistant agent specialized in scientific literature.

You MUST follow these steps in order for EVERY question:
STEP 1: Call search_papers with the user's question.
STEP 2: Call summarize_context with the question and the chunks from Step 1.
STEP 3: Return the answer from summarize_context. Do not add anything else.

You also have:
- get_paper_details: use if you need more info about a specific paper_id
- search_knowledge_graph: use if the question is about relationships between concepts

Never skip summarize_context. Never answer from your own knowledge.
"""

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MAX_ITERATIONS = 6

class ResearchAgent:
    def __init__(self, passcode: str = ""):
        print("[Agent] Connecting to Snowflake...")
        if not passcode:
            passcode = input("Enter Snowflake MFA code: ").strip()
        self.conn = get_conn(passcode=passcode)
        print("[Agent] Connected.")
        self.client = InferenceClient(token=os.getenv("HF_TOKEN"))
        self.history: list[dict] = []

        # Cache chunks to avoid re-fetching every query
        self._chunk_cache: list | None = None
        self._last_citations: list = []

    def _prefetch_chunks(self):
        """Fetch all chunks once and cache them in memory."""
        if self._chunk_cache is not None:
            return
        print("[Agent] Pre-fetching chunks from Snowflake (one-time)...")
        import numpy as np
        cur = self.conn.cursor()
        cur.execute(f"USE WAREHOUSE {os.getenv('SNOWFLAKE_WAREHOUSE')}")
        cur.execute(f"USE DATABASE {os.getenv('SNOWFLAKE_DATABASE')}")
        cur.execute(
            "SELECT CHUNK_ID, PAPER_ID, TITLE, SECTION_NAME, TEXT_CONTENT, EMBEDDING "
            "FROM APP.CHUNKS_V"
        )
        rows = cur.fetchall()
        self._chunk_cache = rows
        print(f"[Agent] Cached {len(rows)} chunks.")

    def _call_tool(self, tool_name: str, tool_args: dict) -> Any:
        fn = TOOL_FUNCTIONS.get(tool_name)
        if fn is None:
            return {"error": f"Unknown tool: {tool_name}"}

        print(f"  [Tool] Calling {tool_name}({list(tool_args.keys())})")

        # search_papers uses cache for speed
        if tool_name == "search_papers":
            return self._fast_search(
                query=tool_args.get("query", ""),
                top_k=tool_args.get("top_k", 5)
            )

        if tool_name == "summarize_context":
            tool_args = {
                "question": tool_args.get("question") or tool_args.get("q", ""),
                "chunks":   tool_args.get("chunks")   or tool_args.get("c", []),
            }
            if not tool_args["chunks"] and self._last_citations:
                tool_args["chunks"] = self._last_citations
            return fn(**tool_args)

        snowflake_tools = {"get_paper_details", "search_knowledge_graph"}
        if tool_name in snowflake_tools:
            return fn(conn=self.conn, **tool_args)
        else:
            return fn(**tool_args)

    def _fast_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Vector search using cached chunks — much faster than re-fetching."""
        import numpy as np
        import json as _json
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        query_vec = model.encode([query], normalize_embeddings=True)[0]

        self._prefetch_chunks()

        results = []
        for chunk_id, paper_id, title, section, text, emb_json in self._chunk_cache:
            try:
                emb = np.array(_json.loads(emb_json))
                score = float(np.dot(query_vec, emb))
                results.append({
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "title":    title,
                    "section":  section,
                    "text":     text,
                    "score":    round(score, 4),
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["score"], reverse=True)
        top_k = int(top_k)
        top = results[:top_k]
        print(f"  [Tool] search_papers → top score: {top[0]['score'] if top else 'none'}")
        return top

    def run(self, question: str) -> dict:
        start = time.time()
        print(f"\n[Agent] Question: {question}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history,
            {"role": "user", "content": question},
        ]

        tools_used = []
        citations  = []
        answer     = ""
        steps      = 0

        for iteration in range(MAX_ITERATIONS):
            steps += 1
            print(f"\n[Agent] Iteration {steps}...")

            response = self.client.chat_completion(
                model=MODEL_ID,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                max_tokens=1000,
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append({
                    "role":       "assistant",
                    "content":    msg.content or "",
                    "tool_calls": [
                        {
                            "id":       tc.id,
                            "type":     "function",
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)
                    tools_used.append(tool_name)
                    result = self._call_tool(tool_name, tool_args)

                    if tool_name == "search_papers" and isinstance(result, list):
                        for chunk in result:
                            if "error" not in chunk:
                                citations.append(chunk)
                            self._last_citations = citations

                    if tool_name == "search_knowledge_graph" and isinstance(result, list):
                        if not citations:
                            extra = self._fast_search(query=question, top_k=5)
                            citations.extend([c for c in extra if "error" not in c])

                    if tool_name == "summarize_context" and isinstance(result, str):
                        answer = result

                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      json.dumps(result, default=str),
                    })

            else:
                if msg.content:
                    answer = msg.content
                break

        if citations:
            print("\n[Agent] Calling summarize_context with retrieved chunks...")
            if "summarize_context" not in tools_used:
                tools_used.append("summarize_context")
            answer = TOOL_FUNCTIONS["summarize_context"](
                question=question,
                chunks=citations[:5]
            )

        if not answer:
            answer = "I was unable to find relevant information for your question."

        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": answer})

        latency_ms = int((time.time() - start) * 1000)
        print(f"\n[Agent] Done in {latency_ms}ms | Tools used: {tools_used}")

        return {
            "answer":     answer,
            "citations":  citations[:5],
            "tools_used": tools_used,
            "steps":      steps,
            "latency_ms": latency_ms,
        }

    def reset_history(self):
        self.history = []


def main():
    print("=" * 60)
    print("  CS 5542 Research Assistant — Agent CLI")
    print("=" * 60)

    agent = ResearchAgent()

    # Pre-fetch chunks once at startup so queries are fast
    agent._prefetch_chunks()

    print("\nType your question and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = agent.run(question)

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nCitations ({len(result['citations'])}):")
        for i, c in enumerate(result['citations'], 1):
            print(f"  [{i}] {c.get('title', 'Unknown')} | "
                  f"{c.get('section', '')} | score={c.get('score', 0):.3f}")
        print(f"\nTools used: {result['tools_used']}")
        print(f"Steps: {result['steps']} | Latency: {result['latency_ms']}ms")
        print("-" * 60)


if __name__ == "__main__":
    main()