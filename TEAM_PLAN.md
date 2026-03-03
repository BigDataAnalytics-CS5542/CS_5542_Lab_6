# CS 5542 Lab 6: AI Agent Integration — Team Plan

## Objective
Extend the Phase 2 Snowflake RAG pipeline with an AI agent layer capable of reasoning over user queries, selecting tools automatically, and generating grounded answers with citations.

The agent interprets natural language questions, decides which tools to call (vector search, knowledge graph, LLM generation), and returns grounded answers backed by retrieved evidence from Snowflake.

---

## Agent Architecture

```
User question
      ↓
Llama 3.2 (HuggingFace) reads system prompt + tool schemas
      ↓ decides which tools to call
search_papers()          — vector search over 35,349 chunks in Snowflake
search_knowledge_graph() — entity/relation lookup in GRAPH tables
get_paper_details()      — fetch paper metadata from RAW.PAPERS
      ↓ citations collected
summarize_context()      — Llama 3.2 writes grounded answer from chunks
      ↓
Answer + citations + tools_used + latency returned
```

---

## Tool Design

| Tool | Owner | Description |
|---|---|---|
| `search_papers` | Rohan | Vector similarity search over APP.CHUNKS_V |
| `get_paper_details` | Rohan | Fetch paper metadata from RAW.PAPERS |
| `search_knowledge_graph` | Rohan | Entity + relation lookup in GRAPH tables |
| `summarize_context` | Rohan | Llama 3.2 generates grounded answer from chunks |

---

## Role-Based Task Breakdown

### Rohan Ashraf Hashmi

Completed:
- `tools.py` — 4 agent-callable tool functions
- `tool_schemas.py` — OpenAI-compatible schemas + TOOL_FUNCTIONS mapping
- `agent.py` — ResearchAgent class with tool-calling loop, chunk caching, CLI mode
- `task1_cursot_report.md` — Cursor IDE analysis report
- README.md update

---

### Kenneth Kakie
- `task4_evaluation_report.md` — 3 evaluation scenarios
- Demo video (3-5 minutes)

---

### Blake Simpson
- Update `frontend/app.py` — Streamlit chat interface connected to agent
  - `st.chat_input` and `st.chat_message` for conversation UI
  - `st.session_state` for conversation history
  - Loading indicator while agent runs
  - Display citations and tools used below each answer
