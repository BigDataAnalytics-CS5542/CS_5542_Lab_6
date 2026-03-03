## CS 5542 – Lab 6: AI Agent Integration

| Member | Role |
|---|---|---|
| Rohan Ashraf Hashmi | Engineer 1 — Agent & Tools |
| Kenneth Kakie | Engineer 2 — Evaluation & Demo |
| Blake Simpson | Engineer 3 — Streamlit Agent UI |

**Total: 100%**

## Rohan Hashmi
- tools.py — 4 agent tools (search_papers, get_paper_details, search_knowledge_graph, summarize_context)
- tool_schemas.py — OpenAI-compatible schemas + TOOL_FUNCTIONS mapping
- agent.py — ResearchAgent with tool-calling loop, chunk caching, MFA connection, CLI mode, fallbacks
- task1_cursor_report.md — Cursor IDE analysis
- README.md

**Reflection:** Building the agent showed how tool-calling works — the LLM reads schemas and decides which function to call. Main challenge was Llama 3.2 being unreliable with schema adherence, requiring fallbacks and argument remapping.

## Kenneth
- task4_evaluation_report.md — 3 evaluation scenarios
- Demo video (3-5 minutes)

## Blake
- frontend/app.py — Streamlit chat UI with agent integration, conversation history, loading indicator, citations display