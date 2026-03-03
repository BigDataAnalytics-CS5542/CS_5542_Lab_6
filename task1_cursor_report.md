# Task 1 — Cursor IDE Analysis Report
## CS 5542 Lab 6 | Rohan Ashraf Hashmi

---

## 1. Setup

**IDE Used:** Cursor (approved alternative to Google Antigravity IDE)

Cursor was opened on the full Lab 6 project repository. The project structure was visible in the file explorer including `agent.py`, `tools.py`, `tool_schemas.py`, `backend/`, `frontend/`, `data/`, and `scripts/`.

**Prompt given to Cursor:**
> "Analyze this project structure and suggest improvements to the agent architecture"

Cursor analyzed all project files and produced a detailed architecture report saved to `docs/AGENT_ARCHITECTURE_ANALYSIS.md`.

---

## 2. What Cursor Identified

### 2.1 Project Structure Summary
Cursor correctly identified the two separate RAG paths in the project:

- **Agent path:** `agent.py` → HuggingFace tool-calling loop → `tools.py` + in-agent `_fast_search` (cached vector search)
- **API path:** `backend/app.py` → `backend/retrieval.py` → direct LLM call with no agent or tool use

### 2.2 Key Issues Found

**Duplication:**
- Vector search implemented in three places: `agent._fast_search`, `tools.search_papers`, and `backend/retrieval.get_top_chunks` — logic can drift between them
- KG search in two places with conflicting entity normalization: `tools.search_knowledge_graph` uses lowercase while `backend/retrieval.graph_search` uses uppercase — risk of silent empty results

**Agent Design:**
- The agent bypasses `tools.search_papers` and uses its own `_fast_search`, so the tools layer is not the actual execution path
- Citation handling and summarize fallback are hard-coded in the agent loop, making it harder to add new tools
- `MODEL_ID` and embedding model name are hardcoded in source instead of `data/config.py`

**Backend Issues:**
- `input("OTP : ")` inside FastAPI HTTP handlers blocks the server — unusable in production
- `return { result }` is a Python set, not a dict — should be `return result`
- Hardcoded Snowflake warehouse and database names instead of using env vars

**Frontend:**
- `frontend/app.py` is a stub with dummy responses — not connected to backend or agent

---

## 3. Changes Accepted

Based on Cursor's suggestions, the following improvements were accepted and implemented for Lab 6:

| Change | Accepted | Notes |
|---|---|---|
| Chunk caching in agent (`_fast_search`) | ✅ Yes | Reduces query latency from ~52s to ~9s |
| Forced `summarize_context` fallback | ✅ Yes | Ensures grounded answers even if LLM skips the step |
| KG search auto-triggers `search_papers` | ✅ Yes | Ensures citations always present |
| Single shared Snowflake connection | ✅ Yes | MFA entered once at startup |
| Move model IDs to config | ⏳ Partial | `MODEL_ID` still in `agent.py` — planned for Phase 3 |
| Remove `input()` from backend handlers | ⏳ Deferred | Kenneth's backend — noted for next iteration |
| Fix `return { result }` bug | ⏳ Deferred | Kenneth's backend — noted for next iteration |
| Unify retrieval into single module | ⏳ Deferred | Planned refactor for Phase 3 |
| Connect Streamlit to backend | ⏳ Deferred | Blake's deliverable |

---

## 4. Changes Modified or Rejected

| Suggestion | Decision | Reason |
|---|---|---|
| Dependency injection for conn/client | Modified | Kept simple `__init__` with MFA prompt — injection adds complexity not needed for Lab 6 scope |
| Remove `_fast_search` entirely | Modified | Kept cache for performance — but added fallback to ensure tools always produce citations |
| Single retrieval module (`core/retrieval.py`) | Deferred | Good long-term architecture but too large a refactor for tonight's deadline |

---

## 5. Reflection

Cursor behaved as an effective intelligent assistant for this task. It:

- **Understood the full project** without any manual explanation — it read `agent.py`, `tools.py`, `backend/retrieval.py`, and `data/config.py` and identified relationships between them
- **Found real bugs** — the `return { result }` bug in Kenneth's backend and the `input()` blocking issue are genuine problems that could cause failures in production
- **Prioritized suggestions** by impact (High/Medium) and gave a concrete recommended order of work rather than just listing problems
- **Identified architectural drift** — noting that three separate vector search implementations could produce inconsistent results was a non-obvious insight that required understanding the whole codebase

The most valuable output was identifying the **entity normalization inconsistency** between `tools.py` (lowercase) and `backend/retrieval.py` (uppercase). This would cause `search_knowledge_graph` to return empty results in one of the two paths silently — exactly the kind of bug that is hard to spot in code review but obvious to a tool that can compare two files side by side.

Overall Cursor accelerated the architectural review significantly. A manual review of the same scope would have taken considerably longer.

---

## 6. Screenshot

*(Screenshot of Cursor analyzing the project attached separately — see `docs/cursor_screenshot.png`)*