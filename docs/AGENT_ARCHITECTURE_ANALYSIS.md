# Agent Architecture Analysis & Improvement Suggestions

## 1. Current Project Structure (Summary)

```
CS_5542_Lab_6/
├── agent.py              # CLI agent: tool-calling loop, HuggingFace, Snowflake
├── tools.py              # Tool implementations (search_papers, get_paper_details, KG, summarize)
├── tool_schemas.py       # LLM tool schemas + TOOL_FUNCTIONS mapping
├── backend/
│   ├── app.py            # FastAPI: /query, /papers, /health (separate RAG pipeline)
│   ├── retrieval.py     # get_top_chunks, graph_search (duplicates tools logic)
│   └── history.json     # Query history
├── frontend/
│   └── app.py           # Streamlit UI (placeholder, no backend/agent integration)
├── data/
│   ├── config.py        # Central config (embedding model, spacy, paths)
│   └── ingestion.py     # Pipeline into Snowflake
├── scripts/
│   └── sf_connect.py    # Snowflake connection (env-based)
└── evaluation/
    └── evaluate.py      # Eval harness
```

**Two separate RAG paths today:**
- **Agent path:** `agent.py` → tool-calling loop → `tools.py` / in-agent `_fast_search` → HuggingFace chat + tools.
- **API path:** `backend/app.py` → `backend/retrieval.py` → direct HuggingFace call. No agent, no tool use.

---

## 2. Issues Identified

### 2.1 Duplication & Inconsistency

| Concern | Where | Detail |
|--------|--------|--------|
| **Vector search** | agent, tools, backend | Three implementations: `agent._fast_search` (cached in-memory), `tools.search_papers` (Snowflake every call), `backend/retrieval.get_top_chunks` (Snowflake, no cache). Logic and behavior can drift. |
| **KG search** | tools vs backend/retrieval | `tools.search_knowledge_graph` normalizes entities to **lowercase**; `retrieval.graph_search` uses **UPPERCASE**. Ingestion uses `data/config`; schema may expect one convention. Risk of silent empty results. |
| **Snowflake context** | backend vs rest | `backend/retrieval.py` and `backend/app.py` use **hardcoded** `ROHAN_BLAKE_KENNETH_WH` and `CS5542_PROJECT_ROHAN_BLAKE_KENNETH`. Everywhere else uses `os.getenv("SNOWFLAKE_WAREHOUSE")` / `SNOWFLAKE_DATABASE`. |
| **Embedding model** | tools vs agent | `tools.py` uses `config.EMBEDDING_MODEL` and a shared `_get_embedding_model()`. `agent._fast_search` instantiates `SentenceTransformer("sentence-transformers/all-mpnet-base-v2")` directly — duplicates config and can diverge. |
| **Summarize step** | agent vs backend | Agent has a fallback that calls `summarize_context` with collected citations if the LLM never did. Backend has its own prompt and no tool loop. Two different “answer from context” behaviors. |

### 2.2 Agent-Specific Design Issues

- **Bypass of tools layer:** For `search_papers`, the agent calls `_fast_search` instead of `TOOL_FUNCTIONS["search_papers"]`, so the “single implementation” in `tools.py` is not the one used by the CLI agent. Harder to test and evolve one search behavior.
- **Heavy coupling in `run()`:** Citation accumulation, fallback summarize, and “if KG used without citations then run _fast_search” are all hard-coded in the loop. Adding a new tool or flow requires editing the agent loop.
- **No dependency injection:** Snowflake `conn`, HuggingFace client, and embedding model are created and cached inside the agent. This makes unit tests and swapping backends (e.g. local embedding server) harder.
- **Model and config in code:** `MODEL_ID` and embedding model name are in source; they should come from config/env for different environments and experiments.

### 2.3 Backend / API Issues

- **`input()` in HTTP handlers:** `backend/app.py` uses `input("OTP : ")` and `input("OPT : ")` inside `/query` and `/papers`. This blocks the server and is unusable in a real API; auth should be token/header or out-of-band MFA.
- **Backend does not use the agent:** The API reimplements “retrieve → format → LLM” instead of reusing the agent. Bug fixes and behavior changes must be done in two places.
- **Return structure typo:** `return { result }` returns a set-like single-key dict; likely intended `return result`.

### 2.4 Frontend & Integration

- **Streamlit is a stub:** `frontend/app.py` shows a dummy response and does not call the backend or the agent. No unified path from UI → API → agent.

### 2.5 Operational / Config

- **Single config for Snowflake:** Good: `scripts/sf_connect.py` and most code use env. Bad: backend ignores env for warehouse/database.
- **Entity normalization:** KG ingestion and retrieval should document and share one normalization (e.g. in `data/config`: `KG_NORMALIZE = "lower"` or `"upper"`) and use it in both `tools.py` and `backend/retrieval.py`.

---

## 3. Suggested Improvements

### 3.1 Unify Retrieval and Config (High impact)

- **Single retrieval module:** Move all “search papers” and “search knowledge graph” logic into one place, e.g. `backend/retrieval.py` or a new `core/retrieval.py`. Have both the agent and the FastAPI app call this module only.
- **Agent uses tools that call the core:** Implement `tools.search_papers` and `tools.search_knowledge_graph` as thin wrappers that call the same functions used by the API (e.g. `get_top_chunks`, `graph_search`). Remove `_fast_search` from the agent and route `search_papers` through `TOOL_FUNCTIONS` only.
- **One entity normalization:** Decide one convention (e.g. uppercase to match `retrieval.py` and many SQL setups). Put it in `data/config.py` (e.g. `KG_ENTITY_NORMALIZE = "upper"`) and use it in both tools and backend retrieval.
- **Backend Snowflake:** Use `os.getenv("SNOWFLAKE_WAREHOUSE")` and `os.getenv("SNOWFLAKE_DATABASE")` in `backend/retrieval.py` and `backend/app.py`, and remove hardcoded names.

### 3.2 Agent Architecture (High impact)

- **Single tool execution path:** Do not special-case `search_papers` in the agent. Either:
  - (A) Make `tools.search_papers` fast by giving it an optional “connection or preloaded chunk cache” and use it from the agent, or  
  - (B) Keep a cache in the agent but expose it via a dedicated “search service” that both `tools.search_papers` and the agent use, so there is still one logical implementation.
- **Extract orchestration from citation bookkeeping:** Move “collect citations,” “run summarize if missing,” and “fallback search when KG-only” into a small helper or a separate “response builder” that consumes tool results. Keep the main loop as: get LLM response → execute tools → append to messages → repeat. Easier to add tools and change behavior without touching the loop.
- **Config-driven model and prompts:** Move `MODEL_ID`, summarize model, and `SYSTEM_PROMPT` (or path to it) into `data/config.py` or env (e.g. `HF_CHAT_MODEL`, `HF_SUMMARIZE_MODEL`). Use the same config in agent and in any backend summarize step.
- **Optional dependency injection:** Allow passing `conn`, `llm_client`, and an optional “search backend” (e.g. cached vs live Snowflake) into `ResearchAgent.__init__`. Default to current behavior; tests and alternative deployments can inject mocks or different implementations.

### 3.3 Backend and API (High impact)

- **Remove interactive `input()`:** Require connection (or a server-side session) to be established at startup or via a dedicated auth flow (e.g. token or one-time MFA at startup). Pass `conn` or a connection factory into the app; never block request handlers on stdin.
- **Use the agent from the API:** Add an endpoint (e.g. `POST /agent/query`) that instantiates or reuses the same `ResearchAgent` (or a stateless “run once” function that uses the same tools and config). Forward the request body to `agent.run(question)` and return `answer`, `citations`, `tools_used`, etc. This gives one behavior for both CLI and API.
- **Fix return value:** In `/query`, change `return { result }` to `return result` (and optionally keep a simple RAG-only path at `/query` that uses the shared retrieval module).

### 3.4 Frontend (Medium impact)

- **Connect Streamlit to backend:** Call the FastAPI backend (e.g. `POST /query` or `POST /agent/query`) from the Streamlit app. Display `answer`, expandable citations, and optionally `tools_used` and latency. Remove the dummy response and hardcoded confidence/chunk IDs.

### 3.5 Testing and Maintainability (Medium impact)

- **Shared tool schema and mapping:** Keep `tool_schemas.py` as the single place for tool names, descriptions, and `TOOL_FUNCTIONS`. Ensure every tool the agent can call is implemented in `tools.py` (or the chosen core module) and that the agent does not bypass them with in-class logic.
- **Embedding model loading:** Use `data.config.EMBEDDING_MODEL` and a single loader (e.g. from `tools.py` or a small `core/embeddings.py`) everywhere, including any in-agent search path until it is removed.

### 3.6 Optional Longer-Term Improvements

- **Structured logging:** Use a single logger (e.g. `logging.getLogger("research_assistant")`) and log tool calls, retrieval counts, and latency in a consistent format for debugging and evaluation.
- **Evaluation alignment:** Point `evaluation/evaluate.py` at the unified retrieval and (if possible) the same agent entrypoint used by the API, so metrics reflect production behavior.
- **Rate limiting and timeouts:** In the API, add timeouts for Snowflake and HuggingFace calls and optional rate limiting to protect the service.

---

## 4. Recommended Order of Work

1. **Unify Snowflake and KG:** Env-based warehouse/database in backend; single entity normalization in config and use it in both tools and retrieval.
2. **Single retrieval implementation:** One `get_top_chunks` and one `graph_search`; agent and backend both call them (tools become thin wrappers); remove `_fast_search` from the agent.
3. **Fix backend API:** Remove `input()`; fix `return { result }`; optionally add `POST /agent/query` that uses the agent.
4. **Config and prompts:** Move model IDs and main prompts to config/env; use shared embedding model loader everywhere.
5. **Connect frontend** to the backend and optionally to the agent endpoint.
6. **Refine agent loop:** Extract citation/summarize fallback into a helper; consider dependency injection for tests and flexibility.

This order reduces duplication first, then aligns API and agent behavior, then improves maintainability and UX.
