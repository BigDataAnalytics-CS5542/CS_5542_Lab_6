# CS 5542 - Week 6 AI Agent Integration

## End-to-End Cloud Data Pipeline with AI Agent

This project extends our Lab 5 Snowflake-backed cloud data pipeline into a fully functional Agentic system.

### Final Architecture

Structured Output / Document Parsing → Snowflake → **Google GenAI Agent** (Tool Calling) → Streamlit Chat Interface

This lab demonstrates integrating an LLM capable of complex reasoning and routing questions between structured SQL queries and unstructured RAG knowledge base searches.

------------------------------------------------------------------------

## Repository Structure

```
/backend/
  agent.py               # AI Agent execution loop, system prompt, and Chat integration
  tools.py               # Explicit tools for the agent (Snowflake SQL, Schema Lookup, Vector Search)
  rag_pipeline.py        # Vector/BM25 Hybrid retrieval engine

/sql/                    # Snowflake SQL (run in order or via run_sql_file.py)
  01_create_schema.sql   # Create RAW + APP schemas and RAW.CHUNKS table
  02_create_app_view.sql # Create APP.CHUNKS_V view over RAW.CHUNKS
  
/scripts/
  sf_connect.py          # Central Snowflake connection (env, authenticator, MFA support)
  load_chunks_to_snowflake.py  # Load data/chunks.csv → Snowflake stage → RAW.CHUNKS
  
/app/
  streamlit_app.py       # Dual-tab Streamlit dashboard (Metrics + AI Agent Chat)
```

------------------------------------------------------------------------

## Week 6 Scope (≈60% of Project)

| Item | Included This Week | Deferred |
|------|--------------------|----------|
| Agent Tools | Explicit Python tools for Database lookup and RAG search | Agent graph or long-term dynamic memory |
| Conversational UI | Streamlit integrated chat holding session history | Multi-agent collaboration UX |
| Agent Reasoning | LLM executing tools sequentially (e.g., Schema lookup -> Write SQL) | User auth/RBA |

------------------------------------------------------------------------

## Snowflake Setup

Database/warehouse/schema are configured via `.env` (see Local setup below).

**Run SQL in order** (from project root, with venv activated and `.env` set):

```bash
python scripts/run_sql_file.py sql/01_create_schema.sql
python scripts/run_sql_file.py sql/02_create_app_view.sql
```

**Verify connection and context:**

```bash
python scripts/test_connection.py
python scripts/run_sql_file.py sql/00_verify_context.sql
```

------------------------------------------------------------------------

## Data Loading

**Chunk pipeline (RAG → Snowflake):**

1. Export knowledge base to CSV: `python scripts/export_kb_to_csv.py` (reads `data/processed/kb.jsonl`, writes `data/chunks.csv`).
2. Load into Snowflake: `python scripts/load_chunks_to_snowflake.py` (stages and copies into `RAW.CHUNKS`).

**Verification:** run `sql/00_verify_context.sql` or query `RAW.CHUNKS` / `APP.CHUNKS_V` in Snowflake.

------------------------------------------------------------------------

## SQL & Transformation Layer

Implemented:

1.  Aggregation query (GROUP BY analytics)\
2.  Time-based filtering query\
3.  Join across multiple tables

Created view:

APP.V_APP_DATA (application-facing view)

------------------------------------------------------------------------

## Application Integration

### Local setup

1. Clone the repo and go to the project root.
2. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment variables** — Copy `.env.example` to `.env` in the project root and set:

   **Required:** `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`

   **Optional:**  
   - `SNOWFLAKE_AUTHENTICATOR` — e.g. `externalbrowser` for SSO (takes priority; omit for password-only).  
   - `SNOWFLAKE_MFA_CODE` — current 6-digit TOTP when MFA is required (use with password, not with externalbrowser).  
   - `SNOWFLAKE_ROLE` — role name if needed.

   Use the account identifier only in `SNOWFLAKE_ACCOUNT` (e.g. `xy12345` or `org-account`), not the full `.snowflakecomputing.com` host.

5. Run the app:

   ```bash
   streamlit run app/streamlit_app.py
   ```

The application:

-   Connects securely to Snowflake\
-   Executes parameterized SQL queries\
-   Displays query results\
-   Shows latency and returned row count\
-   Logs usage to pipeline_logs.csv

------------------------------------------------------------------------

## Monitoring

Each query logs:

-   timestamp\
-   query/feature used\
-   latency\
-   returned record count

Minimum 10 entries recorded.

------------------------------------------------------------------------

## Team Responsibilities

Please see **`CONTRIBUTIONS.md`** for detailed evidence and testing notes for Lab 6 AI Agent Integration.

------------------------------------------------------------------------

## Demo Video

[Insert Lab 6 Demo Link Here]

------------------------------------------------------------------------

## Screenshots & Architecture Diagram

*(Add screenshots of your Antigravity setup, the new Agent Interface, and any updated architecture diagrams here before final submission)*
