# Week 5 Contributions

## Project Title:

CS5542 - Lab 5

------------------------------------------------------------------------

### Member 1: Rohan Hashmi

**Responsibilities:**
- Snowflake database and schema design
- Table creation and staging configuration
- Implementation of CSV → Stage → COPY ingestion
- Validation of data load and record counts
- Connection and environment setup (scripts, `.env`, README)

**Scripts & SQL (implemented or updated):**
- **`scripts/sf_connect.py`** — Central Snowflake connection: loads `.env`, normalizes account (strips `.snowflakecomputing.com`), supports `SNOWFLAKE_AUTHENTICATOR` (priority) and `SNOWFLAKE_MFA_CODE` (TOTP), consistent env variable handling.
- **`scripts/test_connection.py`** — Quick connection test; prints current warehouse, database, schema.
- **`scripts/run_sql_file.py`** — Runs a SQL file (semicolon-separated statements); prints result rows when available.
- **`scripts/load_chunks_to_snowflake.py`** — Loads `data/chunks.csv` into Snowflake stage and `RAW.CHUNKS`.
- **`scripts/export_kb_to_csv.py`** — Exports `data/processed/kb.jsonl` to `data/chunks.csv` for the load pipeline.
- **`sql/00_verify_context.sql`** — Verifies connection context and chunk counts (RAW.CHUNKS, APP.CHUNKS_V).
- **`sql/01_create_schema.sql`** — Creates RAW/APP schemas and `RAW.CHUNKS` table.
- **`sql/02_create_app_view.sql`** — Creates `APP.CHUNKS_V` view over `RAW.CHUNKS`.

**Evidence (PR/commits):** SQL setup and verification scripts; ingestion and export scripts; `sf_connect` auth and env handling; README and CONTRIBUTIONS updates; `.gitignore` and `.env.example`.

**Tested:** Table and view creation in Snowflake; connection test and `run_sql_file` for SQL files; account normalization and external-browser vs password/MFA auth paths; chunk export and load pipeline.

------------------------------------------------------------------------

### Member 2: Blake Simpson

Responsibilities: - Design and implementation of ≥3 meaningful SQL
queries\
- Creation of application-facing views\
- Join, aggregation, and time-based analytics\
- Query validation and performance testing

Evidence (PR/commits): - SQL query scripts\
- View creation scripts\
- Query performance validation commits

Tested: - Aggregation query results\
- Join correctness\
- Time-based filtering accuracy

------------------------------------------------------------------------

### Member 3: Kenneth Kakie

Responsibilities: - Streamlit integration with Snowflake\
- Secure environment configuration handling\
- Dynamic query execution\
- Latency and row-count metrics display\
- pipeline_logs.csv implementation

Evidence (PR/commits): - Streamlit connection code\
- Logging implementation\
- End-to-end integration commits

Tested: - Application-to-database connectivity\
- Query execution and result rendering\
- Logging functionality and file creation
