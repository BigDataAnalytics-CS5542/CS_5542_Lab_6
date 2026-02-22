# CS 5542 - Week 5 Snowflake Integration

## End-to-End Cloud Data Pipeline

This project extends our Lab 4 RAG-based system into a structured,
Snowflake-backed cloud data pipeline.

### Final Architecture

Data → Snowflake Stage + COPY → Tables & Views → SQL Queries → Streamlit
App → Monitoring Logs

This lab demonstrates reproducible ingestion, warehouse-backed
analytics, application integration, and structured monitoring.

------------------------------------------------------------------------

## Repository Structure

```
/sql/                    # Snowflake SQL (run in order or via run_sql_file.py)
  00_verify_context.sql  # Verify connection, show warehouse/DB/schema and chunk counts
  01_create_schema.sql   # Create RAW + APP schemas and RAW.CHUNKS table
  02_create_app_view.sql # Create APP.CHUNKS_V view over RAW.CHUNKS
  03_queries.sql        # Sample analytics (EVENTS/USERS aggregation, time filter, join)

/scripts/
  sf_connect.py          # Central Snowflake connection (env, authenticator, MFA support)
  test_connection.py    # Quick connection test (warehouse, database, schema)
  run_sql_file.py        # Run a SQL file (semicolon-separated statements); prints results
  load_chunks_to_snowflake.py  # Load data/chunks.csv → Snowflake stage → RAW.CHUNKS
  export_kb_to_csv.py    # Export data/processed/kb.jsonl → data/chunks.csv

/app/
  streamlit_app.py       # Streamlit dashboard (queries, filters, logs)

/data/
  chunks.csv             # Chunk data for loading (from export_kb_to_csv.py)
  processed/kb.jsonl     # Source for export
/logs/
  pipeline_logs.csv      # Query usage and latency logs
README.md, CONTRIBUTIONS.md
```

------------------------------------------------------------------------

## Week 5 Scope (≈50% of Project)

| Item | Included This Week | Deferred |
|------|--------------------|----------|
| Dataset Subset | ~50% of project documents loaded into Snowflake | Remaining documents |
| Structured Metadata | Document + chunk metadata tables | Full embedding storage |
| Analytics Queries | Aggregation, time-based, and join queries | Advanced optimization |

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

See **CONTRIBUTIONS.md** for detailed evidence and testing notes.

### Rohan Hashmi — Snowflake & Data Engineering

- Designed database, schemas, and tables (RAW, APP, RAW.CHUNKS)
- Implemented staging and COPY ingestion (`load_chunks_to_snowflake.py`), export pipeline (`export_kb_to_csv.py`)
- Connection and env: `sf_connect.py` (account normalization, authenticator vs MFA), `test_connection.py`, `run_sql_file.py`
- SQL setup/verification: `00_verify_context.sql`, `01_create_schema.sql`, `02_create_app_view.sql`

### Blake Simpson — SQL & Analytics Layer

- Designed and implemented required SQL queries (aggregation, time-based, join)
- Created application-facing views
- Query validation and performance testing

### Kenneth Kakie — Application & Monitoring

- Streamlit integration with Snowflake backend
- Dynamic query execution, latency and row-count metrics
- Pipeline logging (`pipeline_logs.csv`), end-to-end verification

------------------------------------------------------------------------

## Extensions Completed

-   Structured multi-table join analytics\
-   Application-level query performance metrics\
-   Enhanced monitoring and structured logging

------------------------------------------------------------------------

## Demo Video

\[Insert Demo Link Here\]

------------------------------------------------------------------------

## Reflection

This lab transformed our project into a structured cloud-backed system
by separating storage, transformation, and application layers.
Integrating Snowflake improved reproducibility, scalability, and
production-readiness of our architecture.
