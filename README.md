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

/sql\
01_create_schema.sql\
02_stage_and_load.sql\
03_queries.sql

/scripts\
load_local_csv_to_stage.py

/app\
streamlit_app.py

/data\
/logs\
pipeline_logs.csv\
README.md\
CONTRIBUTIONS.md

------------------------------------------------------------------------

## Week 5 Scope (≈50% of Project)

| Item | Included This Week | Deferred |
|------|--------------------|----------|
| Dataset Subset | ~50% of project documents loaded into Snowflake | Remaining documents |
| Structured Metadata | Document + chunk metadata tables | Full embedding storage |
| Analytics Queries | Aggregation, time-based, and join queries | Advanced optimization |

------------------------------------------------------------------------

## Snowflake Setup

Database: CS5542_DB\
Schemas: RAW, APP\
Warehouse: Configured via environment variables

Run in order: 1. sql/01_create_schema.sql\
2. sql/02_stage_and_load.sql

------------------------------------------------------------------------

## Data Loading

Example:

python scripts/load_local_csv_to_stage.py data/events.csv EVENTS\
python scripts/load_local_csv_to_stage.py data/users.csv USERS

Verification:

SELECT COUNT(\*) FROM CS5542_DB.RAW.DOCS;

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

Run locally:

pip install -r requirements.txt\
streamlit run app/streamlit_app.py

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

### Rohan Hashmi - Snowflake & Data Engineering

-   Designed database, schemas, and tables\
-   Implemented staging and COPY ingestion workflow\
-   Loaded and validated 50% dataset subset\
-   Managed secure environment configuration

### Blake Simpson - SQL & Analytics Layer

-   Designed and implemented required SQL queries\
-   Created application-facing views\
-   Implemented aggregation, filter, and join analytics\
-   Validated query correctness and performance

### Kenneth Kakie - Application & Monitoring

-   Integrated Streamlit with Snowflake backend\
-   Replaced local data reads with warehouse queries\
-   Implemented latency and row-count metrics\
-   Built pipeline logging system\
-   Verified end-to-end functionality

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
