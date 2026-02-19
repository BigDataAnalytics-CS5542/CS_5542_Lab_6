# CS 5542 - Lab 5

## Snowflake Integration: End-to-End Cloud Data Pipeline

### Overview

In Lab 5, we extended our Lab 4 RAG application into a Snowflake-backed
cloud data pipeline.

Architecture: Data → Snowflake → SQL Layer → Streamlit App → Monitoring

This lab focuses on structured cloud storage, reproducible ingestion,
SQL transformations, and application-level integration.

------------------------------------------------------------------------

## What We Implemented

### 1. Snowflake Setup

-   Created database: CS5542_DB\
-   Created schemas: RAW (ingested data) and APP (application-facing
    objects)\
-   Configured warehouse and roles\
-   Used environment variables for secure credential handling

### 2. Data Ingestion

-   Loaded \~50% of our project dataset into Snowflake\
-   Implemented reproducible ingestion using `python ingest.py`\
-   Created RAW.DOCS table (and APP.CHUNKS if applicable)\
-   Verified successful load using COUNT queries

### 3. SQL & Transformation Layer

Implemented three meaningful SQL queries: - Aggregation (GROUP BY
analytics) - Time-based filtering - Join across tables (if multiple
tables exist)

Created view: APP.V_APP_DATA --- used by the application for querying.

### 4. Application Integration

-   Connected Streamlit to Snowflake\
-   Replaced local data reads with SQL queries\
-   Displayed:
    -   Query results
    -   Latency (seconds)
    -   Returned record count

Run locally: pip install -r requirements.txt\
streamlit run python/app_streamlit.py

### 5. Monitoring

Each application query logs to: pipeline_logs.csv

Logged fields: - timestamp\
- query/feature used\
- latency\
- returned record count

------------------------------------------------------------------------

## Repository Structure

/sql\
/python\
/diagrams\
pipeline_logs.csv\
README.md\
CONTRIBUTIONS.md

------------------------------------------------------------------------

## Team Responsibilities

### Rohan Hashmi --- Snowflake & Data Engineering

-   Designed database, schemas, and tables\
-   Implemented ingestion script\
-   Loaded and validated dataset in Snowflake\
-   Ensured secure credential management

### Blake Simpson --- SQL & Analytics Layer

-   Designed and implemented required SQL queries\
-   Created application-facing view\
-   Validated joins and aggregations\
-   Documented query logic and performance behavior

### Kenneth Kakie --- Application & Monitoring

-   Integrated Streamlit with Snowflake\
-   Implemented dynamic query execution\
-   Added latency and row-count metrics\
-   Implemented pipeline logging system\
-   Validated end-to-end functionality

------------------------------------------------------------------------

## Demo

\[Insert demo video link here\]

------------------------------------------------------------------------

## Reflection

Lab 5 transformed our project into a structured, warehouse-backed
system. By separating storage, query logic, and application layers, we
improved scalability, monitoring, and production readiness.
