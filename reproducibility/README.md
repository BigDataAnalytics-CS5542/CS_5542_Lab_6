# Reproducibility Guide
## CS 5542 – Snowflake Research Assistant

This document describes how to fully reproduce the Phase 2 system from scratch.

---

## Environment

**Python version:** 3.12
**OS tested:** macOS (Apple Silicon), Ubuntu 24.04
**Package manager:** pip + venv

### Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### scispaCy model (required for KG extraction)
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

---

## Dataset

| Property | Value |
|---|---|
| Name | ccdv/arxiv-summarization |
| Source | https://huggingface.co/datasets/ccdv/arxiv-summarization |
| Split | train |
| Format | Parquet (no loading script required) |
| Fields used | article, abstract |
| Papers ingested | 1000 (configurable via NUM_PAPERS in data/config.py) |
| Streaming | Yes — no full download required |

Dataset is deterministic — same first N papers are always returned in the same order from the train split.

---

## Model Versions

| Component | Model | Version | Source |
|---|---|---|---|
| Embeddings | all-mpnet-base-v2 | Latest | HuggingFace sentence-transformers |
| NER / KG | en_core_sci_sm | 0.5.4 | scispaCy / Allen AI |
| Dimension | 768 | — | Fixed in schema and config |

---

## Configuration

All parameters are in `data/config.py`. Key values:

```python
NUM_PAPERS          = 1000
CHUNK_SIZE_WORDS    = 200
CHUNK_OVERLAP_WORDS = 30
MIN_CHUNK_WORDS     = 30
EMBEDDING_MODEL     = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM       = 768
EMBEDDING_BATCH_SIZE = 64
SPACY_MODEL         = "en_core_sci_sm"
KG_MIN_NAME_LENGTH  = 3
```

---

## Snowflake Configuration

```
Database:  CS5542_PROJECT_ROHAN_BLAKE_KENNETH
Warehouse: ROHAN_BLAKE_KENNETH_WH
Schemas:   RAW, GRAPH, APP
```

Required `.env` variables:
```
SNOWFLAKE_ACCOUNT
SNOWFLAKE_USER
SNOWFLAKE_PASSWORD
SNOWFLAKE_ROLE
SNOWFLAKE_WAREHOUSE
SNOWFLAKE_DATABASE
SNOWFLAKE_SCHEMA
```

Optional (recommended):
```
HF_TOKEN=your_huggingface_token   # For higher rate limits when loading datasets and models
```
Create a token at https://huggingface.co/settings/tokens

---

## Full Reproduction Steps

```bash
# 1. Clone repo and set up environment
git clone <repo-url>
cd snowflake-research-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Edit .env with Snowflake credentials and optionally HF_TOKEN

# 3. Create Snowflake schema
python scripts/run_sql_file.py sql/01_create_schema.sql

# 4. Run full ingestion pipeline
python data/ingestion.py
# Prompts for MFA before Snowflake upload (~1 hour total)

# 5. Start backend
uvicorn backend.app:app --reload --port 3001

# 6. Start frontend
streamlit run frontend/app.py --server.port 3000
```

---

## Checkpoint System

The pipeline saves parquet checkpoints after each stage:

```
data/checkpoints/
├── papers.parquet       # Stage 1 output
├── chunks.parquet       # Stage 2+3 output (includes embeddings)
├── nodes.parquet        # Stage 4 output
├── edges.parquet        # Stage 4 output
└── chunk_entity_map.parquet  # Stage 4 output
```

To resume from checkpoints (skip completed stages):
```bash
python data/ingestion.py --resume
```

Checkpoints are gitignored — each team member runs ingestion independently.

---

## Expected Output

After full ingestion of 1000 papers:

| Table | Expected Rows |
|---|---|
| RAW.PAPERS | ~1,000 |
| RAW.CHUNKS | ~35,000 |
| GRAPH.KNOWLEDGE_NODES | ~190,000+ |
| GRAPH.KNOWLEDGE_EDGES | ~25,000,000+ |
| GRAPH.CHUNK_ENTITY_MAP | ~1,600,000+ |
| APP.EVAL_METRICS | 0 (populated at query time) |