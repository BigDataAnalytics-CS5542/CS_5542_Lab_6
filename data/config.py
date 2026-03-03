# ============================================================
# Central configuration for the ingestion pipeline.
# ============================================================

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).resolve().parent.parent
DATA_DIR        = ROOT_DIR / "data"
CHECKPOINT_DIR  = DATA_DIR / "checkpoints"

PAPERS_CHECKPOINT   = CHECKPOINT_DIR / "papers.parquet"
CHUNKS_CHECKPOINT   = CHECKPOINT_DIR / "chunks.parquet"
NODES_CHECKPOINT    = CHECKPOINT_DIR / "nodes.parquet"
EDGES_CHECKPOINT    = CHECKPOINT_DIR / "edges.parquet"
MAP_CHECKPOINT      = CHECKPOINT_DIR / "chunk_entity_map.parquet"

# ── Dataset ──────────────────────────────────────────────────
DATASET_NAME    = "armanc/scientific_papers"
DATASET_SPLIT   = "arxiv"       # use arxiv subset (not pubmed)
NUM_PAPERS      = 1000           # increase for final submission

# ── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE_WORDS    = 200       # target words per chunk
CHUNK_OVERLAP_WORDS = 30        # overlap between consecutive chunks
MIN_CHUNK_WORDS     = 30        # discard chunks shorter than this

# ── Embedding ────────────────────────────────────────────────
EMBEDDING_MODEL     = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM       = 768       # must match VECTOR(FLOAT, 768) in schema
EMBEDDING_BATCH_SIZE = 64       # reduce if you run out of RAM

# ── Knowledge Graph ──────────────────────────────────────────
SPACY_MODEL         = "en_core_sci_sm"
KG_MIN_NAME_LENGTH  = 3         # ignore entities shorter than this

# ── Snowflake table names ────────────────────────────────────
SF_TABLE_PAPERS     = "RAW.PAPERS"
SF_TABLE_CHUNKS     = "RAW.CHUNKS"
SF_TABLE_NODES      = "GRAPH.KNOWLEDGE_NODES"
SF_TABLE_EDGES      = "GRAPH.KNOWLEDGE_EDGES"
SF_TABLE_MAP        = "GRAPH.CHUNK_ENTITY_MAP"