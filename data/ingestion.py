"""
data/ingestion.py
=================
End-to-end ingestion pipeline for the CS 5542 Research Assistant.

Stages:
    1. load_and_clean_dataset()   - Load arXiv papers from HuggingFace
    2. chunk_documents()          - Split papers into text chunks
    3. generate_embeddings()      - Embed chunks with all-mpnet-base-v2
    4. extract_knowledge_graph()  - Extract KG nodes/edges via scispaCy
    5. upload_to_snowflake()      - Push all DataFrames to Snowflake
    6. verify_ingestion()         - Sanity check row counts

Usage:
    python data/ingestion.py                  # run all stages, n=500
    python data/ingestion.py --n 10           # smaller run for testing
    python data/ingestion.py --stage load     # run one stage only
    python data/ingestion.py --resume         # skip stages with checkpoints
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from snowflake.connector.pandas_tools import write_pandas
from tqdm import tqdm

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.config import (
    CHECKPOINT_DIR,
    CHUNK_OVERLAP_WORDS,
    CHUNK_SIZE_WORDS,
    CHUNKS_CHECKPOINT,
    EDGES_CHECKPOINT,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    KG_MIN_NAME_LENGTH,
    MAP_CHECKPOINT,
    MIN_CHUNK_WORDS,
    NODES_CHECKPOINT,
    NUM_PAPERS,
    PAPERS_CHECKPOINT,
    SPACY_MODEL,
)
from scripts.sf_connect import get_conn

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════
# STAGE 1 — Load and Clean Dataset
# ════════════════════════════════════════════════════════════

def _clean_text(text: str) -> str:
    """Remove LaTeX, normalize whitespace."""
    if not text:
        return ""
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", " ", text)
    text = re.sub(r"\\[a-zA-Z]+\{.*?\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean_dataset(n: int = NUM_PAPERS, resume: bool = False) -> pd.DataFrame:
    """
    Load n papers from HuggingFace ccdv/arxiv-summarization.
    Standard Parquet format — no loading script, works with latest datasets lib.
    Fields: article (full text), abstract.

    Returns DataFrame with columns:
        paper_id, title, authors, abstract, publication_year,
        source, source_url, categories, sections, section_names

    Saves checkpoint to PAPERS_CHECKPOINT.
    """
    if resume and PAPERS_CHECKPOINT.exists():
        print(f"[Stage 1] Resuming from checkpoint: {PAPERS_CHECKPOINT}")
        df = pd.read_parquet(PAPERS_CHECKPOINT)
        print(f"[Stage 1] Loaded {len(df)} papers from checkpoint.")
        return df

    print(f"[Stage 1] Loading {n} papers from HuggingFace (ccdv/arxiv-summarization)...")
    print("[Stage 1] Streaming — no full dataset download needed.")

    dataset = load_dataset(
        "ccdv/arxiv-summarization",
        split="train",
        streaming=True,
    )

    records = []
    skipped = 0

    for i, item in enumerate(tqdm(dataset, total=n, desc="Loading papers")):
        if len(records) >= n:
            break

        abstract = _clean_text(item.get("abstract", ""))
        article  = _clean_text(item.get("article", ""))

        if not abstract or len(abstract.split()) < 20:
            skipped += 1
            continue
        if not article or len(article.split()) < 50:
            skipped += 1
            continue

        paper_id = f"arxiv_{i:06d}"

        records.append({
            "paper_id":         paper_id,
            "title":            f"arXiv Paper {paper_id}",
            "authors":          "",
            "abstract":         abstract,
            "publication_year": None,
            "source":           "arxiv",
            "source_url":       "",
            "categories":       "",
            "section_names":    ["abstract", "body"],
            "sections":         [abstract, article],
        })

    df = pd.DataFrame(records)
    print(f"[Stage 1] Loaded {len(df)} papers. Skipped {skipped} (too short/empty).")
    df.to_parquet(PAPERS_CHECKPOINT, index=False)
    print(f"[Stage 1] Checkpoint saved → {PAPERS_CHECKPOINT}")
    return df


# ════════════════════════════════════════════════════════════
# STAGE 2 — Chunk Documents
# ════════════════════════════════════════════════════════════

def chunk_documents(papers_df: pd.DataFrame, resume: bool = False) -> pd.DataFrame:
    """
    Split each paper into overlapping word-based chunks.
    Strategy:
        - Abstract → always ONE chunk (section_name = 'abstract')
        - Body → sliding window of CHUNK_SIZE_WORDS with CHUNK_OVERLAP_WORDS overlap

    chunk_id format: {paper_id}_{section_slug}_c{index:03d}
    e.g. "arxiv_000000_abstract_c000", "arxiv_000000_body_c001"

    Returns DataFrame with columns:
        chunk_id, paper_id, chunk_index, section_name, text_content, word_count

    Saves checkpoint to CHUNKS_CHECKPOINT.
    """
    if resume and CHUNKS_CHECKPOINT.exists():
        print(f"[Stage 2] Resuming from checkpoint: {CHUNKS_CHECKPOINT}")
        df = pd.read_parquet(CHUNKS_CHECKPOINT)
        print(f"[Stage 2] Loaded {len(df)} chunks from checkpoint.")
        return df

    print(f"[Stage 2] Chunking {len(papers_df)} papers...")

    def split_into_chunks(text: str, size: int, overlap: int) -> list[str]:
        """Split text into overlapping word windows."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end >= len(words):
                break
            start += size - overlap
        return chunks

    records = []
    chunk_index_global = 0

    for _, row in tqdm(papers_df.iterrows(), total=len(papers_df), desc="Chunking papers"):
        paper_id      = row["paper_id"]
        section_names = row["section_names"]
        sections      = row["sections"]

        for sec_name, sec_text in zip(section_names, sections):
            if not sec_text or len(sec_text.split()) < MIN_CHUNK_WORDS:
                continue

            slug = re.sub(r"[^a-z0-9]", "_", sec_name.lower())

            if sec_name == "abstract":
                # Abstract is always one single chunk
                chunks = [sec_text]
            else:
                chunks = split_into_chunks(sec_text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)

            for i, chunk_text in enumerate(chunks):
                word_count = len(chunk_text.split())
                if word_count < MIN_CHUNK_WORDS:
                    continue

                chunk_id = f"{paper_id}_{slug}_c{i:03d}"

                records.append({
                    "chunk_id":     chunk_id,
                    "paper_id":     paper_id,
                    "chunk_index":  chunk_index_global,
                    "section_name": sec_name,
                    "text_content": chunk_text,
                    "word_count":   word_count,
                })
                chunk_index_global += 1

    df = pd.DataFrame(records)
    print(f"[Stage 2] Created {len(df)} chunks from {len(papers_df)} papers.")
    print(f"[Stage 2] Avg chunks per paper: {len(df)/len(papers_df):.1f}")

    df.to_parquet(CHUNKS_CHECKPOINT, index=False)
    print(f"[Stage 2] Checkpoint saved → {CHUNKS_CHECKPOINT}")

    return df


# ════════════════════════════════════════════════════════════
# STAGE 3 — Generate Embeddings
# ════════════════════════════════════════════════════════════

def generate_embeddings(chunks_df: pd.DataFrame, resume: bool = False) -> pd.DataFrame:
    """
    Add 'embedding' column (list of 768 floats) to chunks_df.
    Uses all-mpnet-base-v2 via sentence-transformers.
    Processes in batches, shows progress bar.
    Normalizes embeddings (L2 norm) for cosine similarity.

    Saves updated chunks_df to CHUNKS_CHECKPOINT.
    """
    if resume and CHUNKS_CHECKPOINT.exists():
        df = pd.read_parquet(CHUNKS_CHECKPOINT)
        if "embedding" in df.columns:
            print(f"[Stage 3] Resuming from checkpoint — {len(df)} chunks already embedded.")
            return df

    print(f"[Stage 3] Embedding {len(chunks_df)} chunks with {EMBEDDING_MODEL}...")
    print(f"[Stage 3] Batch size: {EMBEDDING_BATCH_SIZE}. This will take a few minutes.")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = chunks_df["text_content"].tolist()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), EMBEDDING_BATCH_SIZE), desc="Embedding chunks"):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        embeddings = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.extend(embeddings.tolist())

    chunks_df = chunks_df.copy()
    chunks_df["embedding"] = all_embeddings

    # Verify dimension
    assert len(all_embeddings[0]) == EMBEDDING_DIM, \
        f"Expected {EMBEDDING_DIM}-dim embeddings, got {len(all_embeddings[0])}"

    chunks_df.to_parquet(CHUNKS_CHECKPOINT, index=False)
    print(f"[Stage 3] Embedded {len(chunks_df)} chunks. Dim={EMBEDDING_DIM}")
    print(f"[Stage 3] Checkpoint saved → {CHUNKS_CHECKPOINT}")

    return chunks_df


# ════════════════════════════════════════════════════════════
# STAGE 4 — Extract Knowledge Graph
# ════════════════════════════════════════════════════════════

def extract_knowledge_graph(chunks_df: pd.DataFrame, resume: bool = False):
    """
    Extract entities and relationships from chunk text using scispaCy.

    - Each unique entity → one KNOWLEDGE_NODE
    - Two entities in same chunk → CO_OCCURS edge
    - chunk_entity_map links each chunk to entities it mentions

    Returns: nodes_df, edges_df, map_df
    Saves checkpoints for all three.
    """
    if resume and NODES_CHECKPOINT.exists() and EDGES_CHECKPOINT.exists() and MAP_CHECKPOINT.exists():
        print(f"[Stage 4] Resuming from checkpoints.")
        return (
            pd.read_parquet(NODES_CHECKPOINT),
            pd.read_parquet(EDGES_CHECKPOINT),
            pd.read_parquet(MAP_CHECKPOINT),
        )

    print(f"[Stage 4] Extracting knowledge graph from {len(chunks_df)} chunks...")
    print(f"[Stage 4] Loading scispaCy model: {SPACY_MODEL}...")

    nlp = spacy.load(SPACY_MODEL)

    # ── Entity extraction ────────────────────────────────────
    # node_name_normalized → { node_id, name, count, paper_ids }
    node_registry = {}
    map_records   = []
    edge_pairs    = {}  # (node_id_a, node_id_b, paper_id) → weight

    for _, row in tqdm(chunks_df.iterrows(), total=len(chunks_df), desc="Extracting entities"):
        chunk_id = row["chunk_id"]
        paper_id = row["paper_id"]
        text     = row["text_content"]

        doc = nlp(text)

        # Get unique entities in this chunk
        chunk_entities = []
        for ent in doc.ents:
            name = ent.text.strip()
            if len(name) < KG_MIN_NAME_LENGTH:
                continue

            # Normalize: lowercase, collapse whitespace
            normalized = re.sub(r"\s+", " ", name.lower()).strip()
            normalized = re.sub(r"[^a-z0-9 ]", "", normalized).strip()
            if not normalized:
                continue

            # Register node if new
            if normalized not in node_registry:
                node_id = "node_" + re.sub(r"\s+", "_", normalized)[:60]
                node_registry[normalized] = {
                    "node_id":         node_id,
                    "name":            name,
                    "name_normalized": normalized,
                    "label":           "Entity",
                    "paper_ids":       set(),
                }
            node_registry[normalized]["paper_ids"].add(paper_id)
            chunk_entities.append(normalized)

            # chunk_entity_map record
            map_records.append({
                "map_id":     str(uuid.uuid4()),
                "chunk_id":   chunk_id,
                "node_id":    node_registry[normalized]["node_id"],
                "confidence": 1.0,
            })

        # ── CO_OCCURS edges (entities in same chunk) ─────────
        seen = list(dict.fromkeys(chunk_entities))  # deduplicate, preserve order
        for i in range(len(seen)):
            for j in range(i + 1, len(seen)):
                a = node_registry[seen[i]]["node_id"]
                b = node_registry[seen[j]]["node_id"]
                key = (min(a, b), max(a, b), paper_id)
                edge_pairs[key] = edge_pairs.get(key, 0) + 1

    # ── Build nodes DataFrame ────────────────────────────────
    node_records = []
    for normalized, info in node_registry.items():
        node_records.append({
            "node_id":         info["node_id"],
            "label":           info["label"],
            "name":            info["name"],
            "name_normalized": normalized,
            "paper_count":     len(info["paper_ids"]),
            "embedding":       None,   # node embeddings added in future phase
        })
    nodes_df = pd.DataFrame(node_records)

    # ── Build edges DataFrame ────────────────────────────────
    edge_records = []
    for (src, tgt, paper_id), weight in edge_pairs.items():
        edge_records.append({
            "edge_id":        str(uuid.uuid4()),
            "source_node_id": src,
            "target_node_id": tgt,
            "relation_type":  "CO_OCCURS",
            "paper_id":       paper_id,
            "weight":         float(weight),
        })
    edges_df = pd.DataFrame(edge_records)
    map_df   = pd.DataFrame(map_records)

    print(f"[Stage 4] Extracted {len(nodes_df)} unique entities.")
    print(f"[Stage 4] Created {len(edges_df)} edges.")
    print(f"[Stage 4] Created {len(map_df)} chunk-entity mappings.")

    nodes_df.to_parquet(NODES_CHECKPOINT, index=False)
    edges_df.to_parquet(EDGES_CHECKPOINT, index=False)
    map_df.to_parquet(MAP_CHECKPOINT,   index=False)
    print(f"[Stage 4] Checkpoints saved.")

    return nodes_df, edges_df, map_df


# ════════════════════════════════════════════════════════════
# STAGE 5 — Upload to Snowflake
# ════════════════════════════════════════════════════════════

def truncate_tables(conn):
    """
    Truncate all tables in correct FK order before re-ingestion.
    Safe to run multiple times.
    """
    cur = conn.cursor()
    cur.execute(f"USE WAREHOUSE {os.getenv('SNOWFLAKE_WAREHOUSE')}")
    cur.execute(f"USE DATABASE {os.getenv('SNOWFLAKE_DATABASE')}")

    tables = [
        "GRAPH.CHUNK_ENTITY_MAP",
        "GRAPH.KNOWLEDGE_EDGES",
        "GRAPH.KNOWLEDGE_NODES",
        "RAW.CHUNKS",
        "RAW.PAPERS",
    ]
    print("[Stage 5] Truncating existing tables before upload...")
    for table in tables:
        cur.execute(f"TRUNCATE TABLE {table}")
        print(f"  -> Truncated {table}")
    print("[Stage 5] Tables cleared.")


def upload_to_snowflake(papers_df, chunks_df, nodes_df, edges_df, map_df, passcode=""):
    """
    Upload all DataFrames to Snowflake in correct foreign key order:
        1. RAW.PAPERS
        2. RAW.CHUNKS       (references PAPERS)
        3. GRAPH.KNOWLEDGE_NODES
        4. GRAPH.KNOWLEDGE_EDGES   (references NODES + PAPERS)
        5. GRAPH.CHUNK_ENTITY_MAP  (references CHUNKS + NODES)

    EMBEDDING column is uploaded as Snowflake VECTOR type.
    Uses OVERWRITE mode — safe to re-run.
    """
    print(f"[Stage 5] Connecting to Snowflake...")
    conn = get_conn(passcode=passcode)
    cur  = conn.cursor()

    warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
    database  = os.getenv("SNOWFLAKE_DATABASE")
    cur.execute(f"USE WAREHOUSE {warehouse}")
    cur.execute(f"USE DATABASE {database}")

    truncate_tables(conn)

    # ── 1. RAW.PAPERS ────────────────────────────────────────
    print(f"[Stage 5] Uploading {len(papers_df)} rows → RAW.PAPERS...")
    upload_df = papers_df[[
        "paper_id", "title", "authors", "abstract",
        "publication_year", "source", "source_url", "categories"
    ]].copy()
    upload_df.columns = [c.upper() for c in upload_df.columns]
    upload_df["PUBLICATION_YEAR"] = upload_df["PUBLICATION_YEAR"].astype(object)
    write_pandas(conn, upload_df, "PAPERS", schema="RAW", overwrite=True, auto_create_table=False)
    print(f"[Stage 5] RAW.PAPERS done.")

    # ── 2. RAW.CHUNKS (embedding stored as JSON string) ──────
    print(f"[Stage 5] Uploading {len(chunks_df)} rows → RAW.CHUNKS...")
    chunks_upload = chunks_df[[
        "chunk_id", "paper_id", "chunk_index",
        "section_name", "text_content", "word_count", "embedding"
    ]].copy()
    chunks_upload.columns = [c.upper() for c in chunks_upload.columns]
    chunks_upload["EMBEDDING"] = chunks_upload["EMBEDDING"].apply(
        lambda x: json.dumps(list(x))
    )
    write_pandas(conn, chunks_upload, "CHUNKS", schema="RAW", overwrite=True, auto_create_table=False)
    print(f"[Stage 5] RAW.CHUNKS done.")

    # ── 3. GRAPH.KNOWLEDGE_NODES ─────────────────────────────
    print(f"[Stage 5] Uploading {len(nodes_df)} rows → GRAPH.KNOWLEDGE_NODES...")
    nodes_upload = nodes_df[[
        "node_id", "label", "name", "name_normalized", "paper_count"
    ]].copy()
    nodes_upload.columns = [c.upper() for c in nodes_upload.columns]
    write_pandas(conn, nodes_upload, "KNOWLEDGE_NODES", schema="GRAPH", overwrite=True, auto_create_table=False)
    print(f"[Stage 5] GRAPH.KNOWLEDGE_NODES done.")

    # ── 4. GRAPH.KNOWLEDGE_EDGES ─────────────────────────────
    print(f"[Stage 5] Uploading {len(edges_df)} rows → GRAPH.KNOWLEDGE_EDGES...")
    edges_upload = edges_df[[
        "edge_id", "source_node_id", "target_node_id",
        "relation_type", "paper_id", "weight"
    ]].copy()
    edges_upload.columns = [c.upper() for c in edges_upload.columns]
    write_pandas(conn, edges_upload, "KNOWLEDGE_EDGES", schema="GRAPH", overwrite=True, auto_create_table=False)
    print(f"[Stage 5] GRAPH.KNOWLEDGE_EDGES done.")

    # ── 5. GRAPH.CHUNK_ENTITY_MAP ────────────────────────────
    print(f"[Stage 5] Uploading {len(map_df)} rows → GRAPH.CHUNK_ENTITY_MAP...")
    map_upload = map_df[["map_id", "chunk_id", "node_id", "confidence"]].copy()
    map_upload.columns = [c.upper() for c in map_upload.columns]
    write_pandas(conn, map_upload, "CHUNK_ENTITY_MAP", schema="GRAPH", overwrite=True, auto_create_table=False)
    print(f"[Stage 5] GRAPH.CHUNK_ENTITY_MAP done.")

    print(f"\n[Stage 5] All tables uploaded successfully.")
    return conn  # pass to verify so it reuses same connection/MFA session


# ════════════════════════════════════════════════════════════
# STAGE 6 — Verify Ingestion
# ════════════════════════════════════════════════════════════

def verify_ingestion(passcode="", conn=None):
    """
    Run COUNT(*) on all tables and print a summary.
    """
    print(f"[Stage 6] Verifying Snowflake tables...")
    if conn is None:
        conn = get_conn(passcode=passcode)
    cur  = conn.cursor()

    cur.execute(f"USE WAREHOUSE {os.getenv('SNOWFLAKE_WAREHOUSE')}")
    cur.execute(f"USE DATABASE {os.getenv('SNOWFLAKE_DATABASE')}")

    tables = [
        ("RAW",   "PAPERS"),
        ("RAW",   "CHUNKS"),
        ("GRAPH", "KNOWLEDGE_NODES"),
        ("GRAPH", "KNOWLEDGE_EDGES"),
        ("GRAPH", "CHUNK_ENTITY_MAP"),
        ("APP",   "EVAL_METRICS"),
    ]

    print(f"\n{'Schema':<10} {'Table':<25} {'Row Count':>10}")
    print("-" * 48)
    all_good = True
    for schema, table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
        count = cur.fetchone()[0]
        status = "OK" if count > 0 or table == "EVAL_METRICS" else "EMPTY"
        print(f"{schema:<10} {table:<25} {count:>10,}  {status}")
        if count == 0 and table != "EVAL_METRICS":
            all_good = False

    try:
        conn.close()
    except Exception:
        pass
    print("-" * 48)
    if all_good:
        print("\n[Stage 6] All tables populated. Ingestion successful!")
    else:
        print("\n[Stage 6] Some tables are empty. Check Stage 5 logs.")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CS 5542 Ingestion Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "load", "chunk", "embed", "kg", "upload", "verify"],
        default="all",
        help="Which stage to run (default: all)",
    )
    parser.add_argument("--n", type=int, default=NUM_PAPERS,
                        help=f"Number of papers to load (default: {NUM_PAPERS})")
    parser.add_argument("--resume", action="store_true",
                        help="Skip stages that already have checkpoints")
    args = parser.parse_args()

    mfa = ""
    conn = None

    if args.stage in ("all", "load"):
        papers_df = load_and_clean_dataset(n=args.n, resume=args.resume)
        print(f"\n[Stage 1] Sample output:")
        print(papers_df[["paper_id", "title", "abstract"]].head(3).to_string())
        print()

    if args.stage in ("all", "chunk"):
        if args.stage == "chunk":
            papers_df = pd.read_parquet(PAPERS_CHECKPOINT)
        chunks_df = chunk_documents(papers_df, resume=args.resume)

    if args.stage in ("all", "embed"):
        if args.stage == "embed":
            chunks_df = pd.read_parquet(CHUNKS_CHECKPOINT)
        chunks_df = generate_embeddings(chunks_df, resume=args.resume)

    if args.stage in ("all", "kg"):
        if args.stage == "kg":
            chunks_df = pd.read_parquet(CHUNKS_CHECKPOINT)
        nodes_df, edges_df, map_df = extract_knowledge_graph(chunks_df, resume=args.resume)

    if args.stage in ("all", "upload"):
        mfa = input("MFA code for Snowflake (or Enter to skip): ").strip()
        if args.stage == "upload":
            papers_df = pd.read_parquet(PAPERS_CHECKPOINT)
            chunks_df = pd.read_parquet(CHUNKS_CHECKPOINT)
            nodes_df  = pd.read_parquet(NODES_CHECKPOINT)
            edges_df  = pd.read_parquet(EDGES_CHECKPOINT)
            map_df    = pd.read_parquet(MAP_CHECKPOINT)
        conn = upload_to_snowflake(papers_df, chunks_df, nodes_df, edges_df, map_df, passcode=mfa)

    if args.stage in ("all", "verify"):
        if conn is not None:
            verify_ingestion(conn=conn)
        else:
            if not mfa:
                mfa = input("MFA code for Snowflake verify (or Enter to skip): ").strip()
            verify_ingestion(passcode=mfa)


if __name__ == "__main__":
    main()