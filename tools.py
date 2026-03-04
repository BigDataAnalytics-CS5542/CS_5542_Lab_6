"""
Agent-callable tools for the CS 5542 Research Assistant.

Each tool wraps an existing project capability (Snowflake retrieval,
knowledge graph search, LLM generation) into a clean function the
agent can call by name.

All Snowflake tools accept a shared `conn` object so MFA is only
entered once at agent startup and reused across all tool calls.

Tools:
    1. search_papers          — vector search over APP.CHUNKS_V
    2. get_paper_details      — fetch metadata from RAW.PAPERS
    3. search_knowledge_graph — entity + relation lookup in GRAPH tables
    4. summarize_context      — LLM answer generation via HuggingFace
"""

from __future__ import annotations
import json
import os
from typing import Any
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

import data.config as config

load_dotenv()

# ── Shared embedding model (loaded once) ────────────────────
_embedding_model: SentenceTransformer | None = None

def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedding_model


# ════════════════════════════════════════════════════════════
# TOOL 1 — search_papers
# ════════════════════════════════════════════════════════════

def search_papers(conn: Any, query: str, top_k: int = 5) -> list[dict]:
    """
    Search research papers using vector similarity.

    Encodes the query with all-mpnet-base-v2, fetches all chunk
    embeddings from APP.CHUNKS_V, computes cosine similarity in
    numpy, and returns the top-k most relevant chunks.

    Args:
        conn:   Active Snowflake connection (shared, MFA already done).
        query:  Natural language search query.
        top_k:  Number of results to return (default 5).

    Returns:
        List of dicts with keys:
            chunk_id, paper_id, title, section, text, score
    """
    try:
        model = _get_embedding_model()
        query_vec = model.encode([query], normalize_embeddings=True)[0]

        cur = conn.cursor()
        cur.execute(f"USE WAREHOUSE {os.getenv('SNOWFLAKE_WAREHOUSE')}")
        cur.execute(f"USE DATABASE {os.getenv('SNOWFLAKE_DATABASE')}")
        cur.execute(
            "SELECT CHUNK_ID, PAPER_ID, TITLE, SECTION_NAME, TEXT_CONTENT, EMBEDDING "
            "FROM APP.CHUNKS_V"
        )
        rows = cur.fetchall()

        results = []
        for chunk_id, paper_id, title, section, text, emb_json in rows:
            emb = np.array(json.loads(emb_json))
            score = float(np.dot(query_vec, emb))
            results.append({
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "title":    title,
                "section":  section,
                "text":     text,
                "score":    round(score, 4),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    except Exception as e:
        return [{"error": f"search_papers failed: {str(e)}"}]


# ════════════════════════════════════════════════════════════
# TOOL 2 — get_paper_details
# ════════════════════════════════════════════════════════════

def get_paper_details(conn: Any, paper_id: str) -> dict:
    """
    Fetch full metadata for a single paper from RAW.PAPERS.

    Args:
        conn:     Active Snowflake connection.
        paper_id: Paper ID string e.g. "arxiv_000042".

    Returns:
        Dict with keys: paper_id, title, authors, abstract,
        publication_year, source, source_url, categories.
        Returns {"error": ...} if not found.
    """
    try:
        cur = conn.cursor()
        cur.execute(f"USE WAREHOUSE {os.getenv('SNOWFLAKE_WAREHOUSE')}")
        cur.execute(f"USE DATABASE {os.getenv('SNOWFLAKE_DATABASE')}")
        cur.execute(
            "SELECT PAPER_ID, TITLE, AUTHORS, ABSTRACT, "
            "PUBLICATION_YEAR, SOURCE, SOURCE_URL, CATEGORIES "
            "FROM RAW.PAPERS WHERE PAPER_ID = %s",
            (paper_id,)
        )
        row = cur.fetchone()

        if not row:
            return {"error": f"Paper '{paper_id}' not found."}

        return {
            "paper_id":         row[0],
            "title":            row[1],
            "authors":          row[2],
            "abstract":         row[3],
            "publication_year": row[4],
            "source":           row[5],
            "source_url":       row[6],
            "categories":       row[7],
        }

    except Exception as e:
        return {"error": f"get_paper_details failed: {str(e)}"}


# ════════════════════════════════════════════════════════════
# TOOL 3 — search_knowledge_graph
# ════════════════════════════════════════════════════════════

def search_knowledge_graph(conn: Any, query: str, top_k: int = 10) -> list[dict]:
    """
    Find entities and relationships in the knowledge graph
    related to the query.

    Extracts scientific entities from the query using scispaCy,
    then looks up connected nodes and edges in GRAPH tables.

    Args:
        conn:   Active Snowflake connection.
        query:  Natural language query to extract entities from.
        top_k:  Max number of relations to return (default 10).

    Returns:
        List of dicts with keys: source, relation, target, weight.
        Returns empty list if no entities found.
    """
    try:
        try:
            import spacy
            nlp = spacy.load(config.SPACY_MODEL)
            doc = nlp(query)
        except ImportError:
            return [{"error": "spacy is not installed in the current environment (requires Python <3.12). Please use a different environment."}]

        entities = []
        for ent in doc.ents:
            name = ent.text.strip()
            if len(name) >= config.KG_MIN_NAME_LENGTH:
                normalized = name.lower().strip()
                entities.append(normalized)

        if not entities:
            return []

        cur = conn.cursor()
        cur.execute(f"USE WAREHOUSE {os.getenv('SNOWFLAKE_WAREHOUSE')}")
        cur.execute(f"USE DATABASE {os.getenv('SNOWFLAKE_DATABASE')}")

        placeholders = ", ".join(["%s"] * len(entities))
        sql = f"""
            WITH target_nodes AS (
                SELECT NODE_ID, NAME
                FROM GRAPH.KNOWLEDGE_NODES
                WHERE NAME_NORMALIZED IN ({placeholders})
            )
            SELECT tn.NAME, e.RELATION_TYPE, n2.NAME, e.WEIGHT
            FROM target_nodes tn
            JOIN GRAPH.KNOWLEDGE_EDGES e ON tn.NODE_ID = e.SOURCE_NODE_ID
            JOIN GRAPH.KNOWLEDGE_NODES n2 ON e.TARGET_NODE_ID = n2.NODE_ID
            UNION ALL
            SELECT n2.NAME, e.RELATION_TYPE, tn.NAME, e.WEIGHT
            FROM target_nodes tn
            JOIN GRAPH.KNOWLEDGE_EDGES e ON tn.NODE_ID = e.TARGET_NODE_ID
            JOIN GRAPH.KNOWLEDGE_NODES n2 ON e.SOURCE_NODE_ID = n2.NODE_ID
            LIMIT {top_k}
        """
        cur.execute(sql, entities * 2)
        rows = cur.fetchall()

        return [
            {
                "source":   row[0],
                "relation": row[1],
                "target":   row[2],
                "weight":   float(row[3]) if row[3] else 1.0,
            }
            for row in rows
        ]

    except Exception as e:
        return [{"error": f"search_knowledge_graph failed: {str(e)}"}]


# ════════════════════════════════════════════════════════════
# TOOL 4 — summarize_context
# ════════════════════════════════════════════════════════════

def summarize_context(question: str, chunks: list[dict]) -> str:
    """
    Generate a grounded answer using retrieved chunks as context.

    Sends the question and retrieved chunk texts to Llama 3.2
    via HuggingFace Inference API. The LLM is instructed to
    answer only from the provided context and cite chunk indices.

    Args:
        question: The user's original question.
        chunks:   List of chunk dicts from search_papers()
                  (must have 'text', 'title', 'section' keys).

    Returns:
        Generated answer string. Returns error message on failure.
    """
    try:
        if not chunks:
            return "No relevant context found to answer the question."

        formatted = []
        for i, chunk in enumerate(chunks, start=1):
            title   = chunk.get("title", "Unknown")
            section = chunk.get("section", "")
            text    = chunk.get("text", "")
            formatted.append(f"[{i}] Title: {title} | Section: {section}\n{text}")

        context = "\n\n".join(formatted)

        prompt = (
            f"You are a research assistant. Answer the question using ONLY "
            f"the context below. Cite sources using their bracketed index "
            f"(e.g. 'kernel methods are effective [1].'). "
            f"Never invent facts. If the answer is not in the context, "
            f"say 'I don't know based on the available papers.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        client = InferenceClient(token=os.getenv("HF_TOKEN"))
        response = client.chat_completion(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"summarize_context failed: {str(e)}"