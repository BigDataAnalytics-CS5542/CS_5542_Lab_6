from typing import List, Dict

import json, os, numpy as np


import spacy
import data.config as config



# Load the scientific NLP model once to avoid overhead on every request
# We disable 'parser' and 'linker' to speed up the extraction process
nlp = spacy.load(config.SPACY_MODEL, disable=["parser", "ner"])
# We add the entity ruler or use the default pipe depending on needs
# For scientific entities, the default pipe usually suffices:
nlp_ner = spacy.load(config.SPACY_MODEL)

def extract_query_entities(query: str) -> List[str]:
    """
    Extracts and normalizes scientific entities from a natural language query.
    
    Args:
        query: The user's search string.
        
    Returns:
        A list of unique, normalized (uppercase) entity strings.
    """
    if not query:
        return []

    # Process the text
    doc = nlp_ner(query)
    
    entities = []
    for ent in doc.ents:
        # 1. Basic Cleaning
        cleaned_ent = ent.text.strip()
        
        # 2. Filter by length based on config.py (KG_MIN_NAME_LENGTH = 3)
        if len(cleaned_ent) >= config.KG_MIN_NAME_LENGTH:
            # 3. Normalize for database lookup (matching the ingestion logic)
            entities.append(cleaned_ent.upper())
            
    # Return unique entities only
    return list(set(entities))


def get_top_chunks(conn, query_text, top_k=5, passcode=''):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    query_vec = model.encode([query_text], normalize_embeddings=True)[0]

    cur = conn.cursor()
    cur.execute('USE WAREHOUSE ROHAN_BLAKE_KENNETH_WH')
    cur.execute('USE DATABASE CS5542_PROJECT_ROHAN_BLAKE_KENNETH')
    cur.execute(
        'SELECT CHUNK_ID, PAPER_ID, TITLE, SECTION_NAME, TEXT_CONTENT, EMBEDDING FROM APP.CHUNKS_V'
    )
    rows = cur.fetchall()
    

    results = []
    for chunk_id, paper_id, title, section, text, emb_json in rows:
        emb = np.array(json.loads(emb_json))
        score = float(np.dot(query_vec, emb))
        results.append((score, chunk_id, paper_id, title, section, text))

    results.sort(reverse=True)
    return results[:top_k]



def graph_search(conn, query: str) -> List[Dict]:
    '''
    Query GRAPH.KNOWLEDGE_EDGES and KNOWLEDGE_NODES to find relations 
    connected to the extracted entities.
    '''
    entities = extract_query_entities(query=query)
    if not entities:
        return []

    
    cur = conn.cursor()
    
    # 1. Setup session context
    cur.execute('USE WAREHOUSE ROHAN_BLAKE_KENNETH_WH')
    cur.execute('USE DATABASE CS5542_PROJECT_ROHAN_BLAKE_KENNETH')

    # 2. Normalize entities for matching (assuming case-insensitive search)
    # We use a parameterized query to prevent injection and handle the IN clause
    placeholders = ', '.join(['%s'] * len(entities))
    normalized_entities = [e.strip().upper() for e in entities]

    query = f"""
    WITH target_nodes AS (
        -- Find IDs for the entities provided
        SELECT NODE_ID, NAME 
        FROM GRAPH.KNOWLEDGE_NODES 
        WHERE NAME_NORMALIZED IN ({placeholders})
    )
    SELECT 
        tn.NAME as SOURCE_ENTITY,
        e.RELATION_TYPE,
        n2.NAME as TARGET_ENTITY,
        e.WEIGHT
    FROM target_nodes tn
    JOIN GRAPH.KNOWLEDGE_EDGES e ON tn.NODE_ID = e.SOURCE_NODE_ID
    JOIN GRAPH.KNOWLEDGE_NODES n2 ON e.TARGET_NODE_ID = n2.NODE_ID
    
    UNION ALL
    
    SELECT 
        n2.NAME as SOURCE_ENTITY,
        e.RELATION_TYPE,
        tn.NAME as TARGET_ENTITY,
        e.WEIGHT
    FROM target_nodes tn
    JOIN GRAPH.KNOWLEDGE_EDGES e ON tn.NODE_ID = e.TARGET_NODE_ID
    JOIN GRAPH.KNOWLEDGE_NODES n2 ON e.SOURCE_NODE_ID = n2.NODE_ID
    """

    cur.execute(query, normalized_entities + normalized_entities)
    rows = cur.fetchall()


    # 3. Format results into a list of dictionaries
    graph_results = []
    for src, rel, tgt, weight in rows:
        graph_results.append({
            "source": src,
            "relation": rel,
            "target": tgt,
            "weight": float(weight) if weight else 1.0
        })

    return graph_results

