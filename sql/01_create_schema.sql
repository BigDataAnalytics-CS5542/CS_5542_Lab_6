CREATE SCHEMA IF NOT EXISTS RAW;
CREATE SCHEMA IF NOT EXISTS GRAPH;
CREATE SCHEMA IF NOT EXISTS APP;

-- ── RAW.PAPERS ──────────────────────────────────────────────
-- One row per source paper
CREATE OR REPLACE TABLE RAW.PAPERS (
    PAPER_ID            VARCHAR PRIMARY KEY,    -- arXiv ID e.g. "2305.14283"
    TITLE               VARCHAR,
    AUTHORS             VARCHAR,                -- comma-separated string
    ABSTRACT            STRING,
    PUBLICATION_YEAR    INT,
    SOURCE              VARCHAR DEFAULT 'arxiv',-- 'arxiv', 'pubmed', 'user_upload'
    SOURCE_URL          VARCHAR,
    CATEGORIES          VARCHAR,                -- e.g. "cs.IR cs.CL"
    INGESTED_AT         TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ── RAW.CHUNKS ───────────────────────────────────────────────
-- Text segments for vector RAG. One paper → many chunks.
CREATE OR REPLACE TABLE RAW.CHUNKS (
    CHUNK_ID            VARCHAR PRIMARY KEY,    -- e.g. "2305.14283_abstract_c001"
    PAPER_ID            VARCHAR REFERENCES RAW.PAPERS(PAPER_ID),
    CHUNK_INDEX         INT,                    -- position within paper
    SECTION_NAME        VARCHAR,                -- e.g. "abstract", "introduction"
    TEXT_CONTENT        STRING,
    WORD_COUNT          INT,
    EMBEDDING           VARCHAR,     -- all-mpnet-base-v2 output
    INGESTED_AT         TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ── RAW.FIGURES ──────────────────────────────────────────────
-- Image metadata for future multimodal support
CREATE OR REPLACE TABLE RAW.FIGURES (
    FIGURE_ID           VARCHAR PRIMARY KEY,
    PAPER_ID            VARCHAR REFERENCES RAW.PAPERS(PAPER_ID),
    PAGE_NUMBER         INT,
    CAPTION             STRING,
    IMAGE_PATH          VARCHAR
);

-- ── GRAPH.KNOWLEDGE_NODES ────────────────────────────────────
-- Extracted entities: methods, datasets, concepts, tasks
CREATE OR REPLACE TABLE GRAPH.KNOWLEDGE_NODES (
    NODE_ID             VARCHAR PRIMARY KEY,    -- e.g. "node_retrieval_augmented_generation"
    LABEL               VARCHAR,                -- 'Method', 'Dataset', 'Concept', 'Task'
    NAME                VARCHAR,                -- e.g. "Retrieval-Augmented Generation"
    NAME_NORMALIZED     VARCHAR,                -- lowercased, no punctuation
    PAPER_COUNT         INT DEFAULT 0,          -- distinct papers mentioning this node
    EMBEDDING           VARCHAR      -- for node-level similarity search
);

-- ── GRAPH.KNOWLEDGE_EDGES ────────────────────────────────────
-- Relationships between entities
CREATE OR REPLACE TABLE GRAPH.KNOWLEDGE_EDGES (
    EDGE_ID             VARCHAR PRIMARY KEY,
    SOURCE_NODE_ID      VARCHAR REFERENCES GRAPH.KNOWLEDGE_NODES(NODE_ID),
    TARGET_NODE_ID      VARCHAR REFERENCES GRAPH.KNOWLEDGE_NODES(NODE_ID),
    RELATION_TYPE       VARCHAR,                -- 'USES', 'IMPROVES', 'EVALUATED_ON', 'CO_OCCURS'
    PAPER_ID            VARCHAR REFERENCES RAW.PAPERS(PAPER_ID),
    WEIGHT              FLOAT DEFAULT 1.0,
    INGESTED_AT         TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ── GRAPH.CHUNK_ENTITY_MAP ───────────────────────────────────
-- Links chunks to entities they mention.
-- Critical for Engineer 2's graph-enhanced retrieval.
CREATE OR REPLACE TABLE GRAPH.CHUNK_ENTITY_MAP (
    MAP_ID              VARCHAR PRIMARY KEY,
    CHUNK_ID            VARCHAR REFERENCES RAW.CHUNKS(CHUNK_ID),
    NODE_ID             VARCHAR REFERENCES GRAPH.KNOWLEDGE_NODES(NODE_ID),
    CONFIDENCE          FLOAT DEFAULT 1.0
);

-- ── APP.CHUNKS_V ─────────────────────────────────────────────
-- Application-facing view: chunks joined with paper metadata.
-- This is what Engineer 2 queries for retrieval.
CREATE OR REPLACE VIEW APP.CHUNKS_V AS
SELECT
    c.CHUNK_ID,
    c.PAPER_ID,
    c.CHUNK_INDEX,
    c.SECTION_NAME,
    c.TEXT_CONTENT,
    c.WORD_COUNT,
    c.EMBEDDING,
    p.TITLE,
    p.AUTHORS,
    p.PUBLICATION_YEAR,
    p.CATEGORIES,
    p.SOURCE_URL
FROM RAW.CHUNKS c
JOIN RAW.PAPERS p ON c.PAPER_ID = p.PAPER_ID;

-- ── APP.EVAL_METRICS ─────────────────────────────────────────
-- Every query the system answers gets logged here by Engineer 3.
CREATE OR REPLACE TABLE APP.EVAL_METRICS (
    LOG_ID                  VARCHAR PRIMARY KEY,
    QUESTION                STRING,
    GENERATED_RESPONSE      STRING,
    CONTEXT_USED            STRING,
    RETRIEVAL_MODE          VARCHAR,
    FAITHFULNESS_SCORE      FLOAT,
    ANSWER_RELEVANCE_SCORE  FLOAT,
    CONFIDENCE              FLOAT,
    LATENCY_MS              INT,
    TIMESTAMP               TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ── Verification ─────────────────────────────────────────────
SELECT 'Schema setup complete' AS STATUS;

SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA IN ('RAW', 'GRAPH', 'APP')
ORDER BY TABLE_SCHEMA, TABLE_NAME;