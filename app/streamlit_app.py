# import os
# import sys

# # Ensure project root is on path so "scripts" can be imported
# _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if _PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, _PROJECT_ROOT)

# import time
# import pandas as pd
# import streamlit as st
# import altair as alt
# from datetime import datetime
# from scripts.sf_connect import get_conn

# LOG_PATH = os.path.join(_PROJECT_ROOT, "logs", "pipeline_logs.csv")

# def log_event(team: str, user: str, query_name: str, latency_ms: int, rows: int, error: str = ""):
#     os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
#     row = {
#         "timestamp": datetime.utcnow().isoformat(),
#         "team": team,
#         "user": user,
#         "query_name": query_name,
#         "latency_ms": latency_ms,
#         "rows_returned": rows,
#         "error": error,
#     }
#     df = pd.DataFrame([row])
#     header = not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0
#     df.to_csv(LOG_PATH, mode="a", header=header, index=False)

# def run_query(sql: str):
#     t0 = time.time()
#     with get_conn() as conn:
#         df = pd.read_sql(sql, conn)
#     latency_ms = int((time.time() - t0) * 1000)
#     return df, latency_ms

# st.title("CS 5542 — Week 5 Snowflake Dashboard Starter")

# team = st.text_input("Team name", value="TeamX")
# user = st.text_input("Your name", value="StudentName")

# st.subheader("Filters")
# category = st.text_input("Category filter (optional)", value="")
# limit = st.slider("Limit rows", 10, 200, 50)

# base_where = ""
# if category.strip():
#     safe = category.strip().replace("'", "''")
#     base_where = f"WHERE CATEGORY ILIKE '%{safe}%'"

# q1 = f"""
# SELECT TEAM, CATEGORY, COUNT(*) AS N, AVG(VALUE) AS AVG_VALUE
# FROM CS5542_WEEK5.PUBLIC.EVENTS
# {base_where}
# GROUP BY TEAM, CATEGORY
# ORDER BY N DESC
# LIMIT {limit};
# """

# q2 = f"""
# SELECT CATEGORY, COUNT(*) AS N_24H
# FROM CS5542_WEEK5.PUBLIC.EVENTS
# WHERE EVENT_TIME >= DATEADD('hour', -24, CURRENT_TIMESTAMP())
# GROUP BY CATEGORY
# ORDER BY N_24H DESC
# LIMIT 20;
# """

# q3 = f"""
# SELECT U.TEAM, U.ROLE, E.CATEGORY, COUNT(*) AS N
# FROM CS5542_WEEK5.PUBLIC.USERS U
# JOIN CS5542_WEEK5.PUBLIC.EVENTS E
#   ON U.TEAM = E.TEAM
# GROUP BY U.TEAM, U.ROLE, E.CATEGORY
# ORDER BY N DESC
# LIMIT {limit};
# """

# choice = st.selectbox("Choose query", ["Q1: Team x Category stats", "Q2: Category last 24h", "Q3: Join users x events"])
# sql = {"Q1: Team x Category stats": q1, "Q2: Category last 24h": q2, "Q3: Join users x events": q3}[choice]

# if st.button("Run"):
#     try:
#         df, latency_ms = run_query(sql)
#         st.caption(f"Latency: {latency_ms} ms | Rows: {len(df)}")
#         st.dataframe(df, width='stretch')

#         if "N" in df.columns and "CATEGORY" in df.columns:
#             chart = alt.Chart(df).mark_bar().encode(x="CATEGORY:N", y="N:Q")
#             st.altair_chart(chart, width='stretch')

#         log_event(team, user, choice, latency_ms, len(df), "")
#     except Exception as e:
#         st.error(str(e))
#         log_event(team, user, choice, 0, 0, str(e))

# st.subheader("Logs preview")
# if os.path.exists(LOG_PATH):
#     st.dataframe(pd.read_csv(LOG_PATH).tail(50), width='stretch')
# else:
#     st.info("No logs yet. Run a query to generate logs.")

import os
import sys

# Ensure project root is on path so "scripts" can be imported
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import time
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime
from scripts.sf_connect import get_conn

LOG_PATH = os.path.join(_PROJECT_ROOT, "logs", "pipeline_logs.csv")

def log_event(team: str, user: str, query_name: str, latency_ms: int, rows: int, error: str = ""):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "team": team,
        "user": user,
        "query_name": query_name,
        "latency_ms": latency_ms,
        "rows_returned": rows,
        "error": error,
    }
    df = pd.DataFrame([row])
    header = not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0
    df.to_csv(LOG_PATH, mode="a", header=header, index=False)

def run_query(sql: str, passcode: str = ""):
    """Execute a SQL query against Snowflake and return (DataFrame, latency_ms)."""
    conn = get_conn(passcode=passcode)
    try:
        t0 = time.time()
        df = pd.read_sql(sql, conn)
        latency_ms = int((time.time() - t0) * 1000)
    finally:
        conn.close()
    return df, latency_ms

def fetch_evidence_chunks(evidence_ids: list, passcode: str = "") -> pd.DataFrame:
    """
    Given an ordered list of evidence_ids (ranked by the RAG pipeline),
    query APP.CHUNKS_V for matching rows and return them in rank order.
    """
    if not evidence_ids:
        return pd.DataFrame()

    # Build a safe IN-list of quoted IDs
    id_list = ", ".join(f"'{eid.replace(chr(39), chr(39)*2)}'" for eid in evidence_ids)
    sql = f"""
        SELECT EVIDENCE_ID, SOURCE_FILE, PAGE, CHUNK_INDEX, CHUNK_TEXT
        FROM APP.CHUNKS_V
        WHERE EVIDENCE_ID IN ({id_list});
    """
    conn = get_conn(passcode=passcode)
    try:
        df = pd.read_sql(sql, conn)
    finally:
        conn.close()

    if df.empty:
        return df

    # Normalise column names to upper for consistent lookup
    df.columns = [c.upper() for c in df.columns]

    # Re-sort to match the RAG-ranked order (SQL IN doesn't guarantee order)
    rank_map = {eid: idx for idx, eid in enumerate(evidence_ids)}
    df["_RANK"] = df["EVIDENCE_ID"].map(rank_map)
    df = df.sort_values("_RANK").drop(columns=["_RANK"]).reset_index(drop=True)
    df.index += 1  # 1-based rank display
    return df

# ---------------------------------------------------------------------------
# Page header (unchanged)
# ---------------------------------------------------------------------------
st.title("CS 5542 — Week 5 Snowflake Dashboard Starter")

team = st.text_input("Team name", value="TeamX")
user = st.text_input("Your name", value="StudentName")

# MFA passcode — validated as exactly 6 digits; passed as "" if not provided
mfa_passcode_raw = st.text_input(
    "MFA Passcode (6-digit, optional)",
    value="",
    max_chars=6,
    type="password",
    help="Enter your 6-digit MFA passcode if your Snowflake account requires one.",
)

# ---------------------------------------------------------------------------
# Filters (unchanged layout; limit still used by Q1 & Q3)
# ---------------------------------------------------------------------------
st.subheader("Filters")
limit = st.slider("Limit rows", 10, 200, 50)

# ---------------------------------------------------------------------------
# Predefined queries against CS5542_LAB5_ROHAN_BLAKE_KENNETH
# ---------------------------------------------------------------------------

# Q1 — Aggregation: Average Token Count per Paper
# NOTE: No dedicated token-count column exists in RAW.CHUNKS; token count is
#       approximated as LENGTH(CHUNK_TEXT) (character count). If a tokeniser
#       column is added later, replace the expression below.
q1 = f"""
SELECT
    DOC_ID,
    SOURCE_FILE,
    COUNT(*)                          AS CHUNK_COUNT,
    AVG(LENGTH(CHUNK_TEXT))           AS AVG_CHAR_COUNT,
    SUM(LENGTH(CHUNK_TEXT))           AS TOTAL_CHARS
FROM CS5542_LAB5_ROHAN_BLAKE_KENNETH.RAW.CHUNKS
GROUP BY DOC_ID, SOURCE_FILE
ORDER BY AVG_CHAR_COUNT DESC
LIMIT {limit};
"""

# Q2 — Join: List all Figures for Papers published in 2024
# NOTE: No explicit publication-date or figure-type column exists in the schema.
#       "Published in 2024" is inferred from SOURCE_FILE containing '2024',
#       and "Figures" are identified by CHUNK_TEXT containing the word 'Figure'.
#       Adjust the ILIKE patterns if your naming convention differs.
q2 = f"""
SELECT
    c.EVIDENCE_ID,
    c.DOC_ID,
    c.SOURCE_FILE,
    c.PAGE,
    c.CHUNK_INDEX,
    c.CHUNK_TEXT
FROM CS5542_LAB5_ROHAN_BLAKE_KENNETH.RAW.CHUNKS c
WHERE c.SOURCE_FILE ILIKE '%2024%'
  AND c.CHUNK_TEXT  ILIKE '%figure%'
ORDER BY c.DOC_ID, c.PAGE, c.CHUNK_INDEX
LIMIT {limit};
"""

# Q3 — Complex: Compare Performance Metrics across Papers (Rank by chunk density)
# NOTE: RAG metrics (r_at_10, p_at_5, etc.) are produced at query time and are
#       not stored in APP.CHUNKS_V. This query uses chunk count per paper as a
#       structural proxy and ranks papers accordingly. Wire in an actual metrics
#       table if/when one is persisted to Snowflake.
q3 = f"""
SELECT
    DOC_ID,
    SOURCE_FILE,
    COUNT(*)                                      AS TOTAL_CHUNKS,
    MAX(PAGE)                                     AS MAX_PAGE,
    COUNT(DISTINCT PAGE)                          AS DISTINCT_PAGES,
    ROUND(COUNT(*) / NULLIF(MAX(PAGE), 0), 2)     AS CHUNKS_PER_PAGE,
    RANK() OVER (ORDER BY COUNT(*) DESC)          AS CHUNK_RANK
FROM CS5542_LAB5_ROHAN_BLAKE_KENNETH.RAW.CHUNKS
GROUP BY DOC_ID, SOURCE_FILE
ORDER BY CHUNK_RANK
LIMIT {limit};
"""

QUERY_LABELS = [
    "Q1: Average Token Count per Paper",
    "Q2: List all Figures for Papers published in 2024",
    "Q3: Compare Performance Metrics across Papers",
]
QUERY_MAP = {
    QUERY_LABELS[0]: q1,
    QUERY_LABELS[1]: q2,
    QUERY_LABELS[2]: q3,
}

choice = st.selectbox("Choose query", QUERY_LABELS)
sql = QUERY_MAP[choice]

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------
if st.button("Run"):
    # --- Predefined query ---
    try:
        mfa_passcode=""
        # Only use the passcode if it is exactly 6 numeric digits
        mfa_passcode = mfa_passcode_raw.strip() if mfa_passcode_raw.strip().isdigit() and len(mfa_passcode_raw.strip()) == 6 else ""

        df, latency_ms = run_query(sql, passcode=mfa_passcode)
        st.caption(f"Latency: {latency_ms} ms | Rows: {len(df)}")
        st.dataframe(df, width='stretch')
        log_event(team, user, choice, latency_ms, len(df), "")
    except Exception as e:
        st.error(str(e))
        log_event(team, user, choice, 0, 0, str(e))

    # --- RAG Evidence Retrieval ---
    # Reads the last RAG result stored in session state by the existing RAG
    # pipeline (expected key: "rag_result"). If none is present the section
    # is silently skipped — no RAG question box is added here.
    st.subheader("RAG Evidence Chunks")
    rag_result = st.session_state.get("rag_result")

    if rag_result is None:
        st.info("No RAG result in session. Run the RAG pipeline first to populate evidence.")
    else:
        evidence = rag_result.get("evidence", [])
        if not evidence:
            st.info("RAG result contains no evidence entries.")
        else:
            # Extract evidence_ids in ranked order (list is already ranked by pipeline)
            ranked_ids = [e["evidence_id"] for e in evidence if e.get("evidence_id")]

            st.caption(
                f"Question: *{rag_result.get('question', 'N/A')}* | "
                f"Faithfulness: {'✅' if rag_result.get('faithfulness_pass') else '❌'} | "
                f"RAG Latency: {rag_result.get('latency_ms', 'N/A')} ms"
            )

            try:
                evidence_df = fetch_evidence_chunks(ranked_ids, passcode=mfa_passcode)
                if evidence_df.empty:
                    st.warning("No matching chunks found in APP.CHUNKS_V for the returned evidence IDs.")
                else:
                    st.dataframe(evidence_df, width='stretch')
                    log_event(team, user, "RAG Evidence Fetch", 0, len(evidence_df), "")
            except Exception as e:
                st.error(f"Evidence fetch failed: {e}")
                log_event(team, user, "RAG Evidence Fetch", 0, 0, str(e))

# ---------------------------------------------------------------------------
# Logs preview (unchanged)
# ---------------------------------------------------------------------------
st.subheader("Logs preview")
if os.path.exists(LOG_PATH):
    st.dataframe(pd.read_csv(LOG_PATH).tail(50), width='stretch')
else:
    st.info("No logs yet. Run a query to generate logs.")