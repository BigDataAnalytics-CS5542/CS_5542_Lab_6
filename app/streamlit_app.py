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
from dotenv import load_dotenv

# Load environment variables (including GEMINI_API_KEY)
load_dotenv()

from scripts.sf_connect import get_conn
from backend.agent import create_agent_session

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
# Tabs Setup
# ---------------------------------------------------------------------------
tab_dash, tab_agent = st.tabs(["📊 Standard Dashboard", "🤖 AI Agent Chat"])

with tab_dash:
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
        COUNT(*)                        AS TOTAL_CHUNKS,
        AVG(LENGTH(CHUNK_TEXT))         AS AVG_CHUNK_LENGTH,
        MIN(LENGTH(CHUNK_TEXT))         AS MIN_CHUNK_LENGTH,
        MAX(LENGTH(CHUNK_TEXT))         AS MAX_CHUNK_LENGTH,
        SUM(LENGTH(CHUNK_TEXT))         AS TOTAL_TEXT_LENGTH
    FROM CS5542_LAB5_ROHAN_BLAKE_KENNETH.APP.CHUNKS_V
    GROUP BY DOC_ID, SOURCE_FILE
    ORDER BY TOTAL_CHUNKS DESC
    LIMIT {limit};
    """
    
    # Q2 — Join: List all Figures for Papers published in 2024
    # NOTE: No explicit publication-date or figure-type column exists in the schema.
    #       "Published in 2024" is inferred from SOURCE_FILE containing '2024',
    #       and "Figures" are identified by CHUNK_TEXT containing the word 'Figure'.
    #       Adjust the ILIKE patterns if your naming convention differs.
    q2 = f"""
    SELECT
        DOC_ID,
        SOURCE_FILE,
        PAGE,
        COUNT(*)                                          AS CHUNKS_ON_PAGE,
        AVG(LENGTH(CHUNK_TEXT))                           AS AVG_CHUNK_LENGTH_ON_PAGE,
        RANK() OVER (PARTITION BY DOC_ID ORDER BY COUNT(*) DESC) AS PAGE_DENSITY_RANK
    FROM CS5542_LAB5_ROHAN_BLAKE_KENNETH.APP.CHUNKS_V
    GROUP BY DOC_ID, SOURCE_FILE, PAGE
    ORDER BY DOC_ID, PAGE_DENSITY_RANK
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
        PAGE,
        CHUNK_INDEX,
        LENGTH(CHUNK_TEXT)      AS CHAR_COUNT,
        CHUNK_TEXT
    FROM CS5542_LAB5_ROHAN_BLAKE_KENNETH.APP.CHUNKS_V
    WHERE LENGTH(CHUNK_TEXT) < 200
    ORDER BY DOC_ID, PAGE, CHUNK_INDEX
    LIMIT {limit};
    """
    
    QUERY_LABELS = [
        "Q1: Average Chunk Length and Total Chunks per Document",
        "Q2: Chunk Density per Page across Documents",
        "Q3: Short Chunks (Likely Noise or Headers) by Document",
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
    
    # ---------------------------------------------------------------------------
    # Logs preview (unchanged)
    # ---------------------------------------------------------------------------
    st.subheader("Logs preview")
    if os.path.exists(LOG_PATH):
        st.dataframe(pd.read_csv(LOG_PATH).tail(50), width='stretch')
    else:
        st.info("No logs yet. Run a query to generate logs.")

with tab_agent:
    st.subheader("Data & Knowledge Base Agent")
    st.markdown("Ask questions about the database statistics or the research papers in the Knowledge Base.")
    
    # Initialize chat session
    if "agent_chat" not in st.session_state:
        try:
            st.session_state.agent_chat = create_agent_session()
            st.session_state.messages = []
        except Exception as e:
            st.error(f"Failed to initialize Agent: {str(e)}")
            st.info("Are you missing the GEMINI_API_KEY in your .env file?")
            st.stop()
            
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # React to user input
    if prompt := st.chat_input("Ask me anything about the metrics or papers..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Agent is reasoning and querying data..."):
                try:
                    # Pass the message to the Gemini agent. Tool calling happens automatically inside SDK.
                    response = st.session_state.agent_chat.send_message(prompt)
                    
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Agent encountered an error: {str(e)}")