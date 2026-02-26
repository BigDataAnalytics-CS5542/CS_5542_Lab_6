import json
import traceback
import pandas as pd
from typing import Dict, Any, List

from scripts.sf_connect import get_conn
from backend.rag_pipeline import run_hybrid_query

def run_snowflake_query(sql_query: str) -> str:
    """
    Executes a given SQL query against the Snowflake database.
    Use this tool to answer quantitative questions or fetch specific data points from the tables.
    Returns the result as a JSON string.
    
    Args:
        sql_query (str): The SQL SELECT query to execute.
        
    Returns:
        str: JSON string containing the list of rows (as dictionaries) returned by the query, or an error message.
    """
    print(f"[TOOL LOG] Executing run_snowflake_query: {sql_query}")
    try:
        conn = get_conn()
        df = pd.read_sql(sql_query, conn)
        conn.close()
        # Convert to JSON records
        return df.to_json(orient="records", date_format="iso")
    except Exception as e:
        error_msg = f"Error executing query: {str(e)}\n{traceback.format_exc()}"
        print(f"[TOOL ERROR] {error_msg}")
        return json.dumps({"error": str(e)})

def get_database_schema(table_name: str) -> str:
    """
    Retrieves the database schema (columns and data types) for a specified table in Snowflake.
    Use this tool BEFORE writing a SQL query to ensure you use the correct column names.
    The primary table available is 'APP.CHUNKS_V' (or just 'CHUNKS_V').
    
    Args:
        table_name (str): The name of the table to inspect.
        
    Returns:
        str: JSON string containing the column names and their data types, or an error message.
    """
    print(f"[TOOL LOG] Executing get_database_schema for: {table_name}")
    # Prevent SQL injection in the simple schema query by stripping quotes and taking the last part
    clean_table = table_name.replace("'", "").replace('"', '').split('.')[-1].upper()
    sql_query = f"""
    SELECT COLUMN_NAME, DATA_TYPE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = '{clean_table}'
    ORDER BY ORDINAL_POSITION;
    """
    try:
        conn = get_conn()
        df = pd.read_sql(sql_query, conn)
        conn.close()
        if df.empty:
            return json.dumps({"error": f"No schema found for table {clean_table}"})
        return df.to_json(orient="records")
    except Exception as e:
        error_msg = f"Error retrieving schema: {str(e)}\n{traceback.format_exc()}"
        print(f"[TOOL ERROR] {error_msg}")
        return json.dumps({"error": str(e)})

def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """
    Searches the RAG knowledge base for text chunks related to a query.
    Use this tool to answer qualitative literature questions or questions about the research papers.
    
    Args:
        query (str): The search phrase or question to look up in the documents.
        top_k (int, optional): The number of chunks to retrieve. Defaults to 5.
        
    Returns:
        str: JSON string containing the retrieved chunks, citations, and scores.
    """
    print(f"[TOOL LOG] Executing search_knowledge_base for: '{query}' (top_k={top_k})")
    try:
        res = run_hybrid_query(question=query, top_k=top_k)
        
        # We simplify the output a bit to fit inside agent context context windows easily
        simplified_evidence = []
        for e in res.get("evidence", []):
            simplified_evidence.append({
                "citation": e.get("citation"),
                "text": e.get("text"),
                "score": e.get("hybrid_score")
            })
            
        result = {
            "answer": res.get("answer"),
            "evidence": simplified_evidence,
            "missing_evidence_behavior": res.get("missing_evidence_behavior")
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        error_msg = f"Error searching knowledge base: {str(e)}\n{traceback.format_exc()}"
        print(f"[TOOL ERROR] {error_msg}")
        return json.dumps({"error": str(e)})
