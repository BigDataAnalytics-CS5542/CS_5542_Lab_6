import os
from google import genai
from google.genai import types

from backend.tools import run_snowflake_query, get_database_schema, search_knowledge_base

SYSTEM_PROMPT = """
You are a highly capable AI Assistant integrated into a data analytics system for the CS 5542 course.
You have access to two primary data sources:
1. A Snowflake Data Warehouse containing tabular statistics and metadata about chunked documents.
2. A Knowledge Base (Vector + BM25 hybrid search) built from a corpus of research papers (e.g., about GraphFlow, RAG, etc.).

When the user asks a question, determine which data source is appropriate:
- If the question requires aggregations (e.g., "average chunk length", "count of documents") or inspecting tabular properties, formulate a SQL query.
  IMPORTANT: ALWAYS call the `get_database_schema` tool FIRST to learn the columns available before you try to write a SQL query for `run_snowflake_query`. 
  The primary view is `APP.CHUNKS_V`.
- If the question asks to explain a concept, find information from the research papers, or interpret text, call the `search_knowledge_base` tool to retrieve relevant text chunks from the literature.

You may use multiple tools in sequence.
If a tool throws an error, read the error message and adjust your approach.
Your final answer must be comprehensive, directly addressing the user's query using the information retrieved by the tools.
"""

def get_agent_client():
    """Initializes and returns a configured GenAI client and model identifier."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment or .env file.")
    
    client = genai.Client(api_key=api_key)
    # Using Gemini 2.5 Flash as the standard tool-calling capable model
    model = "gemini-2.5-flash"
    
    return client, model

def create_agent_session():
    """
    Creates a chat session loaded with the system instructions and tools.
    """
    client, model = get_agent_client()
    
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[run_snowflake_query, get_database_schema, search_knowledge_base],
        temperature=0.1,  # Lower temperature for more deterministic tool use
    )
    
    chat = client.chats.create(model=model, config=config)
    return chat
