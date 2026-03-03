"""
Tool definitions for the CS 5542 Research Assistant agent.

These schemas tell the LLM agent what tools are available,
what each tool does, and what arguments to pass. The agent
uses these to decide which tool(s) to call for a given question.

Format follows the OpenAI/HuggingFace function-calling schema.
"""

TOOL_SCHEMAS = [
    {
        "name": "search_papers",
        "description": (
            "Search research papers using semantic vector similarity. "
            "Use this tool first for any question about research topics, "
            "methods, findings, or concepts. Returns the most relevant "
            "text chunks from the paper corpus."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query, e.g. 'kernel methods for regression'"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return. Default is 5.",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_paper_details",
        "description": (
            "Fetch full metadata for a specific paper by its ID. "
            "Use this tool when you already know a paper_id from search_papers "
            "results and need more details like the full abstract, authors, "
            "or source URL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "Paper ID string e.g. 'arxiv_000042'. Get this from search_papers results."
                }
            },
            "required": ["paper_id"]
        }
    },
    {
        "name": "search_knowledge_graph",
        "description": (
            "Find related entities and relationships in the knowledge graph. "
            "Use this tool when the user asks about connections between concepts, "
            "how topics relate to each other, or what entities co-occur in papers. "
            "Works best with specific scientific terms or method names."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query containing scientific entities to look up, e.g. 'support vector machine kernel'"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max number of relations to return. Default is 10.",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "summarize_context",
        "description": (
            "Generate a grounded natural language answer from retrieved chunks. "
            "Always call this tool LAST after search_papers or search_knowledge_graph "
            "to synthesize retrieved context into a final answer for the user. "
            "Never call this tool without retrieved chunks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The original user question to answer."
                },
                "chunks": {
                    "type": "array",
                    "description": "List of chunk dicts from search_papers. Each must have 'text', 'title', 'section' keys.",
                    "items": {"type": "object"}
                }
            },
            "required": ["question", "chunks"]
        }
    }
]

# ── Tool name → function mapping (used by agent.py) ─────────
from tools import (
    search_papers,
    get_paper_details,
    search_knowledge_graph,
    summarize_context,
)

TOOL_FUNCTIONS = {
    "search_papers":          search_papers,
    "get_paper_details":      get_paper_details,
    "search_knowledge_graph": search_knowledge_graph,
    "summarize_context":      summarize_context,
}