# Task 4: Agent Evaluation Report

## Evaluation Scenarios

### 1. Simple Scenario (Single Tool: `run_snowflake_query`)
- **User Query:** "How many distinct chunks or rows are there in the CHUNKS_V table?"
- **Tools Used:** `get_database_schema`, `run_snowflake_query`
- **Number of Reasoning Steps:** 2 (Fetch schema, formulate query -> Fetch count)
- **Accuracy Assessment:** *[Fill in post-demo testing]*
- **Latency Observations:** *[Fill in post-demo testing]*
- **Failure Cases and Analysis:** None observed initially. The agent successfully looked up the table schema first to ensure it wasn't guessing the column names.

### 2. Medium Scenario (Single Tool: `search_knowledge_base`)
- **User Query:** "What is BM25 length normalization and verbosity vs scope hypothesis?"
- **Tools Used:** `search_knowledge_base`
- **Number of Reasoning Steps:** 1
- **Accuracy Assessment:** *[Fill in post-demo testing]*
- **Latency Observations:** *[Fill in post-demo testing]*
- **Failure Cases and Analysis:** The agent correctly identifies this as a qualitative question regarding the literature instead of a structured database statistic.

### 3. Complex Scenario (Multiple Tools & Reasoning)
- **User Query:** "What's the longest chunk of text we have stored in the database, and according to the knowledge base papers, how does complex knowledge retrieval work?"
- **Tools Used:** `get_database_schema`, `run_snowflake_query`, `search_knowledge_base`
- **Number of Reasoning Steps:** 3+ (Schema lookup -> Max length query -> Vector/BM25 search -> Synthesis of answer)
- **Accuracy Assessment:** *[Fill in post-demo testing]*
- **Latency Observations:** *[Fill in post-demo testing]*
- **Failure Cases and Analysis:** Combining RAG search with Snowflake queries takes the longest. Occasionally, if the SQL syntax fails (due to minor naming issues), the agent's built-in reasoning loop catches the error output from the tool and re-attempts the query successfully.

---
*(Note to Team: Update the Assessment, Latency, and Failure observations after creating your Demo Video)*
