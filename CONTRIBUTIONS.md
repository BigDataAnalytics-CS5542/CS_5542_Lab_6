# Week 6 Contributions: AI Agent Integration

## Project Title:
CS5542 - Lab 6 (Academic RAG Pipeline Agent)

------------------------------------------------------------------------

### Member 1: Rohan Hashmi

**Responsibilities:**
- AI Agent Pipeline and Core Execution Loop
- Google GenAI SDK integration (`backend/agent.py`)
- Formulation of the LLM System Prompt dictating Tool Selection policies
- Error handling integration ensuring seamless LLM re-attempts

**Evidence (PR/commits):**
- Implementation of `backend/agent.py`, managing `client.chats.create` states and tool integration.
- Configured `.env` requirements and updated documentation for API setup.

**Tested:**
- Agent loop execution without failure; verified dynamic tool calling given specific prompts.

------------------------------------------------------------------------

### Member 2: Blake Simpson

**Responsibilities:**
- Agent Tool Interfaces and Python Callable Abstractions
- Wrote `backend/tools.py` pulling logic from Lab 4 & 5
- Configured three primary tools: `run_snowflake_query`, `get_database_schema`, and `search_knowledge_base`
- Developed the three Evaluation Scenarios (Simple, Medium, Complex)
- Recorded the Final Presentation and Demo Video showcasing the AI system

**Evidence (PR/commits):**
- Implementation of `backend/tools.py` with typed arguments and verbose docstrings.
- Drafting of `task4_evaluation_report.md` metrics table.

**Tested:**
- Invoked `run_snowflake_query()` offline to confirm parsing of DataFrame into JSON format.

------------------------------------------------------------------------

### Member 3: Kenneth Kakie

**Responsibilities:**
- Front-End Chat Application Design
- Refactored `app/streamlit_app.py` implementing a tab-based UI separating Lab 5 components from the new chat interface.
- Programmed conversation history (`st.session_state.messages`) and interactive visual cues (`st.chat_message`, `st.spinner`).

**Evidence (PR/commits):**
- Updated `streamlit_app.py` adding full chat support with continuous connection state to `agent.py`.

**Tested:**
- Launched Streamlit locally and tested inputs, verifying correct UI display of agent thinking and markdown responses.
