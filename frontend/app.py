import os
import sys
import streamlit as st

# Add parent directory to sys.path to import agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent import ResearchAgent

st.set_page_config(page_title="Research Assistant", layout="wide")

st.title("Research Assistant \U0001f9ec")
st.markdown("Welcome to your personalized research assistant powered by **Snowflake** and **Knowledge Graphs**.")

# ----------------
# Sidebar / Setup
# ----------------
with st.sidebar:
    st.header("Settings")
    passcode = st.text_input("Snowflake MFA Passcode", type="password")
    
    if passcode:
        if "agent" not in st.session_state or st.session_state.passcode != passcode:
            with st.spinner("Connecting to Snowflake and initializing Agent..."):
                try:
                    # In case get_conn needs to be initialized, ResearchAgent handles it.
                    st.session_state.agent = ResearchAgent(passcode=passcode)
                    # Init prefetch
                    st.session_state.agent._prefetch_chunks()
                    st.session_state.passcode = passcode
                    st.success("Connected successfully!")
                except Exception as e:
                    st.error(f"Failed to connect: {e}")
    else:
        st.warning("Please enter your Snowflake MFA passcode to connect.")

# ----------------
# Chat Interface
# ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display citations if present in an assistant message
        if message["role"] == "assistant" and "citations" in message and message["citations"]:
            with st.expander("View Citations & Tools Used"):
                if "tools_used" in message and message["tools_used"]:
                    st.write(f"**Tools Used:** {', '.join(message['tools_used'])}")
                
                for i, c in enumerate(message["citations"], 1):
                    title = c.get('title', 'Unknown')
                    section = c.get('section', '')
                    score = c.get('score', 0)
                    chunk_id = c.get('chunk_id', 'N/A')
                    paper_id = c.get('paper_id', 'N/A')
                    st.markdown(f"**[{i}] {title}** (Score: {score:.3f})")
                    st.markdown(f"- **Section**: {section}")
                    st.markdown(f"- **Chunk ID**: `{chunk_id}`, **Paper ID**: `{paper_id}`")
                    st.markdown(f"> {c.get('text', '')[:200]}...")

# Chat Input
if prompt := st.chat_input("Ask a question about your research papers..."):
    # Abort if no agent
    if "agent" not in st.session_state:
        st.error("Please enter your Snowflake MFA passcode in the sidebar first.")
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Agent is working..."):
                try:
                    result = st.session_state.agent.run(prompt)
                    response = result.get("answer", "No answer provided.")
                    citations = result.get("citations", [])
                    tools_used = result.get("tools_used", [])
                    
                    st.markdown(response)
                    
                    if citations:
                        with st.expander("View Citations & Tools Used"):
                            if tools_used:
                                st.write(f"**Tools Used:** {', '.join(tools_used)}")
                            
                            for i, c in enumerate(citations, 1):
                                title = c.get('title', 'Unknown')
                                section = c.get('section', '')
                                score = c.get('score', 0)
                                chunk_id = c.get('chunk_id', 'N/A')
                                paper_id = c.get('paper_id', 'N/A')
                                st.markdown(f"**[{i}] {title}** (Score: {score:.3f})")
                                st.markdown(f"- **Section**: {section}")
                                st.markdown(f"- **Chunk ID**: `{chunk_id}`, **Paper ID**: `{paper_id}`")
                                st.markdown(f"> {c.get('text', '')[:200]}...")
                                
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "citations": citations,
                        "tools_used": tools_used
                    })
                except Exception as e:
                    st.error(f"An error occurred: {e}")
