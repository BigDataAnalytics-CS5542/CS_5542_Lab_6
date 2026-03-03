import streamlit as st

st.set_page_config(page_title="Research Assistant", layout="wide")

st.title("Research Assistant \U0001f9ec")
st.markdown("Welcome to your personalized research assistant powered by **Snowflake** and **Knowledge Graphs**.")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your research papers..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("assistant"):
        # Placeholder for backend integration
        response = "This is a dummy response. In the future, this will hit `backend/retrieval.py`."
        st.markdown(response)
        
        # Citation Display
        with st.expander("View Citations & Confidence"):
            st.write("**Confidence Score:** 0.95")
            st.write("- Chunk ID: `CHK-123` from Paper `GraphFlow`")
            st.write("- Node: `Retrieval-Augmented Generation`")
            
    st.session_state.messages.append({"role": "assistant", "content": response})

