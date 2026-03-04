import streamlit as st
import requests

BACKEND_URL = "http://localhost:3001/query"

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
        # 1. Prepare the payload based on QueryRequest model
        payload = {
            "question": prompt,  # Assuming 'prompt' is your st.chat_input variable
            "top_k": 5
        }

        try:
            # 2. Make the API call
            res = requests.post(BACKEND_URL, json=payload)
            res.raise_for_status()  # Check for HTTP errors
            
            # 3. Parse the result
            # Note: Your backend returns { result }, which creates a nested structure
            data = res.json()
            # If your backend returns a set/dict like { 'answer': ... }, access it here:
            result = data if 'answer' in data else list(data)[0] 
            
            answer = result.get("answer", "No response received.")
            citations = result.get("citations", [])
            confidence = result.get("confidence", 0.0)

            # 4. Display the AI Answer
            st.markdown(answer)

            # 5. Dynamic Citation Display
            with st.expander("View Citations & Confidence"):
                st.write(f"**Confidence Score:** {confidence}")
                st.write(f"**Latency:** {result.get('latency_ms', 0)}ms")
                
                for cite in citations:
                    st.write(f"---")
                    st.write(f"- **Paper:** {cite['title']}")
                    st.write(f"- **Section:** {cite['section']}")
                    st.write(f"- **Chunk ID:** `{cite['chunk_id']}`")
                    st.caption(f"Snippet: {cite['text']}...")
                    
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
            
        

