import streamlit as st
import requests

# --- Stylized Page Configuration ---
st.set_page_config(
    page_title="Enterprise RAG Interface",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a sharper aesthetic
st.markdown("""
    <style>
    .stApp header {background-color: transparent;}
    .title-text {font-weight: 800; font-size: 2.5rem; letter-spacing: -1px; margin-bottom: 0px;}
    .subtitle-text {color: #a3a8b8; font-size: 1rem; margin-bottom: 2rem;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-text">⚡ Document Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Secure, air-gapped retrieval powered by Llama 3.1</div>', unsafe_allow_html=True)

# --- Chat History State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render Previous Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If the AI provided sources, render them in a sleek expander
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Context Sources"):
                for idx, src in enumerate(message["sources"]):
                    st.info(f"**Source {idx + 1}:** {src}")

# --- Chat Input & API Call ---
if prompt := st.chat_input("Ask a question about your secure documents..."):
    
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 2. Display Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Querying FAISS Database & Llama 3.1..."):
            try:
                # Hit your local Docker container
                response = requests.post(
                    "http://localhost:8000/query", 
                    json={"question": prompt},
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer generated.")
                    sources = data.get("source_documents", [])
                    
                    # Show the answer
                    st.markdown(answer)
                    
                    # Show the sources in an expander
                    if sources:
                        with st.expander("🔍 View Context Sources"):
                            for idx, src in enumerate(sources):
                                st.info(f"**Source {idx + 1}:** {src}")
                                
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"API Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 Connection Failed: Is your Docker backend running on port 8000?")
