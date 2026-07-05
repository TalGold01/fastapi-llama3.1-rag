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

# --- Chat Input & API Call ---
if prompt := st.chat_input("Ask a question about your secure documents..."):

    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Display Assistant Response
    with st.chat_message("assistant"):
        try:
            # Hit your local Docker container on port 8000
            response = requests.post(
                "http://localhost:8000/query",
                json={"question": prompt},
                stream=True,
                timeout=60
            )

            if response.status_code == 200:
                def generate_stream():
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            yield chunk

                # st.write_stream creates the real-time typing effect!
                full_response = st.write_stream(generate_stream())

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })
            else:
                st.error(f"API Error: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Failed: Is your Docker backend running on port 8000?")
