import streamlit as st
from indexing_module import IndexingModule
from rag_operators import RAGOperators
from orchestrator import Orchestrator

st.set_page_config(page_title="Modular RAG Explorer", layout="wide")
st.title("🧩 Modular RAG: AI Product Assistant")

with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload Catalog (CSV)", type="csv")
    db_dir = st.text_input("Vector DB Path", "./modular_rag_db")

if uploaded_file:
    if "rag_system" not in st.session_state:
        with st.status("🏗️ Building Modular Index...", expanded=True):
            with open("temp_data.csv", "wb") as f:
                f.write(uploaded_file.getbuffer())
            indexer = IndexingModule("temp_data.csv", db_dir)
            ops = RAGOperators()
            st.session_state.rag_system = Orchestrator(indexer, ops)
        st.success("System Ready!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about products..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    if "rag_system" in st.session_state:
        with st.chat_message("assistant"):
            with st.status("🧠 Orchestrating Modular Flow...", expanded=True):
                st.write("🔄 Running Pre-retrieval Transformation...")
                response, opt_query = st.session_state.rag_system.execute_flow(prompt)
                st.write(f"✅ Query Optimized to: *{opt_query}*")
                st.write("🛰️ Executing Adaptive Retrieval Loop...")
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload a CSV file to begin.")