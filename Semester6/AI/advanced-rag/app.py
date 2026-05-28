import streamlit as st
import os
import pandas as pd
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#  1. CORE LOGIC: ADVANCED RAG COMPONENTS 

def setup_indexing(df, db_path):
    """Indexing Module with Metadata Attachment [4]"""
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    documents = []
    for i, row in df.iterrows():
        # Metadata Attachment helps with filtered retrieval
        doc = Document(
            page_content=f"Product: {row['ProductName']}. Brand: {row['ProductBrand']}. {row['Description']}",
            metadata={"price": row["Price"], "color": row["PrimaryColor"], "id": i}
        )
        documents.append(doc)
    
    return Chroma.from_documents(
        documents=documents, 
        embedding=embeddings, 
        persist_directory=db_path
    )

class AdvancedRAGPipeline:
    def __init__(self, vector_store):
        self.llm = OllamaLLM(model="llama3.1:8b")
        self.vector_store = vector_store
        self.parser = StrOutputParser()

    def pre_retrieval_rewrite(self, question):
        """Pre-retrieval: Bridges the semantic gap via Query Transformation [5]"""
        template = "Rewrite this product query to be specific and keyword-rich: {question}"
        prompt = ChatPromptTemplate.from_template(template)
        return (prompt | self.llm | self.parser).invoke({"question": question})

    def post_retrieval_selection(self, docs):
        """Post-retrieval: Selection to remove noise and anti-fact chunks [3, 6]"""
        return docs[:3] # Reducing noise to prevent hallucination 

    def generate_answer(self, question, context_docs):
        """Generation: Final response synthesis [8]"""
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        template = "Use this context to answer accurately: {context}\nQuestion: {question}"
        prompt = ChatPromptTemplate.from_template(template)
        return (prompt | self.llm | self.parser).invoke({"context": context_text, "question": question})

# 2. STREAMLIT UI

st.set_page_config(page_title="Advanced RAG Assistant", layout="wide")
st.title("🚀 Advanced RAG: Optimized Search")
st.markdown("This system optimizes the retrieval phase using **Pre-retrieval Rewriting** and **Post-retrieval Selection** [1].")

with st.sidebar:
    st.header("Settings")
    db_dir = st.text_input("DB Folder", "./advanced_db")
    uploaded_file = st.file_uploader("Upload Product Catalog (CSV)", type="csv")

if uploaded_file:
    if "rag_pipeline" not in st.session_state:
        with st.status("📦 Indexing Data...", expanded=True):
            df = pd.read_csv(uploaded_file)
            v_store = setup_indexing(df, db_dir)
            st.session_state.rag_pipeline = AdvancedRAGPipeline(v_store)
        st.success("Indexing Complete!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

if user_input := st.chat_input("Ask about products..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if "rag_pipeline" in st.session_state:
        pipeline = st.session_state.rag_pipeline
        with st.chat_message("assistant"):
            with st.status("🛠️ Processing Advanced RAG Steps...", expanded=True):
                # Step 1: Pre-retrieval
                st.write("🔍 **Pre-retrieval**: Rewriting query...")
                opt_query = pipeline.pre_retrieval_rewrite(user_input)
                st.info(f"Optimized Query: {opt_query}")
                
                # Step 2: Retrieval
                st.write("🛰️ **Retrieval**: Searching vector database...")
                raw_docs = pipeline.vector_store.similarity_search(opt_query, k=10)
                
                # Step 3: Post-retrieval
                st.write("✂️ **Post-retrieval**: Selecting top chunks to reduce noise...")
                final_docs = pipeline.post_retrieval_selection(raw_docs)
            
            # Step 4: Generation
            response = pipeline.generate_answer(user_input, final_docs)
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    else:
        st.warning("Please upload a CSV file to start.")