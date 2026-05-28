import streamlit as st
import pandas as pd
import os
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

#  UI Configuration
st.set_page_config(page_title="Shop Catalog Assistant", layout="wide")
st.title("🛍️ Shop Product Catalog Assistant")
st.markdown("Ask questions about our product catalog powered by **Modular RAG**.")

#  1. Indexing Module Configuration 
# Setting up the persistent directory and models
DB_LOCATION = './data_streamlit'
EMBEDDING_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.1:8b"

@st.cache_resource
def get_vector_store():
    """Initializes or loads the Indexing Module (Vector Database) [2, 5]."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    # Check if database already exists
    if not os.path.exists(DB_LOCATION):
        st.info("No existing index found. Please ensure 'shop-product-catalog.csv' is available.")
        if os.path.exists("shop-product-catalog.csv"):
            df = pd.read_csv("shop-product-catalog.csv")
            documents = []
            for i, row in df.iterrows():
                # Metadata Attachment: Enriching chunks for better context [6]
                doc = Document(
                    page_content=f"{row['ProductName']} {row['ProductBrand']}",
                    metadata={
                        "Gender": row["Gender"], "Price": row["Price"],
                        "Description": row["Description"], "Primary Color": row["PrimaryColor"]
                    }
                )
                documents.append(doc)
            
            # Create and persist the vector database
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=DB_LOCATION,
                collection_name="shop-product-catalog"
            )
            st.success("Product catalog indexed successfully!")
            return vector_store
        else:
            st.error("Source CSV file not found!")
            return None
    else:
        return Chroma(
            collection_name="shop-product-catalog",
            persist_directory=DB_LOCATION,
            embedding_function=embeddings
        )

#  2. Generation Module Setup 
def get_rag_chain():
    """Sets up the Generation Module y = LLM([Dq, q]) [7]."""
    model = OllamaLLM(model=LLM_MODEL)
    template = '''
    You are an expert in answering questions about a shop product catalog.
    Use the following relevant products to provide a helpful answer.

    Relevant Products: {products}

    Question: {question}
    '''
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model

#  3. Streamlit Execution Flow 
vector_store = get_vector_store()

if vector_store:
    # Set up the Retriever Operator [8]
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    chain = get_rag_chain()

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What are you looking for?"):
        # Add user question to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # RAG Flow Execution
        with st.chat_message("assistant"):
            with st.spinner("Searching the catalog..."):
                # Module: Retrieval 
                docs = retriever.invoke(prompt)
                
                # For Interpretability: Show retrieved chunks in an expander 
                with st.expander("View Retrieved Catalog Chunks"):
                    for i, d in enumerate(docs):
                        st.write(f"**Item {i+1}:** {d.page_content}")
                        st.write(f"*Details:* {d.metadata['Description']} - {d.metadata['Price']}")

                # Module: Generation 
                response = chain.invoke({"products": docs, "question": prompt})
                st.markdown(response)
                
        st.session_state.messages.append({"role": "assistant", "content": response})