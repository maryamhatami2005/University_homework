from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# CONFIGURATION
DB_LOCATION = './advanced_rag_db'
df = pd.read_csv("shop-product-catalog.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def initialize_vector_store():
    add_documents = not os.path.exists(DB_LOCATION)
    
    vector_store = Chroma(
        collection_name="advanced-shop-catalog",
        persist_directory=DB_LOCATION,
        embedding_function=embeddings
    )

    if add_documents:
        documents = []
        for i, row in df.iterrows():
            # ADVANCED TIP: "Small-to-Big" approach
            # We use the Name + Brand for the retrieval 'hit', 
            # but embed the full metadata for the LLM's 'synthesis'
            full_context = f"Product: {row['ProductName']}\nBrand: {row['ProductBrand']}\nDescription: {row['Description']}"
            
            doc = Document(
                page_content=f"{row['ProductName']} {row['ProductBrand']}", # Small chunk for matching
                metadata={
                    "full_context": full_context, # Big chunk for generation
                    "price": row["Price"],
                    "color": row["PrimaryColor"],
                    "gender": row["Gender"]
                }
            )
            documents.append(doc)
        
        vector_store.add_documents(documents)
        print("Advanced Indexing Complete.")
    
    return vector_store.as_retriever(search_kwargs={"k": 15}) # Retrieve more for reranking