import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

class IndexingModule:
    """Module 1: Responsible for split/organization of product data [4]."""
    def __init__(self, csv_path, db_dir):
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        self.db_dir = db_dir
        self.vector_store = self._build_index(csv_path)

    def _build_index(self, csv_path):
        if not os.path.exists(self.db_dir):
            df = pd.read_csv(csv_path)
            documents = []
            for i, row in df.iterrows():
                # Metadata Attachment: Enriches chunks with context like Price/Gender 
                doc = Document(
                    page_content=f"{row['ProductName']} {row['ProductBrand']}",
                    metadata={
                        "gender": row["Gender"], "price": row["Price"],
                        "description": row["Description"], "color": row["PrimaryColor"]
                    }
                )
                documents.append(doc)
            return Chroma.from_documents(
                documents=documents, embedding=self.embeddings, persist_directory=self.db_dir
            )
        return Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)

    def get_retriever(self):
        # Operator: Sparse/Dense/Hybrid selection can happen here 
        return self.vector_store.as_retriever(search_kwargs={"k": 10})