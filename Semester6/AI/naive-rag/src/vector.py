from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("shop-product-catalog.csv")
emeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = '/home/mary/uni/ai/rag-api/ShopAssistant/data'
add_documents= not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i,row in df.iterrows():
        document = Document(
            page_content= row["ProductName"] + " " + row["ProductBrand"],
            metadata = {"Gender" : row["Gender"],
                        "Price" : row["Price"],
                        "Description" : row["Description"],
                        "Primary Color" : row["PrimaryColor"]},
                         id = str(i)
        )
        ids.append(str(i))
        documents.append(document)


vector_store = Chroma(
    collection_name = "shop-product-catalog",
    persist_directory=db_location,
    embedding_function=emeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs = {"k" : 10})