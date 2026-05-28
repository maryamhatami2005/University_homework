# Shop Assistant RAG (Ollama + LangChain + Chroma)

This project is a simple **RAG (Retrieval-Augmented Generation)** chatbot for a **shop product catalog**.  
It:
1. Loads product data from `shop-product-catalog.csv`
2. Creates vector embeddings using **Ollama embeddings** (`mxbai-embed-large`)
3. Stores vectors in **Chroma** (persistent on disk)
4. Retrieves the most relevant products for a user question
5. Uses **Ollama LLM** (`llama3.1:8b`) to generate answers grounded in the retrieved products

---

## Features
-  Persistent vector database via Chroma
-  Embedding model: `mxbai-embed-large` (Ollama)
-  Retriever: top-k similarity search (`k=10`)
-  Interactive CLI chat loop

---


## Prerequisites

1.  **Python:** Ensure you have Python 3.8+ installed.
2.  **Ollama:** You need Ollama installed and running to serve your LLM and embedding models.
    *   Download Ollama from [ollama.ai](https://ollama.ai/).
    *   Follow the installation instructions for your operating system.
3.  **Ollama Models:** Download the required models via the Ollama CLI:
```bash
ollama pull mxbai-embed-large
ollama pull llama3.1:8b
```

 ## How to run the project:

```bash
streamlit run app.py
```