
# Advanced RAG Shop Assistant #

This project implements an **Advanced Retrieval-Augmented Generation (RAG)** pipeline for a product catalog. It moves beyond the traditional "Naive RAG" paradigm to provide more accurate and contextually relevant answers.

## 🧠 Theoretical Background

### Why Advanced RAG?
The early "Naive RAG" paradigm often fails because it relies on simple similarity matching between question and document chunks. This implementation addresses two primary challenges identified in the sources:
1.  **Shallow Understanding:** Semantic similarity alone does not always reflect the deep relationship between a user's query and the data.
2.  **Retrieval Noise:** Feeding too much redundant or irrelevant information into a model increases the risk of "hallucinations".

### Core Advanced Features
To solve these issues, this project adopts a **Modular RAG** approach, treating system components like "LEGO-like bricks" that can be reconfigured.

*   **Query Rewriting (Pre-Retrieval):** The system uses an LLM to rewrite user queries to be "clearer and specific," which bridges the semantic gap and increases retrieval recall.
*   **Small-to-Big Indexing:** This strategy separates the chunks used for retrieval from those used for synthesis. We index "small" chunks (product names) to improve hit accuracy while providing the LLM with "big" chunks (full descriptions) to ensure it has sufficient context for generation.
*   **Model-Based Reranking (Post-Retrieval):** After the initial search, the system uses a language model to score and reorder results based on relevance. This enhances the LLM's ability to "identify and utilize key information" before generating a final answer.

## 🏗 System Architecture

The pipeline follows a **Linear RAG Flow Pattern**, where modules are processed in a fixed sequential order.

1.  **Indexing:** Organizes the shop catalog using a hierarchical metadata approach.
2.  **Pre-Retrieval:** Transforms the original query into a search-optimized version.
3.  **Retrieval:** Uses dense embedding models to find the top $k$ relevant products.
4.  **Post-Retrieval:** Filters and reranks the results to minimize noise.
5.  **Generation:** Synthesizes a response using the original query and the optimized context.

## 🛠 Setup and Usage
1.  **Install dependencies:** `pip install -r requirements.txt`
2.  **Run Ollama:** Ensure the `llama3.1:8b` and `mxbai-embed-large` models are pulled.
3.  **Launch:** Run `python main.py` to start the interactive assistant.

***
*This implementation is based on the framework described in "Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks" by Gao et al.*.

