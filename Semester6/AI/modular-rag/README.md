# **Modular RAG: Reconfigurable Product Catalog Search**

## **Overview**
This project implements a **Modular RAG** framework, a "LEGO-like" architecture that transcends traditional linear RAG designs. Unlike **Naive RAG** (simple retrieve-then-generate) or **Advanced RAG** (fixed pre/post-processing), this system organizes functionality into **independent modules** and **specialized operators**, allowing for highly flexible and maintainable workflows.

The system specifically addresses common RAG failures such as **"Shallow Understanding of Queries"** and **"Retrieval Redundancy"** by incorporating advanced orchestration, query transformation, and verification loops.

---

## **Project Structure**
The project follows a **three-tier architecture** (L1-L3) to ensure scalability and ease of debugging.

*   **`main.py`**: The entry point of the application. It initializes the modules and invokes the top-level **Orchestration Flow**.
*   **`indexing_module.py`**: Handles **Module A (Indexing)**. It processes the product catalog using **Metadata Attachment** and organizes document chunks into a **Chroma** vector database.
*   **`rag_operators.py`**: Contains the **L3 Operators**—the basic units of operation. This includes operators for **Query Rewriting** (Pre-retrieval), **Selection** (Post-retrieval), and **Verification** (Generation).
*   **`orchestrator.py`**: Implements **Module F (Orchestration)**. It manages the **RAG Flow**, specifically a **Loop Pattern** for **Adaptive (Active) Retrieval**.

---

## **Key Modular Components**
Drawing on the **Modular RAG** framework, this project includes:

1.  **Pre-retrieval (Query Transformation)**: Uses a **Query Rewrite** operator to bridge the semantic gap between raw user questions and the product catalog.
2.  **Post-retrieval (Selection)**: Directly removes irrelevant chunks to combat the **"Lost in the Middle"** effect, where LLMs struggle with long, noisy contexts.
3.  **Adaptive Retrieval Loop**: Instead of a single-pass search, the **Orchestrator** uses a **Judge** module to assess if the generated answer is grounded in facts. If not, it triggers a new retrieval loop.
4.  **Verification Module**: Ensures the final output satisfies the user query and is supported by the retrieved context, significantly reducing **hallucinations**.

---

## **Getting Started**

### **Prerequisites**
*   **Ollama**: Install [Ollama](https://ollama.ai/) and download the required models:
    *   `ollama pull llama3.1:8b` (Generator)
    *   `ollama pull mxbai-embed-large` (Embedder)
*   **Python 3.9+**
*   **Data**: A file named `shop-product-catalog.csv` in the root directory.

### **Installation**
1.  **Install dependencies**:
    ```bash
    pip install langchain-ollama langchain-chroma pandas langchain-core
    ```

2.  **Run the application**:
    ```bash
    python main.py
    ```

---

## **Core RAG Flow Pattern**
This project specifically implements the **Loop Pattern** (Algorithm 7: Active RAG Flow). 
- **Step 1**: Transform the user query for better search precision.
- **Step 2**: Retrieve document chunks based on the optimized query.
- **Step 3**: Filter results using the **Selection** operator to reduce noise.
- **Step 4**: Generate an answer and **Judge** its validity.
- **Step 5**: If the answer fails verification, the system **schedules** a re-retrieval or refinement step.

---

## **Future Extensibility**
Because this framework is modular, you can easily:
*   Add a **Routing Module** to direct queries between a product database and a general knowledge pipeline.
*   Integrate a **Knowledge Graph (KG) Index** to clarify connections between concepts and entities.
*   Implement a **Branching Pattern** to search multiple categories in parallel for comparative queries.
