### **1. `indexing_module.py` (Module: Indexing)**
This file implements the foundation of the system's knowledge base.

*   **`df = pd.read_csv(csv_path)`**: Loads the document repository $D$ from the product catalog.
*   **`doc = Document(...)`**: Creates document chunks $d_i$.
*   **`metadata={...}`**: Implements **Metadata Attachment**. By enriching chunks with details like "gender," "price," and "color," the system allows for filtered retrieval, narrowing the search scope.
*   **`Chroma.from_documents(...)`**: The **Indexing operator** $f_e(\cdot)$. It converts text chunks into vectors $e_i$ and stores them in the vector database $I$.
*   **`vector_store.as_retriever(k=10)`**: Defines the **Retriever** $R(q, D)$. In Modular RAG, this is an independent operator that can be swapped for sparse, dense, or hybrid retrievers.

### **2. `rag_operators.py` (Tier: L3 Operators)**
This file defines the specialized "LEGO bricks" that perform specific tasks within the workflow.

*   **`query_rewrite_op`**: Implements **Query Transformation**. It uses an LLM to rewrite the user's raw query into a keyword-rich version, bridging the **semantic gap** between a poorly worded question and the document chunks.
*   **`selection_op`**: A **Post-retrieval Selection** sub-module. By taking only the top 5 chunks (`docs[:5]`), it combats the **"Lost in the Middle"** effect where LLMs ignore key information in long, noisy contexts.
*   **`generation_op`**: The core **Generation module**. It concatenates the refined context and the original question $[D_q, q]$ for the LLM to synthesize the final answer $y$.
*   **`verify_op`**: A **Verification** sub-module. It acts as a **Judge** to determine if the generated answer is supported by facts, which is essential for minimizing **hallucinations**.

### **3. `orchestrator.py` (Module: Orchestration)**
This module governs the **RAG Flow** ($F$), dynamically selecting steps based on previous outcomes.

*   **`optimized_query = self.ops.query_rewrite_op(user_query)`**: Initiates the **Pre-retrieval** stage to ensure the query is search-friendly.
*   **`while attempts < 2:`**: Implements a **Loop Pattern** specifically for **Adaptive (Active) Retrieval**.
*   **`raw_docs = self.retriever.invoke(...)`**: Executes the **Retrieval** module.
*   **`if self.ops.verify_op(answer):`**: The **Scheduling** component. It uses the verification judge to decide if the output is acceptable or if the system needs to "Do RAG Again" by initiating a new retrieval loop.

### **4. `main.py` (The Entry Point)**
This file manages the "LEGO-like" assembly of the system.

*   **`indexer = IndexingModule(...)`** & **`operators = RAGOperators()`**: Initializes independent modules, making the system highly **reconfigurable**. You can swap these "bricks" without rewriting the orchestration logic.
*   **`manager.execute_flow(question)`**: Triggers the non-linear, modular process that is "controlled by multiple control components for retrieval and generation".

This modular structure enhances **system interpretability and maintainability** because each component is an isolated operator that can be debugged or optimized individually.
