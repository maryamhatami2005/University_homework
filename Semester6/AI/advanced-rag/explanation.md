### **1. Setup and Indexing (`vector.py` logic)**
This section establishes the knowledge base that the system will draw from.

*   **`embeddings = OllamaEmbeddings(model="mxbai-embed-large")`**: This initializes the **embedding model** $f_e(\cdot)$, which is responsible for converting product text into vector representations for storage and similarity search.
*   **`DB_LOCATION = './advanced_rag_db'`**: Defines the persistent directory for the **Vector Database**, where the document embeddings are stored.
*   **`df = pd.read_csv("shop-product-catalog.csv")`**: Loads the source data (the product catalog) to be processed into chunks.

### **2. Pre-Retrieval: Query Rewriting (`main.py`)**
Advanced RAG incorporates pre-processing to bridge the "semantic gap" between user questions and technical product data.

*   **`rewrite_template = """Rewrite the following user shopping query..."""`**: This defines the instructions for the **Query Transformation** sub-module. It instructs the LLM to focus on specific keywords like brand and features.
*   **`rewritten_query = (rewrite_prompt | model).invoke({"question": question})`**: This line executes the transformation. Instead of searching with the raw user input, the system uses a **search-optimized version** to improve the accuracy of the retrieval phase.

### **3. Post-Retrieval: Reranking Logic**
Post-processing ensures that the most relevant information is prioritized, combating the "lost in the middle" effect where LLMs ignore information in long contexts.

*   **`rerank_template = """On a scale of 1-10, how relevant is this product..."""`**: This implements a **Model-based reranker**. Unlike simple vector similarity, it uses the LLM's reasoning to score each retrieved product against the query.
*   **`rerank_prompt = ChatPromptTemplate.from_template(rerank_template)`**: Prepares the scoring instructions to be sent to the LLM for each retrieved chunk.

### **4. Generation and Execution**
The final stage synthesizes an answer using only the refined, high-quality context.

*   **`gen_template = """You are an expert shop assistant. Use the following RERANKED products..."""`**: This is the **Generation module** prompt. It specifically tells the model to use the "RERANKED" products, ensuring the answer is grounded in the most relevant data.
*   **`def advanced_rag_pipeline(question):`**: This function orchestrates the **Linear RAG Flow Pattern**. It follows a fixed sequence: Pre-process (Rewrite) $\rightarrow$ Retrieve $\rightarrow$ Post-process (Rerank) $\rightarrow$ Generate.
*   **`while True: user_input = ... if user_input.lower() == "q": break`**: This is the **Execution Loop**, which keeps the application running until the user decides to quit, allowing for multiple consecutive queries.

By implementing these steps, the code moves beyond **Naive RAG** by actively reducing noise and improving the "Shallow Understanding of Queries" that often leads to poor or hallucinated answers.

