from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import initialize_vector_store

model = OllamaLLM(model="llama3.1:8b")
retriever = initialize_vector_store()

# PRE-RETRIEVAL: Query Rewriting
# Aims to make queries clear and specific to improve recall
rewrite_template = """Rewrite the following user shopping query to be more descriptive 
for a search engine, focusing on product type, brand, and features:
Query: {question}
Rewritten Query:"""
rewrite_prompt = ChatPromptTemplate.from_template(rewrite_template)

# POST-RETRIEVAL: Reranking 
# Helps the LLM identify key information by reordering chunks based on relevance 
rerank_template = """On a scale of 1-10, how relevant is this product to the user's query?
Query: {question}
Product: {product_info}
Score:"""
rerank_prompt = ChatPromptTemplate.from_template(rerank_template)

#  GENERATION
gen_template = """You are an expert shop assistant. Use the following RERANKED products 
to answer the question. If you don't know, say you don't have that in stock.

Relevant Products:
{context}

Question: {question}
Answer:"""
gen_prompt = ChatPromptTemplate.from_template(gen_template)

def advanced_rag_pipeline(question):
    # 1. Query Rewriting (Pre-Retrieval)
    rewritten_query = (rewrite_prompt | model).invoke({"question": question})
    print(f"DEBUG: Rewritten Query: {rewritten_query}")

    # 2. Retrieval
    docs = retriever.invoke(rewritten_query)

    # 3. Model-Based Reranking (Post-Retrieval)
    # We score each retrieved document and sort them
    scored_docs = []
    for doc in docs:
        score_response = (rerank_prompt | model).invoke({
            "question": question, 
            "product_info": doc.metadata['full_context']
        })
        try:
            # Simple extraction of the first digit found in response
            score = int(''.join(filter(str.isdigit, score_response)))
        except:
            score = 0
        scored_docs.append((score, doc))
    
    # Sort docs by score descending
    scored_docs.sort(key=lambda x: x, reverse=True)
    top_docs = [doc for score, doc in scored_docs[:5]] # Keep top 5 after reranking

    # 4. Generation
    context = "\n\n".join([d.metadata['full_context'] for d in top_docs])
    result = (gen_prompt | model).invoke({"context": context, "question": question})
    return result

# Execution Loop
while True:
    user_input = input("\nHow can I help you find a product? (q to quit): ")
    if user_input.lower() == "q": break
    
    response = advanced_rag_pipeline(user_input)
    print(f"\nAssistant: {response}")