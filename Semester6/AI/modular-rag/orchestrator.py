class Orchestrator:
    """Module 6: Controls the coordination of RAG processes"""
    def __init__(self, indexer, operators):
        self.retriever = indexer.get_retriever()
        self.ops = operators

    def execute_flow(self, user_query):
        print(f"[*] Orchestrating Flow...")
        
        # 1. Pre-retrieval: Transform the query to bridge the semantic gap
        optimized_query = self.ops.query_rewrite_op(user_query)
        
        # 2. Loop Pattern: Adaptive Retrieval/Generation 
        attempts = 0
        while attempts < 2:
            # 3. Retrieval & Post-retrieval Selection 
            raw_docs = self.retriever.invoke(optimized_query)
            refined_docs = self.ops.selection_op(raw_docs)
            context = "\n".join([f"{d.page_content}: {d.metadata['description']}" for d in refined_docs])
            
            # 4. Generation [10].
            answer = self.ops.generation_op(user_query, context)
            
            # 5. Scheduling/Verification: Check if result is acceptable 
            if self.ops.verify_op(answer):
                print("[+] Verified Answer Generated.")
                return answer
            
            print("[-] Hallucination detected. Re-retrying flow...")
            attempts += 1
        
        return answer