from indexing_module import IndexingModule
from rag_operators import RAGOperators
from orchestrator import Orchestrator

def main():
    # Setup the LEGO-like framework
    indexer = IndexingModule("shop-product-catalog.csv", "./modular_db")
    operators = RAGOperators()
    manager = Orchestrator(indexer, operators)

    while True:
        question = input("\nProduct Catalog Search (q to quit): ")
        if question.lower() == "q": break
        
        # Execute the reconfigurable RAG Flow 
        response = manager.execute_flow(question)
        print(f"\nFinal Response:\n{response}")

if __name__ == "__main__":
    main()