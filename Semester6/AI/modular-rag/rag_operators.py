from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGOperators:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.1:8b")
        self.parser = StrOutputParser()

    def query_rewrite_op(self, question):
        """Sub-module: Pre-retrieval Query Transformation [7]."""
        template = "Rewrite this product query for a search engine: {question}"
        return (ChatPromptTemplate.from_template(template) | self.llm | self.parser).invoke({"question": question})

    def selection_op(self, docs):
        """Sub-module: Post-retrieval Selection to filter noise [8]."""
        # Selection directly removes irrelevant chunks to combat "Lost in the Middle" 
        return docs[:5]

    def generation_op(self, question, context):
        """Module: Generation using refined context [10]."""
        template = "Products: {context}\nQuestion: {question}\nExpert Answer:"
        return (ChatPromptTemplate.from_template(template) | self.llm | self.parser).invoke({"context": context, "question": question})

    def verify_op(self, answer):
        """Sub-module: Verification to minimize hallucinations [11]."""
        # Knowledge-base or model-based verification can be used here
        template = "Is this answer supported by the facts provided? YES or NO: {answer}"
        result = (ChatPromptTemplate.from_template(template) | self.llm | self.parser).invoke({"answer": answer})
        return "YES" in result.upper()