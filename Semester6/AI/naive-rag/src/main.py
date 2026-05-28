from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.1:8b")

template = '''
You are an expert in answering questions about a shop product catalog

Here are some relevant products: {products}

Here is the question to answer: {question}
'''

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n---------------------------")
    question = input("Ask your question(q to quit): ")
    print("\n\n-----------------------------")
    if question =="q": 
        break

    products = retriever.invoke(question)
    result = chain.invoke({"products": products,"question": question})
                      

print(result)