from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from vectorstore_utills import build_vectorstore
from langchain.schema.output_parser import StrOutputParser


# Initialize the Ollama LLM
model = OllamaLLM(model="gpt-oss:120b-cloud", temperature=0.7)

# Build the vectorstore retriever
retriever = build_vectorstore("reviews.csv")    

# Define the prompt template
template = """You are a helpful assistant. Answer the question based only on the provided context.
Context: {context}
Question: {question}
If the answer is not found in the context, say "I don't know based on the provided documents."
"""

# Create a prompt template
prompt = ChatPromptTemplate.from_template(template)

# Define the chain by combining the prompt and model
chain = ({"context": retriever, "question": RunnablePassthrough()}
         | prompt
         | model
         | StrOutputParser()
        )

def ask(question: str):
    return chain.invoke(question)


if __name__ == "__main__":
    print("🧠 Local RAG Agent (LCEL) Ready!")
    
    # Loop to continuously ask for user input until 'quit' is entered
    while True:
        query = input("Ask a question (or 'exit'): ")
        if query.lower() == "exit":
            break
        print("\nAnswer:", ask(query), "\n")