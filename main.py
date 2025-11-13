from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# Initialize the Ollama LLM
model = OllamaLLM(model="gpt-oss:120b-cloud", temperature=0.7)

# Define the prompt template
template = """You are an expert in answering questions about the pizza restaurant 
Here are some reviews: {reviews}
here is the question: {question}
Provide a detailed answer based on the reviews.
"""

# Create a prompt template
prompt = ChatPromptTemplate.from_template(template)

# Define the chain by combining the prompt and model
chain = (prompt | model)

# Initial user query
user_query=input("Ask yor question here (type quit to exit):" )

# Loop to continuously ask for user input until 'quit' is entered
while user_query.lower() != "quit":
    results = chain.invoke({"reviews": [], "question" : user_query})
    print("Answer:", results)
    user_query=input("Ask yor question here (type quit to exit):" )
    
print(results)