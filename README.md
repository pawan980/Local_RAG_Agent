# Local_RAG_Agent
# 🧠 Local RAG Agent (LCEL)

A **local AI question-answering agent** built using **LangChain Expression Language (LCEL)**.
It can answer user questions based on the contents of a provided file — entirely **offline**, using local models and vector search.

---

## 🚀 Features

* **Retrieval-Augmented Generation (RAG):** Answers are grounded in your document’s content.
* **Local Execution:** Uses `Ollama` + `FAISS` + `HuggingFace` embeddings — no external API calls.
* **Composable LCEL Pipeline:** Fully modular and extensible with LangChain Expression Language.
* **Supports Any Text File:** Plug in `.txt`, `.md`, or even processed PDF text.

---

## 🧩 Architecture Overview

```
        ┌─────────────┐
        │   Document  │
        └──────┬──────┘
               │
               ▼
     ┌───────────────────┐
     │ Text Splitter     │ → chunks for embeddings
     └───────────────────┘
               │
               ▼
     ┌───────────────────┐
     │ Embeddings Model  │ (MiniLM)
     └───────────────────┘
               │
               ▼
     ┌───────────────────┐
     │ Vectorstore (FAISS│
     └───────────────────┘
               │
               ▼
     ┌────────────────────────────────┐
     │ LCEL Chain: Retriever → LLM     │
     └────────────────────────────────┘
```

---

## 🛠️ Setup Instructions

### 1️⃣ Clone & Install

```bash
git clone https://github.com/yourusername/local-rag-agent.git
cd local-rag-agent

pip install -r requirements.txt
```

### 2️⃣ Dependencies

Ensure the following are installed:

```bash
pip install langchain langchain-community langchain-text-splitters faiss-cpu sentence-transformers langchain-ollama
```

### 3️⃣ Install Ollama and Pull a Model

```bash
brew install ollama
ollama pull mistral
```

You can also use `llama3`, `phi3`, or any other supported model.

---

## 📘 Usage

1. Place your reference text file (e.g., `data.txt`) in the root directory.
2. Run the agent:

```bash
python local_rag_agent.py
```

3. Start chatting:

```
🧠 Local RAG Agent (LCEL) Ready!
Ask a question: What is this document about?
```

Type `exit` to quit.

---

## 🧱 Project Structure

```
local-rag-agent/
│
├── data.txt                  # Your knowledge source
├── local_rag_agent.py        # Main RAG agent script
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ How It Works

1. **Load Document:** Text is read and chunked into overlapping segments.
2. **Embed & Index:** Each chunk is embedded using `sentence-transformers` and stored in a FAISS index.
3. **Retrieve:** When you ask a question, top-k similar chunks are fetched.
4. **Generate:** The retrieved chunks + question form a context-aware prompt, passed to the local LLM.
5. **Answer:** The model outputs a grounded, contextually accurate answer.

---

## 🧮 LCEL Chain Definition

```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

Each `|` represents a flow stage — retrieval → prompt → generation → parsing.
This is the core of LangChain Expression Language.

---

## 🧠 Customization

* **Change Model:** Replace `mistral` with `llama3`, `phi3`, etc.
* **Adjust Chunk Size:** Tune `chunk_size` and `chunk_overlap` for optimal context.
* **Add Memory:** Use `ConversationBufferMemory` for contextual conversations.
* **Add Persistence:** Swap FAISS with ChromaDB for saved indexes.

---

## 📈 Future Improvements

* PDF & Markdown ingestion
* Web UI (Streamlit or Gradio)
* LangGraph integration for multi-tool reasoning
* Persistent vectorstore cache

---

> “RAG turns language models into knowledge models — grounded, truthful, and local.”
