# 🧠 Local RAG Agent (ChromaDB + LangChain)

A minimal **Retrieval-Augmented Generation (RAG)** system built with
**LangChain Expression Language (LCEL)**, **ChromaDB**, and **Ollama** for fully local operation.
It answers questions from `.txt` or `.csv` files by retrieving relevant context and generating responses via a local LLM.

---

## 🚀 Features

* 📄 Supports both `.txt` and `.csv` files
* 🧠 Uses **ChromaDB** as a local vector database
* 🔤 Generates **embeddings** with `OllamaEmbeddings`
* 🤖 Answers questions using a local Ollama model (e.g. `gpt-oss:120b-cloud`, `mistral`, `llama3`, etc.)
* ⚙️ Built with **LangChain Expression Language (LCEL)** for composable pipelines

---

## 🧱 Project Structure

```
local-rag-chroma/
│
├── data.txt                # or data.csv
├── local_rag_agent.py      # main RAG agent
├── vectorstore_utils.py    # handles text loading + vectorstore
├── requirements.txt        # dependencies
└── README.md
```

---

## ⚙️ Setup

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Install Ollama and a model

```bash
brew install ollama
ollama pull gpt-oss:120b-cloud
```

---

## 🧠 Usage

1. Place your `.txt` or `.csv` file in the project folder.
2. Run the agent:

   ```bash
   python local_rag_agent.py
   ```
3. Ask questions about your data:

   ```
   Ask a question: What is LangChain?
   Answer: LangChain is a framework for building applications with large language models.
   ```

---

## 🧩 How It Works

1. Loads your file and splits text into chunks.
2. Embeds chunks using OllamaEmbeddings embeddings.
3. Stores embeddings locally in ChromaDB.
4. Retrieves top matches for your question.
5. Generates an answer with Ollama LLM.

---

