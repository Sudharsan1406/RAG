# 📄 RAG Document Analyzer (Hybrid Search + OpenAI)

A **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF documents and interact with them through **question answering and chat**, using a combination of **FAISS vector search**, **BM25 keyword search**, and **OpenAI-powered summarization**.

This project demonstrates a **production-style RAG pipeline** with proper retrieval, ranking, and answer generation — not just a ChatGPT wrapper.

---

## 🚀 Features

- 📂 Upload and analyze **any PDF document**
- 🔍 **Hybrid retrieval** using:
  - FAISS (semantic vector similarity)
  - BM25 (lexical keyword relevance)
- ⚖️ **Normalized relevance scoring (0–100%)**
- 🤖 **OpenAI-based summarization** (with local fallback)
- 💬 **ChatGPT-style chat interface** with memory
- 🧠 Context-aware answers grounded in document content
- 🔐 Secure API key handling using `.env`

---

## 🧠 System Architecture

1. **PDF Ingestion** → Extract raw text using PyMuPDF
2. **Chunking** → Split text into overlapping chunks
3. **Embedding** → Generate embeddings using SentenceTransformers
4. **Indexing** → Store embeddings in FAISS index
5. **Hybrid Search** → Combine FAISS + BM25 scores
6. **Normalization** → Scale scores to [0,1]
7. **RAG Answering** → OpenAI summarizes retrieved chunks
8. **UI Layer** → Streamlit web app (Q&A + Chat)

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit** – Web UI
- **FAISS** – Vector similarity search
- **Sentence-Transformers** – Embedding generation
- **BM25 (rank-bm25)** – Keyword-based retrieval
- **OpenAI API** – Answer generation
- **PyMuPDF (fitz)** – PDF text extraction
- **NLTK** – Sentence tokenization

---

## 📦 Project Structure

```
RAG/
│── app.py              # Streamlit application
│── .env                # OpenAI API key (ignored by git)
│── .gitignore
│── requirements.txt
│── README.md
```

---

## 🔑 Environment Setup

### 1️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # macOS/Linux
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set up OpenAI API key

Create a file named `.env` in the project root:

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
```

⚠️ **Do NOT commit this file to GitHub**

Ensure `.gitignore` contains:
```
.env
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

---

## 🧪 How to Use

1. Upload a PDF document
2. Ask a question in **One-shot Q&A mode** OR
3. Use **Chat mode** to have a conversation with the document
4. View:
   - Generated answer
   - Retrieved chunks
   - Normalized relevance scores

---

## 📊 Relevance Scoring

Each retrieved chunk is scored using:

```
Final Score = α × FAISS_similarity + (1 − α) × BM25_score
```

- FAISS scores → semantic similarity
- BM25 scores → keyword relevance
- Scores are normalized to **0–100%** for clarity

Only ranking matters — higher score = higher relevance.

---

## ❓ Why Hybrid Search?

- **FAISS only** → misses exact keywords
- **BM25 only** → misses semantic meaning

Hybrid search provides:
- Better recall
- More accurate context
- Stronger RAG answers

---

## 🛡️ Security Best Practices

- API keys stored in `.env`
- `.env` excluded via `.gitignore`
- No secrets hardcoded in source code

---

## 🎯 Use Cases

- Document Q&A systems
- Internal knowledge assistants
- Interview / exam preparation
- Research paper analysis
- Enterprise document search

---

## 📈 Possible Enhancements

- Source sentence highlighting
- Clear chat button
- Token/cost monitoring
- Deployment on Streamlit Cloud or Docker

---

## 👤 Author

**Sudharsan M S**  
AI / Data Science Engineer

---

## 📄 License

This project is released under the **MIT License**.

