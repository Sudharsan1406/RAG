import streamlit as st
import faiss
import numpy as np
import os
import fitz
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# SETUP
# -----------------------------
nltk.download("punkt")
load_dotenv()  

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None

# -----------------------------
# CSS STYLING
# -----------------------------
st.markdown("""
<style>
.summary-box {
    background-color: #f7f7f7;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #dedede;
    margin-bottom: 20px;
}

.chunk-box {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #e4e4e4;
    margin-bottom: 15px;
}

.chat-container {
    display: flex;
    flex-direction: column;
}

.chat-user {
    background-color: #DCF8C6;
    padding: 12px 16px;
    border-radius: 12px;
    margin: 8px 0;
    max-width: 75%;
    align-self: flex-end;
}

.chat-assistant {
    background-color: #F1F0F0;
    padding: 12px 16px;
    border-radius: 12px;
    margin: 8px 0;
    max-width: 75%;
    align-self: flex-start;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# FUNCTIONS
# -----------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts):
    return embed_model.encode(texts, convert_to_numpy=True)


def summarize_local(docs, length="medium"):
    sentences = []
    for d in docs:
        sentences.extend(sent_tokenize(d["text"]))
    n = {"short": 2, "medium": 5, "long": 10}.get(length, 5)
    return " ".join(sentences[:n])


def make_prompt(docs, length="medium", char_limit=3500):
    header = f"Summarize the following information into a {length} answer.\n"
    header += "Use only the given passages. Do not add new information.\n\n"

    body = ""
    for i, d in enumerate(docs):
        chunk = f"### Passage {i+1}:\n{d['text']}\n\n"
        if len(body) + len(chunk) > char_limit:
            break
        body += chunk

    return header + body + "Write a clear and concise answer."


def summarize_openai(docs, length="medium"):
    if not client:
        return summarize_local(docs, length)

    prompt = make_prompt(docs, length)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise RAG assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return summarize_local(docs, length)


def hybrid_search(query, vectors, metadatas, bm25, index, k=5, alpha=0.5):
    query_vec = embed_model.encode([query], convert_to_numpy=True)[0]

    D, I = index.search(np.array([query_vec]).astype("float32"), k)
    faiss_scores = 1 / (1 + D[0])

    bm25_scores = bm25.get_scores(query.split())
    if np.max(bm25_scores) > 0:
        bm25_scores = bm25_scores / np.max(bm25_scores)

    combined = []
    for i, idx in enumerate(I[0]):
        score = alpha * faiss_scores[i] + (1 - alpha) * bm25_scores[idx]
        combined.append((score, idx))

    combined.sort(reverse=True)

    return [{
        "text": metadatas[idx]["text"],
        "source": metadatas[idx]["source"],
        "score": float(score)
    } for score, idx in combined[:k]]

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("📄 RAG Document Analyzer (OpenAI + Hybrid Search)")
st.write("Upload a PDF and chat with it using RAG.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(raw_text)

    embeddings = embed_texts(chunks).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    metadatas = [{"text": c, "source": "Uploaded PDF"} for c in chunks]
    bm25 = BM25Okapi([c.split() for c in chunks])

    # -----------------------------
    # ONE-SHOT MODE
    # -----------------------------
    st.subheader("🔍 One-shot Question Answering")
    query = st.text_input("Ask a question")

    if st.button("Search"):
        if query.strip():
            with st.spinner("Searching and summarizing..."):
                results = hybrid_search(query, embeddings, metadatas, bm25, index)
                summary = summarize_openai(results, "medium")

            st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)

            st.subheader("📌 Retrieved Chunks")
            for r in results:
                st.markdown(
                    f"<div class='chunk-box'>"
                    f"<b>Source:</b> {r['source']}<br>"
                    f"<b>Relevance:</b> {round(r['score'] * 100, 1)}%<br>"
                    f"{r['text'][:500]}..."
                    f"</div>",
                    unsafe_allow_html=True
                )

    # -----------------------------
    # CHAT MODE
    # -----------------------------
    st.subheader("💬 Chat with your PDF")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input("Type your message")

    if st.button("Send"):
        if user_input.strip():
            with st.spinner("Thinking..."):
                results = hybrid_search(user_input, embeddings, metadatas, bm25, index)
                answer = summarize_openai(results, "short")

            st.session_state.messages.append(("user", user_input))
            st.session_state.messages.append(("assistant", answer))

    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for role, msg in st.session_state.messages:
        if role == "user":
            st.markdown(f"<div class='chat-user'><b>You</b><br>{msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-assistant'><b>Assistant</b><br>{msg}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
