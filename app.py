"""
High-Concurrency Hybrid RAG Assistant (Streamlit)

Environment setup (example):
1) python -m venv .venv && source .venv/bin/activate
2) pip install streamlit groq chromadb requests
3) Install and run Ollama locally, then pull embedding model:
   ollama pull nomic-embed-text
4) Export Groq API key:
   export GROQ_API_KEY="your_key"
5) Optional TTS engine on Linux:
   sudo apt-get install espeak
6) streamlit run app.py
"""

from __future__ import annotations

import os
import queue
import re
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chromadb
import requests
import streamlit as st
from groq import Groq


# ----------------------------
# Config
# ----------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_store")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docs")
TOP_K = int(os.getenv("TOP_K", "4"))

# Command template must include {text}. Example (Linux): espeak {text}
TTS_CMD_TEMPLATE = os.getenv("TTS_CMD_TEMPLATE", "espeak {text}")


# ----------------------------
# Utilities
# ----------------------------
def split_complete_sentences(buffer: str) -> Tuple[List[str], str]:
    """Return complete sentence chunks and leftover tail for streaming text."""
    if not buffer.strip():
        return [], buffer

    chunks: List[str] = []
    start = 0
    for match in re.finditer(r"[.!?]+(?:\s+|$)", buffer):
        end = match.end()
        sent = buffer[start:end].strip()
        if sent:
            chunks.append(sent)
        start = end
    return chunks, buffer[start:]


def ollama_embed(texts: List[str]) -> List[List[float]]:
    vectors: List[List[float]] = []
    for text in texts:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=120,
        )
        resp.raise_for_status()
        vectors.append(resp.json()["embedding"])
    return vectors


# ----------------------------
# Sync TTS Controller
# ----------------------------
@dataclass
class TTSJob:
    text: str
    response_id: str


class SyncTTSController:
    """
    Producer-consumer TTS engine with atomic interruption and LIFO resume stack.
    """

    def __init__(self) -> None:
        self.queue: "queue.Queue[Dict]" = queue.Queue()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.resume_stack: List[str] = []  # LIFO
        self.process: Optional[subprocess.Popen] = None
        self.current_sentence = ""

        self.is_speaking = False
        self.current_response_id: Optional[str] = None

        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def enqueue_sentence(self, sentence: str, response_id: str) -> None:
        self.queue.put({"type": "sentence", "job": TTSJob(sentence, response_id)})

    def mark_response_done(self, response_id: str) -> None:
        self.queue.put({"type": "done", "response_id": response_id})

    def push_resume_text(self, text: str) -> None:
        if text and text.strip():
            with self.lock:
                self.resume_stack.append(text.strip())

    def interrupt_now(self) -> None:
        """Atomic interruption: kills active subprocess immediately."""
        self.stop_event.set()
        with self.lock:
            proc = self.process
        if proc and proc.poll() is None:
            proc.terminate()  # OS-level terminate

        # Purge queued speech so hardware can be freed immediately and not continue.
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break

    def _speak_sentence(self, sentence: str) -> None:
        quoted = shlex.quote(sentence)
        cmd = TTS_CMD_TEMPLATE.format(text=quoted)

        with self.lock:
            self.current_sentence = sentence
            self.is_speaking = True
            self.process = subprocess.Popen(
                ["bash", "-lc", cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proc = self.process

        # Poll loop gives sub-100ms interruption reactivity.
        while proc.poll() is None:
            if self.stop_event.is_set():
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                break
            time.sleep(0.03)

        with self.lock:
            self.process = None
            self.current_sentence = ""
            self.is_speaking = False

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item["type"] == "sentence":
                    job: TTSJob = item["job"]
                    self.current_response_id = job.response_id
                    self._speak_sentence(job.text)
                elif item["type"] == "done":
                    # Once latest response finishes, auto-resume oldest interrupted context (LIFO pop).
                    with self.lock:
                        has_backlog = not self.queue.empty()
                        should_resume = (not has_backlog) and bool(self.resume_stack)
                        resume_text = self.resume_stack.pop() if should_resume else None
                    if resume_text:
                        # Resume as a single utterance block.
                        self._speak_sentence(resume_text)
            finally:
                self.queue.task_done()
                if self.stop_event.is_set():
                    self.stop_event.clear()


# ----------------------------
# RAG: Chroma + Ollama Embeddings
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_store():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def upsert_document(text: str, source: str) -> None:
    collection = get_store()
    doc_id = str(uuid.uuid4())
    emb = ollama_embed([text])[0]
    collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[emb],
        metadatas=[{"source": source}],
    )


def retrieve_context(query: str, k: int = TOP_K) -> str:
    collection = get_store()
    q_emb = ollama_embed([query])[0]
    result = collection.query(query_embeddings=[q_emb], n_results=k)
    docs = result.get("documents", [[]])[0]
    if not docs:
        return ""
    return "\n\n".join(docs)


# ----------------------------
# LLM Streaming
# ----------------------------
def stream_answer(messages: List[Dict[str, str]]):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        stream=True,
        temperature=0.2,
    )
    for chunk in completion:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Hybrid RAG Assistant", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background-color: #0e1117; color: #f5f5f5; }
      .block-container { padding-top: 1.6rem; max-width: 1000px; }
      .assistant-bubble { background:#1f2937; padding:0.75rem 1rem; border-radius:12px; }
      .user-bubble { background:#111827; padding:0.75rem 1rem; border-radius:12px; border:1px solid #374151; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_tts() -> SyncTTSController:
    return SyncTTSController()


tts = get_tts()

if "chat" not in st.session_state:
    st.session_state.chat = []
if "stream_buffer" not in st.session_state:
    st.session_state.stream_buffer = [""]  # mutable container to avoid nonlocal issues
if "full_response" not in st.session_state:
    st.session_state.full_response = [""]
if "spoken_cursor" not in st.session_state:
    st.session_state.spoken_cursor = 0
if "active_response_id" not in st.session_state:
    st.session_state.active_response_id = None

st.title("🧠 High-Concurrency Hybrid RAG Assistant")

with st.sidebar:
    st.subheader("Knowledge Base")
    uploaded = st.file_uploader("Upload text file", type=["txt", "md"])
    if uploaded is not None:
        content = uploaded.read().decode("utf-8", errors="ignore")
        upsert_document(content, source=uploaded.name)
        st.success(f"Indexed: {uploaded.name}")

    if tts.is_speaking:
        if st.button("✋ Raise Hand (Interrupt)", type="primary"):
            remaining = st.session_state.full_response[0][st.session_state.spoken_cursor :]
            tts.push_resume_text(remaining)
            tts.interrupt_now()
            st.toast("Interrupted and queued for resume.")

for msg in st.session_state.chat:
    role = msg["role"]
    bubble = "assistant-bubble" if role == "assistant" else "user-bubble"
    with st.chat_message(role):
        st.markdown(f"<div class='{bubble}'>{msg['content']}</div>", unsafe_allow_html=True)

prompt = st.chat_input("Ask something...")
if prompt:
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f"<div class='user-bubble'>{prompt}</div>", unsafe_allow_html=True)

    context = retrieve_context(prompt)
    system = (
        "You are a precise assistant. Use retrieved context when relevant. "
        "If context is missing, say what is uncertain."
    )
    user_msg = f"Context:\n{context or 'No retrieved context.'}\n\nQuestion:\n{prompt}"

    response_id = str(uuid.uuid4())
    st.session_state.active_response_id = response_id
    st.session_state.stream_buffer[0] = ""
    st.session_state.full_response[0] = ""
    st.session_state.spoken_cursor = 0

    with st.chat_message("assistant"):
        ph = st.empty()
        for tok in stream_answer([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ]):
            st.session_state.full_response[0] += tok
            st.session_state.stream_buffer[0] += tok

            sentences, leftover = split_complete_sentences(st.session_state.stream_buffer[0])
            st.session_state.stream_buffer[0] = leftover
            for sent in sentences:
                tts.enqueue_sentence(sent, response_id=response_id)
                st.session_state.spoken_cursor += len(sent) + 1

            ph.markdown(
                f"<div class='assistant-bubble'>{st.session_state.full_response[0]}</div>",
                unsafe_allow_html=True,
            )

        # Flush any remaining fragment at end.
        tail = st.session_state.stream_buffer[0].strip()
        if tail:
            tts.enqueue_sentence(tail, response_id=response_id)
            st.session_state.spoken_cursor += len(tail)
            st.session_state.stream_buffer[0] = ""

        tts.mark_response_done(response_id)

    st.session_state.chat.append({"role": "assistant", "content": st.session_state.full_response[0]})
