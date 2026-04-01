import streamlit as st
import threading
import queue
import subprocess
import time
import re
import os

from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# =============================
# CONFIG
# =============================
PERSIST_DIR = "./chroma_db"
UPLOAD_DIR = "./uploads"
EMBED_MODEL = "nomic-embed-text"
MODEL_NAME = "llama-3.3-70b-versatile"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# =============================
# TTS CONTROLLER
# =============================
class SyncTTSController:
    def __init__(self):
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.process = None
        self.is_speaking = False
        self.resume_stack = []
        self.remaining_text = ""
        self.lock = threading.Lock()

        threading.Thread(target=self._worker, daemon=True).start()

    def speak(self, text):
        if text.strip():
            self.queue.put(text)

    def interrupt(self):
        with self.lock:
            self.stop_event.set()

            if self.process:
                try:
                    self.process.terminate()
                except:
                    pass

            if self.remaining_text.strip():
                self.resume_stack.append(self.remaining_text)

            self.is_speaking = False

    def _worker(self):
        while True:
            text = self.queue.get()

            with self.lock:
                self.remaining_text = text
                self.is_speaking = True

            if self.stop_event.is_set():
                self.stop_event.clear()
                continue

            try:
                self.process = subprocess.Popen(
                    ["python", "-c", f"""
import pyttsx3
engine = pyttsx3.init()
engine.say({repr(text)})
engine.runAndWait()
"""]
                )

                while self.process.poll() is None:
                    if self.stop_event.is_set():
                        self.process.terminate()
                        break
                    time.sleep(0.01)

            except:
                pass

            with self.lock:
                self.is_speaking = False
                self.remaining_text = ""

            # LIFO resume
            if self.queue.empty() and self.resume_stack:
                self.queue.put(self.resume_stack.pop())


# =============================
# CACHE
# =============================
@st.cache_resource
def get_tts():
    return SyncTTSController()


@st.cache_resource
def get_vectorstore():
    embedding = OllamaEmbeddings(model=EMBED_MODEL)
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding
    )


# =============================
# PDF PROCESSING
# =============================
def process_pdf(file, vs):
    filepath = os.path.join(UPLOAD_DIR, file.name)

    with open(filepath, "wb") as f:
        f.write(file.getbuffer())

    loader = PyPDFLoader(filepath)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    vs.add_documents(chunks)
    vs.persist()


# =============================
# RETRIEVAL
# =============================
def retrieve_context(vs, query):
    docs = vs.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])


# =============================
# GROQ STREAM
# =============================
def stream_llm(query, context):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Answer based on context."},
            {"role": "user", "content": f"{context}\n\n{query}"}
        ],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# =============================
# UI
# =============================
st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")
st.title("⚡ Hybrid RAG Assistant (PDF + Voice + Interrupt)")

tts = get_tts()
vs = get_vectorstore()

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# PDF Upload
# -----------------------------
st.sidebar.header("📄 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        process_pdf(uploaded_file, vs)
    st.sidebar.success("PDF indexed successfully!")

# -----------------------------
# Chat History
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------
# Chat Input
# -----------------------------
query = st.chat_input("Ask from your PDF...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        placeholder = st.empty()

        full_text = ""
        buffer = ""

        context = retrieve_context(vs, query)

        for token in stream_llm(query, context):
            full_text += token
            buffer += token

            placeholder.markdown(full_text)

            sentences = re.split(r'(?<=[.!?])\s+', buffer)

            for s in sentences[:-1]:
                tts.speak(s)

            buffer = sentences[-1]

        if buffer.strip():
            tts.speak(buffer)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_text}
        )

# -----------------------------
# Raise Hand Button
# -----------------------------
if tts.is_speaking:
    if st.button("✋ Raise Hand (Interrupt)"):
        tts.interrupt()
        st.warning("Interrupted! Will resume automatically.")