import json
import os
import re
import tempfile

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sse_starlette.sse import EventSourceResponse

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "google/gemini-2.5-flash"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVAL_K = 4
LLM_TEMPERATURE = 0.3
MIN_CHUNK_LENGTH = 5

PROMPT_TEMPLATE = (
    "Provide a comprehensive answer based on the context.\n\n"
    "Context:{context}\nQuestion:{question}\n\nAnswer:"
)

# -------------------------------------------------------------------
# APP
# -------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = None
llm_chain = None


def _get_llm_chain():
    global llm_chain
    if llm_chain is None:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
        )
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        llm_chain = prompt | llm
    return llm_chain


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 2]


def _build_word_to_sentence_map(words, sentences):
    mapping = []
    sent_idx = 0
    word_pos = 0
    sent_words = [s.split() for s in sentences]
    for _ in range(len(words)):
        while sent_idx < len(sent_words) and word_pos >= len(sent_words[sent_idx]):
            word_pos = 0
            sent_idx += 1
        mapping.append(min(sent_idx, len(sentences) - 1))
        word_pos += 1
    return mapping



# -------------------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------------------
@app.get("/api/status")
def status():
    return {"ready": retriever is not None}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    global retriever
    file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            file_path = tmp.name

        documents = PyMuPDFLoader(file_path).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        ).split_documents(documents)

        valid_chunks = [
            c for c in chunks if len(c.page_content.strip()) > MIN_CHUNK_LENGTH
        ]
        if not valid_chunks:
            return {"error": "No valid text extracted from PDF."}

        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vector_db = Chroma.from_documents(documents=valid_chunks, embedding=embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVAL_K})

        return {"status": "ok", "chunks": len(valid_chunks)}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)


@app.post("/api/ask")
async def ask(request: Request):
    body = await request.json()
    question = body.get("question", "")

    if retriever is None:
        return {"error": "No PDF indexed yet."}

    chain = _get_llm_chain()
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    async def event_generator():
        full_text = ""
        for chunk in chain.stream({"context": context, "question": question}):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_text += token
            yield {"event": "token", "data": json.dumps({"token": token})}

        sentences = _split_sentences(full_text)
        words = full_text.split()
        word_to_sentence = _build_word_to_sentence_map(words, sentences)

        yield {
            "event": "done",
            "data": json.dumps({
                "text": full_text,
                "sentences": sentences,
                "words": words,
                "wordToSentenceMap": word_to_sentence,
            }),
        }

    return EventSourceResponse(event_generator())



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
