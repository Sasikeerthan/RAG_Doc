import re
import tempfile
import threading
import time
from typing import List, TypedDict

import pyttsx3
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph


STREAM_DELAY = 0.08
VOICE_RATE = 180
AVERAGE_WORD_LENGTH = 5


class ChatState(TypedDict):
    question: str
    context: str
    answer: str
    history: List[str]


class TTSController:
    def __init__(self):
        self.thread = None
        self.engine = None
        self.stop_event = threading.Event()
        self.is_speaking = False
        self.start_time = None
        self.start_index = 0
        self.text_length = 0


def split_words(text: str) -> List[str]:
    return re.findall(r"\S+\s*", text)


def chars_per_second() -> float:
    return (VOICE_RATE / 60) * AVERAGE_WORD_LENGTH


def init_session_state():
    defaults = {
        "history": [],
        "graph": None,
        "loaded_file_key": None,
        "current_question": "",
        "full_response_text": "",
        "response_words": [],
        "displayed_response": "",
        "stream_index": 0,
        "is_streaming": False,
        "is_speaking": False,
        "current_speech_position": 0,
        "interruption_flag": False,
        "response_saved": False,
        "resume_question": "",
        "resume_response_text": "",
        "resume_speech_position": 0,
        "resume_ready": False,
        "status_text": "",
        "tts": TTSController(),
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def estimate_speech_position() -> int:
    controller = st.session_state.tts

    if not controller.is_speaking or controller.start_time is None:
        return st.session_state.current_speech_position

    elapsed = time.monotonic() - controller.start_time
    estimated_offset = int(elapsed * chars_per_second())
    return min(controller.text_length, controller.start_index + estimated_offset)


def stop_tts():
    controller = st.session_state.tts
    controller.stop_event.set()

    if controller.engine is not None:
        try:
            controller.engine.stop()
        except Exception:
            pass

    controller.is_speaking = False
    st.session_state.is_speaking = False


def start_tts(text: str, start_index: int = 0):
    stop_tts()

    remaining_text = text[start_index:].strip()
    controller = st.session_state.tts

    if not remaining_text:
        controller.is_speaking = False
        st.session_state.is_speaking = False
        st.session_state.current_speech_position = len(text)
        return

    controller.stop_event = threading.Event()
    controller.start_time = time.monotonic()
    controller.start_index = start_index
    controller.text_length = len(text)
    controller.is_speaking = True

    def worker():
        engine = pyttsx3.init()
        engine.setProperty("rate", VOICE_RATE)
        controller.engine = engine

        try:
            engine.say(text[start_index:])
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:
                pass

            controller.engine = None
            controller.is_speaking = False

    controller.thread = threading.Thread(target=worker, daemon=True)
    controller.thread.start()

    st.session_state.is_speaking = True
    st.session_state.current_speech_position = start_index


def clear_current_response():
    st.session_state.current_question = ""
    st.session_state.full_response_text = ""
    st.session_state.response_words = []
    st.session_state.displayed_response = ""
    st.session_state.stream_index = 0
    st.session_state.is_streaming = False
    st.session_state.is_speaking = False
    st.session_state.current_speech_position = 0
    st.session_state.response_saved = False
    st.session_state.status_text = ""


def clear_resume_state():
    st.session_state.resume_question = ""
    st.session_state.resume_response_text = ""
    st.session_state.resume_speech_position = 0
    st.session_state.resume_ready = False


def start_response(question: str):
    result = st.session_state.graph.invoke(
        {
            "question": question,
            "context": "",
            "answer": "",
            "history": st.session_state.history,
        }
    )

    answer = result["answer"]

    st.session_state.current_question = question
    st.session_state.full_response_text = answer
    st.session_state.response_words = split_words(answer)
    st.session_state.displayed_response = ""
    st.session_state.stream_index = 0
    st.session_state.is_streaming = True
    st.session_state.is_speaking = False
    st.session_state.current_speech_position = 0
    st.session_state.interruption_flag = False
    st.session_state.response_saved = False
    st.session_state.status_text = "Streaming response..."


def save_current_response():
    if st.session_state.response_saved:
        return

    if st.session_state.current_question and st.session_state.full_response_text:
        st.session_state.history.extend(
            [
                f"User: {st.session_state.current_question}",
                f"Assistant: {st.session_state.full_response_text}",
            ]
        )

    st.session_state.response_saved = True


def begin_speaking_current_response(status_text: str):
    st.session_state.status_text = status_text
    start_tts(
        st.session_state.full_response_text,
        st.session_state.current_speech_position,
    )


def resume_previous_speech():
    if not st.session_state.resume_ready:
        clear_current_response()
        return

    st.session_state.current_question = st.session_state.resume_question
    st.session_state.full_response_text = st.session_state.resume_response_text
    st.session_state.response_words = split_words(st.session_state.resume_response_text)
    st.session_state.displayed_response = st.session_state.resume_response_text
    st.session_state.stream_index = len(st.session_state.response_words)
    st.session_state.is_streaming = False
    st.session_state.current_speech_position = st.session_state.resume_speech_position
    st.session_state.response_saved = True
    st.session_state.interruption_flag = False

    clear_resume_state()
    begin_speaking_current_response("Resuming previous response...")


def handle_speech_finished():
    st.session_state.is_speaking = False
    st.session_state.current_speech_position = len(st.session_state.full_response_text)

    if st.session_state.resume_ready:
        resume_previous_speech()
    else:
        clear_current_response()


def interrupt_speech():
    st.session_state.current_speech_position = estimate_speech_position()

    if st.session_state.full_response_text:
        st.session_state.resume_question = st.session_state.current_question
        st.session_state.resume_response_text = st.session_state.full_response_text
        st.session_state.resume_speech_position = st.session_state.current_speech_position
        st.session_state.resume_ready = True

    stop_tts()
    clear_current_response()
    st.session_state.interruption_flag = True
    st.session_state.status_text = "Speech interrupted. Ask your next question."


def build_graph(file_path: str):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embed)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = Ollama(model="phi3:mini", base_url="http://localhost:11434")

    def get_context(state: ChatState):
        docs = retriever.invoke(state["question"])

        if not docs:
            return {"answer": "Not in document", "history": state["history"]}

        context = "\n".join(doc.page_content for doc in docs)

        if len(context.strip()) < 50:
            return {"answer": "Not enough information", "history": state["history"]}

        return {"context": context, "history": state["history"]}

    def generate_answer(state: ChatState):
        if state.get("answer"):
            return {"answer": state["answer"], "history": state["history"]}

        history_text = "\n".join(state["history"])

        prompt = f"""
Answer only using the given context.
If answer not found, say 'Not in document'.

History:
{history_text}

Context:
{state['context']}

Question:
{state['question']}
"""

        reply = llm.invoke(prompt)
        return {"answer": reply, "history": state["history"]}

    graph = StateGraph(ChatState)
    graph.add_node("context", get_context)
    graph.add_node("answer", generate_answer)
    graph.set_entry_point("context")
    graph.add_edge("context", "answer")
    return graph.compile()


def render_history():
    for message in st.session_state.history:
        if message.startswith("User:"):
            st.chat_message("user").write(message.replace("User: ", "", 1))
        else:
            st.chat_message("assistant").write(message.replace("Assistant: ", "", 1))


def render_active_response():
    if not st.session_state.current_question:
        return

    if not (st.session_state.is_streaming or st.session_state.is_speaking):
        return

    st.chat_message("user").write(st.session_state.current_question)
    st.chat_message("assistant").write(st.session_state.displayed_response or " ")


def render_status():
    if st.session_state.status_text:
        st.caption(st.session_state.status_text)


@st.fragment(run_every=STREAM_DELAY)
def streaming_panel():
    controller = st.session_state.tts
    was_speaking = st.session_state.is_speaking
    st.session_state.is_speaking = controller.is_speaking

    if st.session_state.is_streaming:
        if st.session_state.stream_index < len(st.session_state.response_words):
            st.session_state.stream_index += 1
            st.session_state.displayed_response = "".join(
                st.session_state.response_words[: st.session_state.stream_index]
            )

        if st.session_state.stream_index >= len(st.session_state.response_words):
            st.session_state.is_streaming = False
            save_current_response()
            st.session_state.displayed_response = st.session_state.full_response_text
            st.session_state.current_speech_position = 0
            begin_speaking_current_response("Speaking response...")

    elif was_speaking and not controller.is_speaking:
        handle_speech_finished()

    elif st.session_state.is_speaking:
        st.session_state.current_speech_position = estimate_speech_position()

    render_history()
    render_active_response()
    render_status()


init_session_state()

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 Chat with your PDF (Streaming + Voice)")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    file_key = f"{uploaded_file.name}-{uploaded_file.size}"

    if st.session_state.loaded_file_key != file_key:
        stop_tts()
        clear_current_response()
        clear_resume_state()
        st.session_state.history = []
        st.session_state.interruption_flag = False

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            pdf_path = temp_file.name

        with st.spinner("Processing PDF..."):
            st.session_state.graph = build_graph(pdf_path)

        st.session_state.loaded_file_key = file_key
        st.success("PDF uploaded and ready.")

if st.session_state.graph:
    _, button_col = st.columns([4, 1])

    with button_col:
        can_interrupt = st.session_state.is_speaking and not st.session_state.interruption_flag
        if st.button("Raise Hand ✋", disabled=not can_interrupt, use_container_width=True):
            interrupt_speech()
            st.rerun()

    streaming_panel()

    input_disabled = (st.session_state.is_streaming or st.session_state.is_speaking) and not st.session_state.interruption_flag
    prompt_text = "Ask something..."
    if st.session_state.interruption_flag:
        prompt_text = "Ask your interrupting question..."

    question = st.chat_input(prompt_text, disabled=input_disabled)

    if question:
        start_response(question)
        st.rerun()
else:
    st.info("Upload a PDF to start.")
