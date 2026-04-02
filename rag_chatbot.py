import json
import math
import os
import re
import tempfile
import time

from dotenv import load_dotenv
import streamlit as st

load_dotenv()
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------------------------------------------------
# CONSTANTS
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
WORD_DELAY_MS = 60
TTS_WORDS_PER_SEC = 2.5  # estimated speech rate for pause position calculation

PROMPT_TEMPLATE = (
    "Provide a comprehensive answer based on the context.\n\n"
    "Context:{context}\nQuestion:{question}\n\nAnswer:"
)

# -------------------------------------------------------------------
# PAGE CONFIG & STYLING
# -------------------------------------------------------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.markdown(
    """
    <style>
    .stChatMessage {
        border-radius: 12px;
        border: 1px solid #30363d;
        padding: 15px;
        margin-bottom: 12px;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: bold;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 2]


def _build_word_to_sentence_map(words, sentences):
    """Map each word index to its sentence index."""
    mapping = []
    sent_idx = 0
    word_pos = 0
    sent_words = [s.split() for s in sentences]

    for wi in range(len(words)):
        while sent_idx < len(sent_words) and word_pos >= len(sent_words[sent_idx]):
            word_pos = 0
            sent_idx += 1
        mapping.append(min(sent_idx, len(sentences) - 1))
        word_pos += 1
    return mapping


def _estimate_pause_position(start_time, num_words, num_sentences):
    """Estimate where text reveal and TTS are based on elapsed time."""
    elapsed = time.time() - start_time
    reveal_duration = num_words * WORD_DELAY_MS / 1000.0

    # How many words have been revealed
    reveal_idx = min(num_words, int(elapsed * 1000 / WORD_DELAY_MS))

    # TTS starts immediately alongside reveal. Estimate spoken words.
    # TTS runs in parallel with text reveal. It speaks at ~TTS_WORDS_PER_SEC.
    spoken_words = min(reveal_idx, int(elapsed * TTS_WORDS_PER_SEC))

    # Estimate current sentence based on spoken words (rough)
    # We'll use the word_to_sentence map in the caller
    return reveal_idx, spoken_words


# -------------------------------------------------------------------
# ANSWER PLAYER COMPONENT
#
# Self-contained HTML/JS that handles:
# - Word-by-word text reveal (all words hidden, revealed on timer)
# - TTS starts IMMEDIATELY alongside text reveal (not after)
# - Word-level highlighting via onboundary: spoken=white, pending=gray
# - Raise Hand is a Streamlit button OUTSIDE this iframe
# -------------------------------------------------------------------
_ANSWER_PLAYER_HTML = """
<div id="answer-player" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.8; padding: 10px 0;">
    <div id="text-container" style="font-size: 15px; min-height: 40px;"></div>
</div>
<script>
(function() {
    var p = window.parent;
    var CONFIG = __CONFIG__;
    var words = CONFIG.words;
    var sentences = CONFIG.sentences;
    var wordToSentence = CONFIG.wordToSentenceMap;
    var startSentence = CONFIG.startFromSentence || 0;
    var startWord = CONFIG.startFromWord || 0;
    var wordDelayMs = CONFIG.wordDelayMs || 60;
    var isResume = CONFIG.isResume || false;

    var container = document.getElementById('text-container');

    // Build word spans
    var spans = [];
    for (var i = 0; i < words.length; i++) {
        var span = document.createElement('span');
        span.textContent = words[i] + ' ';
        span.style.transition = 'opacity 0.15s, color 0.15s';
        if (isResume && i < startWord) {
            span.style.opacity = '1';
            span.style.color = '#ffffff';
        } else {
            span.style.opacity = '0';
            span.style.color = '#888888';
        }
        container.appendChild(span);
        spans.push(span);
    }

    // State
    var revealIdx = isResume ? startWord : 0;
    var currentSentenceIdx = startSentence;
    var spokenUpToWord = isResume ? startWord : 0;
    var revealTimer = null;
    var finished = false;
    var ttsSentenceIdx = startSentence;
    var ttsActive = false;

    // Mark player as active on parent (for Raise Hand detection)
    p._ragPlayerActive = true;
    p._ragPlayerDone = false;

    // ---- Text reveal ----
    function revealNext() {
        if (revealIdx >= words.length) return;
        spans[revealIdx].style.opacity = '0.4';
        spans[revealIdx].style.color = '#888888';
        revealIdx++;
        if (revealIdx % 10 === 0 || revealIdx === words.length) {
            container.scrollIntoView({behavior: 'smooth', block: 'end'});
        }
    }

    // ---- Highlight words up to a given word index ----
    function highlightUpTo(wordIdx) {
        for (var i = spokenUpToWord; i <= wordIdx && i < spans.length; i++) {
            spans[i].style.opacity = '1';
            spans[i].style.color = '#ffffff';
        }
        spokenUpToWord = Math.max(spokenUpToWord, wordIdx + 1);
    }

    // ---- Find word range for a sentence ----
    function getSentenceWordRange(sentIdx) {
        var start = -1, end = -1;
        for (var i = 0; i < wordToSentence.length; i++) {
            if (wordToSentence[i] === sentIdx) {
                if (start === -1) start = i;
                end = i;
            }
        }
        return {start: start, end: end};
    }

    // ---- TTS ----
    function speakSentence(sentIdx) {
        if (sentIdx >= sentences.length) {
            // All sentences spoken — highlight everything
            highlightUpTo(words.length - 1);
            finished = true;
            ttsActive = false;
            p._ragPlayerActive = false;
            p._ragPlayerDone = true;
            return;
        }
        ttsSentenceIdx = sentIdx;
        currentSentenceIdx = sentIdx;
        ttsActive = true;

        var range = getSentenceWordRange(sentIdx);

        var u = new SpeechSynthesisUtterance(sentences[sentIdx]);
        u.rate = 1.0;

        u.onboundary = function(e) {
            if (e.name === 'word' && range.start >= 0) {
                var sentText = sentences[sentIdx];
                var spokenSoFar = sentText.substring(0, e.charIndex);
                var wordsSpoken = spokenSoFar.split(/\\s+/).filter(function(w) { return w.length > 0; }).length;
                var targetWord = range.start + wordsSpoken;
                if (targetWord <= range.end) {
                    highlightUpTo(targetWord);
                }
            }
        };

        u.onend = function() {
            if (range.end >= 0) highlightUpTo(range.end);
            speakSentence(sentIdx + 1);
        };

        u.onerror = function() {
            if (range.end >= 0) highlightUpTo(range.end);
            speakSentence(sentIdx + 1);
        };

        p.speechSynthesis.speak(u);
    }

    // ---- Chrome workaround ----
    var chromePoll = setInterval(function() {
        if (finished) { clearInterval(chromePoll); return; }
        if (p.speechSynthesis.paused) {
            p.speechSynthesis.resume();
        }
    }, 250);

    // ---- Start: reveal text AND TTS simultaneously ----
    function start() {
        // Cancel any prior speech
        p.speechSynthesis.cancel();

        // Start text reveal timer
        revealTimer = setInterval(function() {
            revealNext();
            if (revealIdx >= words.length) {
                clearInterval(revealTimer);
                revealTimer = null;
            }
        }, wordDelayMs);

        // Start TTS immediately (runs in parallel with text reveal)
        speakSentence(currentSentenceIdx);
    }

    start();
})();
</script>
"""


def _render_answer_player(text, sentences, words, word_to_sentence,
                          start_sentence=0, start_word=0, is_resume=False):
    """Inject the Answer Player component."""
    config = {
        "fullText": text,
        "words": words,
        "sentences": sentences,
        "wordToSentenceMap": word_to_sentence,
        "startFromSentence": start_sentence,
        "startFromWord": start_word,
        "wordDelayMs": WORD_DELAY_MS,
        "isResume": is_resume,
    }
    html = _ANSWER_PLAYER_HTML.replace("__CONFIG__", json.dumps(config))
    estimated_lines = max(3, len(words) // 12 + 2)
    height = min(800, max(200, estimated_lines * 28 + 60))
    st.components.v1.html(html, height=height)


# -------------------------------------------------------------------
# CANCEL TTS JS — injected on rerun after Raise Hand
# -------------------------------------------------------------------
_CANCEL_TTS_JS = """
<script>
(function() {
    var p = window.parent;
    p.speechSynthesis.cancel();
    p._ragPlayerActive = false;
})();
</script>
"""


# -------------------------------------------------------------------
# STATIC PAUSED TEXT RENDERER
# -------------------------------------------------------------------
def _render_paused_text(words, spoken_up_to_word, reveal_idx):
    """Render paused answer as static HTML preserving highlight state."""
    parts = []
    for i, w in enumerate(words):
        if i < spoken_up_to_word:
            parts.append(f'<span style="color:#ffffff;opacity:1">{w} </span>')
        elif i < reveal_idx:
            parts.append(f'<span style="color:#888888;opacity:0.4">{w} </span>')
        # words beyond reveal_idx stay hidden
    html = (
        '<div style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif; '
        'line-height: 1.8; font-size: 15px; padding: 10px 0;">'
        + "".join(parts)
        + '<div style="text-align:center;margin-top:12px;">'
        '<span style="background:#dc3545;color:#fff;border-radius:8px;'
        'padding:6px 18px;font-size:13px;">'
        '⏸ Paused — will resume after next answer</span></div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# -------------------------------------------------------------------
# LLM RESPONSE FETCHER
# -------------------------------------------------------------------
def _fetch_llm_response(llm_chain, context_str, user_query):
    status = st.empty()
    status.markdown("*Thinking…*")

    response_text = ""
    for chunk in llm_chain.stream({"context": context_str, "question": user_query}):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        response_text += token

    status.empty()
    return response_text


# -------------------------------------------------------------------
# RAG PIPELINE
# -------------------------------------------------------------------
def _build_retriever(uploaded_file):
    file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            file_path = tmp.name

        documents = PyMuPDFLoader(file_path).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        ).split_documents(documents)

        valid_chunks = [
            c for c in chunks if len(c.page_content.strip()) > MIN_CHUNK_LENGTH
        ]
        if not valid_chunks:
            st.error("No valid text extracted from the PDF.")
            return None

        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vector_db = Chroma.from_documents(
            documents=valid_chunks, embedding=embeddings
        )
        return vector_db.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    except Exception as e:
        st.error(f"Failed to process PDF: {e}")
        return None
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)


@st.cache_resource
def _get_llm_chain():
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
    )
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt | llm


# -------------------------------------------------------------------
# SESSION STATE
# -------------------------------------------------------------------
def _init_state():
    defaults = {
        "chat_history": [],
        "_current_file_key": None,
        "_paused_state": None,
        "_resume_after_next": False,
        "_active_answer": None,
        "_play_start_time": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------
_init_state()

# -------------------------------------------------------------------
# Handle Raise Hand button click (from previous rerun)
# -------------------------------------------------------------------
if st.session_state.pop("_do_raise_hand", False):
    active = st.session_state.get("_active_answer")
    start_time = st.session_state.get("_play_start_time")

    if active and start_time:
        words = active["words"]
        sentences = active["sentences"]
        w2s = active["wordToSentenceMap"]

        # Estimate pause position from elapsed time
        reveal_idx, spoken_words = _estimate_pause_position(
            start_time, len(words), len(sentences)
        )
        # Clamp
        reveal_idx = min(reveal_idx, len(words))
        spoken_words = min(spoken_words, len(words))

        # Find which sentence we were on
        if spoken_words > 0 and spoken_words <= len(w2s):
            current_sent = w2s[min(spoken_words, len(w2s) - 1)]
        else:
            current_sent = 0

        st.session_state["_paused_state"] = {
            "text": active["text"],
            "sentences": sentences,
            "words": words,
            "wordToSentenceMap": w2s,
            "currentSentenceIdx": current_sent,
            "spokenUpToWord": spoken_words,
            "revealIdx": reveal_idx,
        }
        st.session_state["_resume_after_next"] = True

        # Replace active answer in history with paused version
        new_history = []
        for msg in st.session_state["chat_history"]:
            if msg.get("_is_active_answer"):
                new_history.append({
                    "role": "assistant",
                    "content": active["text"],
                    "paused": True,
                    "_words": words,
                    "_spoken_up_to_word": spoken_words,
                    "_reveal_idx": reveal_idx,
                })
            else:
                new_history.append(msg)
        st.session_state["chat_history"] = new_history

    st.session_state["_active_answer"] = None
    st.session_state["_play_start_time"] = None

    # Cancel TTS
    st.components.v1.html(_CANCEL_TTS_JS, height=0)

st.title("RAG Chatbot")
st.caption("Streaming text + audio · Raise Hand to pause · auto-resume")

# -- Sidebar --
with st.sidebar:
    st.header("Configuration")
    document_upload = st.file_uploader("Upload PDF", type=["pdf"])

    if document_upload:
        file_key = f"{document_upload.name}-{document_upload.size}"
        if st.session_state["_current_file_key"] != file_key:
            with st.spinner("Building knowledge base..."):
                retriever = _build_retriever(document_upload)
            if retriever:
                st.session_state["retriever"] = retriever
                st.session_state["_current_file_key"] = file_key
                st.session_state["chat_history"] = []
                st.session_state["_paused_state"] = None
                st.session_state["_resume_after_next"] = False
                st.session_state["_active_answer"] = None
                st.session_state["_play_start_time"] = None
                st.rerun()
            else:
                st.error("Failed to index PDF.")

# -- Chat History --
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        if msg.get("paused"):
            _render_paused_text(
                msg["_words"],
                msg["_spoken_up_to_word"],
                msg["_reveal_idx"],
            )
        else:
            st.markdown(msg["content"])

# -- Raise Hand button (visible when an answer is actively playing) --
if st.session_state.get("_active_answer") and st.session_state.get("_play_start_time"):
    if st.button("✋ Raise Hand", type="secondary"):
        st.session_state["_do_raise_hand"] = True
        st.rerun()

# -- Chat Input --
if current_query := st.chat_input("Ask about your document..."):
    if "retriever" not in st.session_state:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state["chat_history"].append(
            {"role": "user", "content": current_query}
        )
        with st.chat_message("user"):
            st.markdown(current_query)

        retrieved_docs = st.session_state["retriever"].invoke(current_query)
        context_data = "\n\n".join(doc.page_content for doc in retrieved_docs)

        with st.chat_message("assistant"):
            llm_chain = _get_llm_chain()
            response_text = _fetch_llm_response(llm_chain, context_data, current_query)

            sentences = _split_sentences(response_text)
            words = response_text.split()
            word_to_sentence = _build_word_to_sentence_map(words, sentences)

            # Store active answer data for Raise Hand
            st.session_state["_active_answer"] = {
                "text": response_text,
                "sentences": sentences,
                "words": words,
                "wordToSentenceMap": word_to_sentence,
            }
            st.session_state["_play_start_time"] = time.time()

            # Render the Answer Player (text + TTS start simultaneously)
            _render_answer_player(response_text, sentences, words, word_to_sentence)

        # Add to chat history
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": response_text, "_is_active_answer": True}
        )

        # If there's a paused answer waiting to resume, render resume player
        if st.session_state.get("_resume_after_next") and st.session_state.get("_paused_state"):
            ps = st.session_state["_paused_state"]

            # Remove paused placeholder from history
            st.session_state["chat_history"] = [
                m for m in st.session_state["chat_history"]
                if not m.get("paused")
            ]

            with st.chat_message("assistant"):
                st.markdown("*Resuming previous answer...*")
                _render_answer_player(
                    ps["text"],
                    ps["sentences"],
                    ps["words"],
                    ps["wordToSentenceMap"],
                    start_sentence=ps["currentSentenceIdx"],
                    start_word=ps["spokenUpToWord"],
                    is_resume=True,
                )

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": ps["text"]}
            )
            st.session_state["_paused_state"] = None
            st.session_state["_resume_after_next"] = False
