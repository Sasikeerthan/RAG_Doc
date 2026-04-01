import streamlit as st
import tempfile
import os
import urllib.parse
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate

# -------------------------------------------------------------------
# 1. UI CONFIGURATION & STYLING
# -------------------------------------------------------------------
st.set_page_config(page_title="Enterprise Parallel RAG", layout="wide")

st.markdown("""
    <style>
    .stChatMessage { border-radius: 12px; border: 1px solid #30363d; padding: 15px; margin-bottom: 12px; }
    .stButton>button { border-radius: 8px; background-color: #d73a49; color: white; font-weight: bold; width: 100%; border: none; }
    .stButton>button:hover { background-color: #cb2431; }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------------
# 2. BROWSER AUDIO DAEMON (Memory Persistent & Parallel)
# -------------------------------------------------------------------
def initialize_audio_daemon():
    """
    Initializes a persistent background audio engine in the browser.
    It queues sentences, tracks exact word boundaries, and handles interruptions.
    """
    daemon_js = """
    <script>
    var pWin = window.parent;
    if (!pWin.audioEngine) {
        pWin.audioEngine = {
            queue: [],
            interruptedStack: [],
            currentUtterance: "",
            lastCharIndex: 0,
            isSpeaking: false,

            // Plays the next item in the queue, or resumes interrupted context
            playNext: function() {
                if (pWin.audioEngine.queue.length === 0) {
                    pWin.audioEngine.isSpeaking = false;
                    // Auto-Resume logic: If queue is empty, check interrupted stack
                    if (pWin.audioEngine.interruptedStack.length > 0) {
                         var resumeText = pWin.audioEngine.interruptedStack.shift();
                         pWin.audioEngine.enqueue(resumeText);
                    }
                    return;
                }
                
                pWin.audioEngine.isSpeaking = true;
                var text = pWin.audioEngine.queue.shift();
                pWin.audioEngine.currentUtterance = text;
                pWin.audioEngine.lastCharIndex = 0;
                
                var msg = new SpeechSynthesisUtterance(text);
                msg.rate = 1.0;
                
                // Track exact character progression
                msg.onboundary = function(e) {
                    pWin.audioEngine.lastCharIndex = e.charIndex;
                };
                
                // Chain to the next sentence seamlessly
                msg.onend = function() {
                    pWin.audioEngine.playNext();
                };
                
                pWin.speechSynthesis.speak(msg);
            },

            // Accepts new text chunks from Python
            enqueue: function(text) {
                pWin.audioEngine.queue.push(text);
                if (!pWin.audioEngine.isSpeaking) {
                    pWin.audioEngine.playNext();
                }
            },

            // Halts audio and calculates the exact remaining string
            interrupt: function() {
                pWin.speechSynthesis.cancel();
                pWin.audioEngine.isSpeaking = false;
                
                var savedContext = [];
                
                // 1. Save the unread portion of the CURRENT sentence
                if (pWin.audioEngine.currentUtterance) {
                    var remaining = pWin.audioEngine.currentUtterance.substring(pWin.audioEngine.lastCharIndex);
                    if (remaining.trim().length > 2) {
                        savedContext.push(remaining.trim());
                    }
                }
                
                // 2. Append all unplayed sentences currently in the queue
                savedContext = savedContext.concat(pWin.audioEngine.queue);
                
                // 3. Clear active states
                pWin.audioEngine.queue = [];
                pWin.audioEngine.currentUtterance = "";
                
                // 4. Push the compiled remainder to the master resume stack
                if (savedContext.length > 0) {
                    pWin.audioEngine.interruptedStack.push(savedContext.join(" "));
                }
            }
        };
    }
    </script>
    """
    st.components.v1.html(daemon_js, height=0)

def push_to_audio_queue(text_chunk):
    """Safely encodes and sends a text chunk to the JS Daemon."""
    if not text_chunk.strip():
        return
    safe_text = urllib.parse.quote(text_chunk.strip())
    js = f"<script>window.parent.audioEngine.enqueue(decodeURIComponent('{safe_text}'));</script>"
    st.components.v1.html(js, height=0)

def trigger_interruption():
    """Commands the JS Daemon to halt and save state."""
    st.components.v1.html("<script>window.parent.audioEngine.interrupt();</script>", height=0)

# Initialize the daemon silently on load
initialize_audio_daemon()

# -------------------------------------------------------------------
# 3. PARALLEL STREAMING GENERATOR
# -------------------------------------------------------------------
def generate_parallel_stream(llm_chain, context_str, user_query):
    """
    Renders text to the UI token-by-token. 
    Upon detecting a sentence boundary, it dispatches the sentence to the audio queue.
    """
    ui_placeholder = st.empty()
    full_response = ""
    sentence_buffer = ""
    
    for chunk in llm_chain.stream({"context": context_str, "question": user_query}):
        token = chunk.content
        full_response += token
        sentence_buffer += token
        
        # Live visual typing effect
        ui_placeholder.markdown(full_response + " ▌")
        
        # Boundary Detection (End of sentence)
        if any(punctuation in token for punctuation in ['.', '!', '?', '\n']):
            if len(sentence_buffer.strip()) > 3:
                push_to_audio_queue(sentence_buffer)
            sentence_buffer = "" 

    # Finalize UI render
    ui_placeholder.markdown(full_response)
    
    # Push any remaining trailing text
    if sentence_buffer.strip():
        push_to_audio_queue(sentence_buffer)
        
    return full_response

# -------------------------------------------------------------------
# 4. TECHNICAL RAG PIPELINE
# -------------------------------------------------------------------
def build_knowledge_base(uploaded_file):
    """Processes the PDF and constructs the local Chroma vector store."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            file_path = tmp.name
        
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(documents)
        
        # Sanitization against empty embeddings
        valid_chunks = [c for c in chunks if len(c.page_content.strip()) > 5]
        
        if not valid_chunks:
            st.error("Document parsing failed: No valid text extracted.")
            return None

        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
        vector_db = Chroma.from_documents(documents=valid_chunks, embedding=embeddings)
        os.remove(file_path)
        return vector_db.as_retriever(search_kwargs={"k": 4})
    except Exception as error:
        st.error(f"Vector Store Error: {error}")
        return None

# -------------------------------------------------------------------
# 5. MAIN APPLICATION
# -------------------------------------------------------------------
st.title("🎙️ Enterprise Sync RAG")
st.caption("True parallel streaming. Halts instantly. Resumes from the exact word.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Configuration")
    selected_model = st.selectbox("Intelligence Engine", ["llama3.2", "mistral", "phi3"])
    document_upload = st.file_uploader("Upload Knowledge Base", type=["pdf"])
    
    if document_upload and "retriever" not in st.session_state:
        with st.spinner("Compiling Vector Store..."):
            st.session_state.retriever = build_knowledge_base(document_upload)
        if st.session_state.retriever:
            st.success("System Operational.")

    st.divider()
    
    # THE INTERRUPTION TRIGGER
    if st.button("✋ HALT & SAVE CONTEXT"):
        trigger_interruption()
        st.error("Process terminated. State preserved for automatic resumption.")

# Render Conversation History
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# User Interaction Flow
if current_query := st.chat_input("Enter technical query..."):
    if "retriever" not in st.session_state:
        st.warning("Knowledge base required. Please upload a PDF.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": current_query})
        with st.chat_message("user"):
            st.markdown(current_query)

        # Context Extraction
        retrieved_docs = st.session_state.retriever.invoke(current_query)
        context_data = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Parallel Generation
        with st.chat_message("assistant"):
            llm_engine = ChatOllama(model=selected_model, temperature=0.3, base_url="http://localhost:11434")
            prompt_config = PromptTemplate.from_template(
                "Provide a comprehensive answer based on the context.\n\nContext:{context}\nQuestion:{question}\n\nAnswer:"
            )
            llm_chain = prompt_config | llm_engine
            
            # Execute parallel generation
            final_output = generate_parallel_stream(llm_chain, context_data, current_query)
            
        st.session_state.chat_history.append({"role": "assistant", "content": final_output})