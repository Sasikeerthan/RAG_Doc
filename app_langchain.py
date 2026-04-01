import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Updated import
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama # Change 1: Using ChatOllama
import tempfile
import os

# Page config
st.set_page_config(page_title="Pro Local RAG", layout="wide")
st.title("Pro-Grade Local RAG Chatbot")

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file and st.session_state.vector_db is None:
        with st.spinner("Processing PDF... Please wait."):
            # Save and Load
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                file_path = tmp.name
            
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Split
            splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            
            # Embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Change 3: Optimized Vector DB storage in session state
            st.session_state.vector_db = Chroma.from_documents(chunks, embeddings)
            
            os.remove(file_path) # Cleanup
            st.success("✅ PDF Indexed Successfully!")

# Chat Interface
if st.session_state.vector_db:
    # Display Chat History
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    # Chat Input
    if query := st.chat_input("Ask about your document..."):
        # Display User Message
        st.session_state.chat_history.append(("user", query))
        with st.chat_message("user"):
            st.markdown(query)

        # Retrieval
        retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Change 2: Instant Streaming Response
        with st.chat_message("assistant"):
            llm = ChatOllama(model="phi3:mini", temperature=0.3)
            
            prompt = f"""You are a professional assistant. Answer based ONLY on the context.
            Context: {context}
            Question: {query}
            Answer:"""
            
            # Streaming the output
            response_placeholder = st.empty()
            full_response = st.write_stream(llm.stream(prompt))
            
        st.session_state.chat_history.append(("assistant", full_response))

else:
    st.info("👈 Please upload a PDF in the sidebar to start.")