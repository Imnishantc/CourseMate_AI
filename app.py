import hashlib
import logging
import os
import uuid
from tempfile import NamedTemporaryFile

import streamlit as st
from dotenv import load_dotenv
from create_database import create_vectorstore
from history_store import (
    add_message,
    clear_conversation,
    create_conversation,
    get_conversation,
    get_messages,
    list_conversations,
)
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

MAX_UPLOAD_SIZE_MB = 20
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MAX_CHAT_CONTEXT_PAIRS = 10


def extract_response_text(content):
    """Return readable text from Gemini's structured content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block if isinstance(block, str) else block["text"]
            for block in content
            if isinstance(block, str)
            or (isinstance(block, dict) and isinstance(block.get("text"), str))
        )
    return str(content)


def format_retrieved_context(docs, fallback_name):
    """Add page provenance to retrieved text and collect unique citations."""
    context_parts, sources = [], []
    for doc in docs:
        page = doc.metadata.get("page")
        page_label = str(page + 1) if isinstance(page, int) else "unknown"
        source = f"{doc.metadata.get('source', fallback_name)}, page {page_label}"
        if source not in sources:
            sources.append(source)
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(context_parts), sources


def collection_for_digest(digest):
    """Use a stable local collection so an existing PDF index can be reused."""
    return f"cm_local_{digest[:32]}"


@st.cache_resource
def load_vectorstore(collection_name):
    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    return Chroma(
        collection_name=collection_name,
        persist_directory="chroma_db",
        embedding_function=embedding_model,
    )


load_dotenv()
st.set_page_config(
    page_title="CourseMate AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1120px; padding-top: 2rem; padding-bottom: 7rem;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(128, 128, 128, .2);}
    [data-testid="stSidebar"] .block-container {padding-top: 1.5rem;}
    .cm-eyebrow {color: #8b91a5; font-size: .78rem; letter-spacing: .12em;
                 text-transform: uppercase; font-weight: 700;}
    .cm-subtitle {color: #8b91a5; margin-top: -.8rem;}
    .cm-empty {border: 1px solid rgba(128, 128, 128, .25); border-radius: 18px;
               padding: 2.2rem; background: rgba(128, 128, 128, .04);}
    </style>
    """,
    unsafe_allow_html=True,
)

if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None
if "pending_upload_digest" not in st.session_state:
    st.session_state.pending_upload_digest = None


def start_new_chat():
    st.session_state.active_conversation_id = None
    st.session_state.pending_upload_digest = None


def select_conversation(conversation_id):
    st.session_state.active_conversation_id = conversation_id
    st.session_state.pending_upload_digest = None


with st.sidebar:
    st.markdown("## 📚 CourseMate AI")
    st.caption("Your study notes, ready to chat.")
    if st.button("＋  New chat", type="primary", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.divider()
    st.markdown("#### Add course notes")
    uploaded_file = st.file_uploader(
        "Choose a PDF", type=["pdf"], accept_multiple_files=False,
        label_visibility="collapsed",
    )
    uploaded_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
    uploaded_digest = hashlib.sha256(uploaded_bytes).hexdigest() if uploaded_bytes else None

    if uploaded_file is not None:
        st.caption(f"📄 {uploaded_file.name}")
        if len(uploaded_bytes) > MAX_UPLOAD_SIZE_BYTES:
            st.error(f"This PDF is over the {MAX_UPLOAD_SIZE_MB} MB limit.")
        elif st.button("Process PDF", use_container_width=True):
            temporary_path = None
            try:
                collection_name = collection_for_digest(uploaded_digest)
                with st.spinner("Preparing your notes…"):
                    with NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_file:
                        temporary_file.write(uploaded_bytes)
                        temporary_path = temporary_file.name
                    existing = load_vectorstore(collection_name)
                    if not existing.get(limit=1).get("ids"):
                        create_vectorstore(temporary_path, collection_name, uploaded_file.name)
                    conversation_id = uuid.uuid4().hex
                    create_conversation(conversation_id, uploaded_digest, uploaded_file.name)
                st.session_state.active_conversation_id = conversation_id
                st.session_state.pending_upload_digest = None
                st.success("Your PDF is ready to chat with.")
                st.rerun()
            except Exception as exc:
                logging.exception("PDF processing failed")
                st.error(f"Could not process this PDF ({type(exc).__name__}). Check the terminal for details.")
            finally:
                if temporary_path and os.path.exists(temporary_path):
                    os.unlink(temporary_path)

    st.divider()
    st.markdown("#### Recent chats")
    conversations = list_conversations()
    if not conversations:
        st.caption("Your conversations will appear here.")
    for conversation in conversations:
        title = conversation["title"].strip() or conversation["document_name"]
        label = f"💬  {title[:32]}" + ("…" if len(title) > 32 else "")
        st.button(
            label,
            key=f"open_{conversation['id']}",
            use_container_width=True,
            type="secondary" if conversation["id"] != st.session_state.active_conversation_id else "primary",
            on_click=select_conversation,
            args=(conversation["id"],),
            help=f"Chat about {conversation['document_name']}",
        )

conversation = get_conversation(st.session_state.active_conversation_id) if st.session_state.active_conversation_id else None

if conversation:
    st.markdown('<div class="cm-eyebrow">CourseMate workspace</div>', unsafe_allow_html=True)
    st.title(conversation["title"] if conversation["title"] != conversation["document_name"] else conversation["document_name"])
    st.markdown(f'<div class="cm-subtitle">Chatting with · {conversation["document_name"]}</div>', unsafe_allow_html=True)
    st.write("")

    collection_name = collection_for_digest(conversation["document_digest"])

    messages = get_messages(conversation["id"])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["sources"]:
                with st.expander("Sources", expanded=False):
                    for source in message["sources"]:
                        st.caption(f"📄 {source}")

    if not messages:
        st.markdown(
            '<div class="cm-empty"><h3>Ask anything about these notes</h3>'
            '<p>Get a clear explanation, a quick summary, or help finding a specific concept.</p></div>',
            unsafe_allow_html=True,
        )
        st.write("")

    query = st.chat_input("Ask a question about your PDF…")
    if query:
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("GOOGLE_API_KEY is missing. Add it to your local .env file and restart the app.")
        else:
            try:
                with st.spinner("Searching your notes and writing an answer…"):
                    vectorstore = load_vectorstore(collection_name)
                    retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
                    )
                    docs = retriever.invoke(query)
                    context, sources = format_retrieved_context(docs, conversation["document_name"])
                    if docs:
                        prompt = ChatPromptTemplate.from_messages(
                            [
                                ("system", "You help students understand their course notes. Answer using only the provided context. If it does not contain the answer, clearly say you could not find it in the document."),
                                ("human", "Recent conversation:\n{history}\n\nContext:\n{context}\n\nQuestion:\n{question}"),
                            ]
                        )
                        history = get_messages(conversation["id"])[-(MAX_CHAT_CONTEXT_PAIRS * 2):]
                        history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history)
                        llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
                        response = llm.invoke(prompt.invoke({"history": history_text, "context": context, "question": query}))
                        answer = extract_response_text(response.content).strip()
                    else:
                        answer, sources = "I could not find relevant text for this question in the document.", []
                add_message(conversation["id"], "user", query)
                add_message(conversation["id"], "assistant", answer, sources)
                st.rerun()
            except Exception as exc:
                logging.exception("Question answering failed")
                st.error(f"Could not answer right now ({type(exc).__name__}). Check the terminal, then try again.")

    with st.sidebar:
        if st.button("Clear this chat", use_container_width=True):
            clear_conversation(conversation["id"])
            st.rerun()

else:
    st.markdown('<div class="cm-eyebrow">Your personal study assistant</div>', unsafe_allow_html=True)
    st.title("Study smarter with your notes")
    st.markdown('<div class="cm-subtitle">Upload a course PDF, then ask questions and get answers grounded in its pages.</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(
        '<div class="cm-empty"><h3>👋 Welcome to CourseMate</h3>'
        '<p>Start a new chat by adding a PDF in the left panel. Your saved conversations will stay here on this computer.</p></div>',
        unsafe_allow_html=True,
    )



