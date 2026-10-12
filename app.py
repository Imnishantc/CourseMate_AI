import hashlib
import logging
import os
import uuid
from tempfile import NamedTemporaryFile

import streamlit as st
from dotenv import load_dotenv
from create_database import create_vectorstore, document_collection_name
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

MAX_UPLOAD_SIZE_MB = 20
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

def extract_response_text(content):
    """Return readable text from Gemini's structured content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_blocks = []
        for block in content:
            if isinstance(block, str):
                text_blocks.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                text_blocks.append(block["text"])
        return "\n".join(text_blocks)
    return str(content)


def format_retrieved_context(docs, fallback_name):
    """Add page provenance to retrieved text and collect unique citations."""
    context_parts = []
    sources = []
    for doc in docs:
        page = doc.metadata.get("page")
        page_label = str(page + 1) if isinstance(page, int) else "unknown"
        source = f"{doc.metadata.get('source', fallback_name)}, page {page_label}"
        if source not in sources:
            sources.append(source)
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(context_parts), sources


load_dotenv()

st.set_page_config(page_title="CourseMate AI", layout="wide")
st.title("📄 CourseMate AI")

if "storage_namespace" not in st.session_state:
    st.session_state.storage_namespace = uuid.uuid4().hex

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
uploaded_digest = None

if uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    if len(uploaded_bytes) > MAX_UPLOAD_SIZE_BYTES:
        st.error(f"PDFs must be {MAX_UPLOAD_SIZE_MB} MB or smaller.")
    else:
        uploaded_digest = hashlib.sha256(uploaded_bytes).hexdigest()
        collection_name = document_collection_name(
            st.session_state.storage_namespace, uploaded_digest
        )
        st.success("PDF uploaded successfully!")

        if st.button("Process PDF"):
            temporary_path = None
            try:
                with st.spinner("Processing..."):
                    with NamedTemporaryFile(suffix=".pdf", delete=False) as temporary_file:
                        temporary_file.write(uploaded_bytes)
                        temporary_path = temporary_file.name

                    create_vectorstore(temporary_path, collection_name, uploaded_file.name)
                    if st.session_state.get("document_digest") != uploaded_digest:
                        st.session_state.chat_history = []
                    st.session_state.collection_name = collection_name
                    st.session_state.document_digest = uploaded_digest
                    st.session_state.db_ready = True

                st.success("Vector database created for this document!")
            except Exception:
                logging.exception("PDF processing failed")
                st.error(
                    "Could not process this PDF. Check that it is readable and "
                    "contains extractable text, then try again."
                )
            finally:
                if temporary_path is not None:
                    os.unlink(temporary_path)

@st.cache_resource
def load_vectorstore(collection_name):
    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    return Chroma(
        collection_name=collection_name,
        persist_directory="chroma_db",
        embedding_function=embedding_model,
    )

has_active_document = (
    st.session_state.get("db_ready", False)
    and uploaded_digest is not None
    and st.session_state.get("document_digest") == uploaded_digest
)

if has_active_document:
    vectorstore = load_vectorstore(st.session_state.collection_name)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
""",
            ),
        ]
    )

    st.subheader("💬 Ask Questions")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.chat_input("Ask something...")
    if query:
        try:
            docs = retriever.invoke(query)
            context, sources = format_retrieved_context(docs, uploaded_file.name)
            if docs:
                final_prompt = prompt.invoke({"context": context, "question": query})
                response = llm.invoke(final_prompt)
                answer = extract_response_text(response.content)
            else:
                answer = "I could not find the answer in the document."
            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("AI", answer))
            if sources:
                st.session_state.chat_history.append(("Sources", sources))
        except Exception:
            logging.exception("Question answering failed")
            st.error("Could not answer right now. Please try again.")

    for role, message in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑 You:** {message}")
        elif role == "AI":
            st.markdown(f"**🤖 AI:** {message}")
        else:
            st.caption("Sources: " + "; ".join(message))
elif uploaded_file is not None and uploaded_digest is not None:
    st.info("Process this PDF before asking questions.")
