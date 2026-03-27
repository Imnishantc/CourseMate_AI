import streamlit as st
import os
from dotenv import load_dotenv
from create_database import create_vectorstore
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="CourseMate AI", layout="wide")

st.title("📄 CourseMate AI")

# -------------------------------
# Upload PDF
# -------------------------------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:

    file_path = "temp.pdf"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")

    if st.button("Process PDF"):
        with st.spinner("Processing..."):

            # Create DB only once
            create_vectorstore(file_path)

            st.session_state.db_ready = True

        st.success("Vector DB created!")

# -------------------------------
# Cache Vectorstore (IMPORTANT ⚡)
# -------------------------------
@st.cache_resource
def load_vectorstore():
    embedding_model = MistralAIEmbeddings(model="mistral-embed")

    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

# -------------------------------
# Q&A Section
# -------------------------------
if st.session_state.get("db_ready", False):

    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    llm = ChatMistralAI(model="mistral-large-2512")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
"""
            )
        ]
    )

    st.subheader("💬 Ask Questions")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.chat_input("Ask something...")

    if query:

        docs = retriever.invoke(query)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        final_prompt = prompt.invoke({
            "context": context,
            "question": query
        })

        response = llm.invoke(final_prompt)

        # Store chat
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("AI", response.content))

    # Display chat
    for role, message in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑 You:** {message}")
        else:
            st.markdown(f"**🤖 AI:** {message}")