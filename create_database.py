import hashlib
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

MAX_PDF_PAGES = 500


def document_collection_name(session_namespace, content_digest):
    """Return a Chroma-safe collection name scoped to one session and PDF."""
    return f"cm_{session_namespace}_{content_digest[:24]}"


def create_vectorstore(file_path, collection_name, source_name=None):
    load_dotenv()

    reader = PdfReader(file_path, strict=False)
    if reader.is_encrypted:
        try:
            if not reader.decrypt(""):
                raise ValueError("Password-protected PDFs are not supported.")
        except Exception as exc:
            raise ValueError("Password-protected PDFs are not supported.") from exc
    if not reader.pages:
        raise ValueError("No pages could be read from this PDF.")
    if len(reader.pages) > MAX_PDF_PAGES:
        raise ValueError(f"PDFs must have {MAX_PDF_PAGES} pages or fewer.")

    source = os.path.basename(source_name or file_path)
    docs = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"source": source, "page": page_number},
        )
        for page_number, page in enumerate(reader.pages)
    ]
    if not any(doc.page_content.strip() for doc in docs):
        raise ValueError("This PDF contains no extractable text.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("This PDF contains no extractable text.")

    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
    chunk_ids = [
        hashlib.sha256(
            f"{collection_name}:{index}:{chunk.page_content}".encode("utf-8")
        ).hexdigest()
        for index, chunk in enumerate(chunks)
    ]

    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        ids=chunk_ids,
        collection_name=collection_name,
        persist_directory="chroma_db",
    )
