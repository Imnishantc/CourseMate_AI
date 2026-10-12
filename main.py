import hashlib
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from create_database import create_vectorstore, document_collection_name


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python main.py <path-to-pdf>")

    pdf_path = Path(sys.argv[1])
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    load_dotenv()
    content_digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    collection_name = document_collection_name(uuid.uuid4().hex, content_digest)
    create_vectorstore(str(pdf_path), collection_name, pdf_path.name)

    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory="chroma_db",
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    llm = ChatMistralAI(model="mistral-large-2512")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Answer using only the provided PDF context. If it does not contain
the answer, say: "I could not find the answer in the document."
""",
            ),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )

    print("CourseMate AI is ready. Enter 0 to exit.")
    while True:
        query = input("You: ").strip()
        if query == "0":
            break
        if not query:
            continue

        docs = retriever.invoke(query)
        context_parts = []
        sources = []
        for doc in docs:
            page = doc.metadata.get("page")
            page_label = str(page + 1) if isinstance(page, int) else "unknown"
            source = f"{doc.metadata.get('source', pdf_path.name)}, page {page_label}"
            if source not in sources:
                sources.append(source)
            context_parts.append(f"[Source: {source}]\n{doc.page_content}")

        response = llm.invoke(
            prompt.invoke(
                {"context": "\n\n".join(context_parts), "question": query}
            )
        )
        print(f"\nAI: {response.content}")
        if sources:
            print("Sources: " + "; ".join(sources))
        print()


if __name__ == "__main__":
    main()
