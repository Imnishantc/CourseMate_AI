import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python 'document loaders/pdf.py' <path-to-pdf>")

    pdf_path = Path(sys.argv[1])
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    documents = [
        Document(page_content=page.extract_text() or "", metadata={"page": index})
        for index, page in enumerate(reader.pages)
    ]
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=10,
    ).split_documents(documents)
    if not chunks:
        raise SystemExit("The PDF did not contain extractable text.")
    print(chunks[0].page_content)


if __name__ == "__main__":
    main()
