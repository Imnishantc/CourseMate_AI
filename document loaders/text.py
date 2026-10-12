from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

text = Path("document loaders/ml_notes.txt").read_text(encoding="utf-8")
document = Document(page_content=text, metadata={"source": "ml_notes.txt"})
chunks = CharacterTextSplitter(
    separator="",
    chunk_size=1000,
    chunk_overlap=1,
).split_documents([document])

for chunk in chunks:
    print(chunk.page_content)
    print()

print(len(chunks))
