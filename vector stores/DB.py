from langchain_community.vectorstores import Chroma
from langchain_mistralai  import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
docs = [ Document(
        page_content="Machine learning is a branch of artificial intelligence that allows systems to learn from data.",
        metadata={"source": "ml_book", "page": 1}
        ),
        Document(
        page_content="Deep learning is a subset of machine learning that uses neural networks with many layers.",
        metadata={"source": "ml_book", "page": 2}
        ),
        Document(
        page_content="Supervised learning uses labeled data to train models.",
        metadata={"source": "ml_book", "page": 3}
        ),
        Document(
        page_content="Unsupervised learning finds patterns without labels.",
        metadata={"source": "ml_book", "page": 4})
]

embedding_model = MistralAIEmbeddings(model = "mistral-embed")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

result = vectorstore.similarity_search("What does supervised learning uses to train models?",k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriver = vectorstore.as_retriever()

docs = retriver.invoke("explain machine learning")

for d in docs:
    print(d.page_content)
