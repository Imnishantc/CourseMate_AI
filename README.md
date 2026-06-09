# CourseMate_AI

An AI-powered academic assistant built using Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), and Vector Databases to help students interact with course materials through natural language queries.

The system processes academic PDFs, generates vector embeddings, retrieves relevant content using semantic search, and produces context-aware responses using the Mistral LLM.


##  Features

- 📄 PDF document ingestion and processing
- 🔍 Semantic search using vector embeddings
- 🤖 AI-powered question answering
- 🧠 Retrieval-Augmented Generation (RAG) architecture
- ⚡ Fast document retrieval with ChromaDB
- 🔐 Secure API key management using environment variables
- 📚 Context-aware responses grounded in course materials



##  Architecture

```text
PDF Documents
      │
      ▼
Document Loader
      │
      ▼
Text Chunking
      │
      ▼
Embedding Generation
      │
      ▼
Chroma Vector Database
      │
      ▼
Semantic Retrieval
      │
      ▼
Mistral LLM
      │
      ▼
Generated Response
```


##  Tech Stack

| Programming Language | Python |
| LLM | Mistral AI |
| Framework | LangChain |
| Vector Database | ChromaDB |
| AI Concepts | RAG, Embeddings, Semantic Search |
| Environment Management | Python Dotenv |



##  Project Structure

```bash
CourseMate_AI/
│
├── chroma_db/              # Persistent vector database
├── document_loaders/       # PDF loading and preprocessing
├── vector_stores/          # Vector store operations
│
├── app.py                  # Application interface
├── main.py                 # Core RAG pipeline
├── create_database.py      # Embedding and database creation
├── temp.pdf                # Sample academic document
├── requirement.txt         # Project dependencies
├── .env                    # Environment variables
└── README.md
```

##  Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/CourseMate_AI.git
cd CourseMate_AI
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirement.txt
```

### Configure Environment Variables

Create a `.env` file in the root directory:

```env
MISTRAL_API_KEY=your_api_key_here
```

### Generate Vector Database

```bash
python create_database.py
```

### Run the Application

```bash
python app.py
```



##  Workflow

1. Load academic PDF documents.
2. Extract and split text into manageable chunks.
3. Generate vector embeddings for each chunk.
4. Store embeddings in ChromaDB.
5. Accept user queries in natural language.
6. Retrieve the most relevant document context.
7. Augment prompts with retrieved information.
8. Generate accurate responses using Mistral AI.



##  Skills Demonstrated

- Machine Learning Fundamentals
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Vector Databases
- Embedding-Based Search
- Semantic Retrieval
- Prompt Engineering
- API Integration
- Information Retrieval Systems
- End-to-End AI Application Development



##  Future Enhancements

- Multi-document support
- Conversational memory
- AI agents for academic assistance
- Response evaluation framework
- Voice-based interaction
- Multi-language support
- Cloud deployment


## ⭐ Acknowledgements

This project was developed to explore modern Generative AI techniques and demonstrate practical implementation of Retrieval-Augmented Generation (RAG), Vector Search, and LLM-powered question-answering systems for educational applications.

