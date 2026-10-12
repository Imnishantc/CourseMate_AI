# CourseMate AI

CourseMate AI is a Streamlit prototype for asking questions about an uploaded course PDF. It extracts PDF text, splits it into chunks, embeds the chunks with Gemini, stores them in Chroma, and uses retrieval-augmented generation (RAG) to answer questions from the selected document.

The Gemini Developer API free tier has usage limits. Google states that free-tier inputs may be used to improve its products, so avoid uploading sensitive or confidential course documents.

## Features

- Upload one PDF at a time (up to 20 MB and 500 pages).
- Retrieve relevant passages with maximal marginal relevance (MMR).
- Generate answers with Gemini using the retrieved document context.
- Keep each Streamlit session's document in a separate Chroma collection.
- Show source filenames and page numbers with answers.
- Report PDF processing and model errors without crashing the app.

## Technology

- Python and Streamlit
- LangChain Core and standalone integration packages
- Gemini embeddings and chat model
- ChromaDB
- PyPDF

## Project structure

```text
CourseMate_AI/
├── app.py                  # Streamlit application
├── create_database.py      # PDF ingestion and vector-store creation
├── main.py                 # Command-line RAG client
├── requirement.txt         # Direct dependency pins for pip
├── pyproject.toml          # Project dependencies for uv
├── uv.lock                 # Resolved transitive dependency lock
├── .env.example            # Environment variable template
├── document loaders/       # Loader experiments and small text notes
├── vector stores/          # Vector-store experiment
├── tests/                  # Unit tests
└── .github/workflows/      # CI checks
```

The Chroma database is created locally in `chroma_db/` when you process a PDF. It is runtime data and is excluded from Git. Temporary uploaded PDFs are removed after processing.

## Run locally

1. Install Python 3.10 or newer and [uv](https://docs.astral.sh/uv/).

2. Install the exact locked environment.

   ```bash
   uv sync --locked
   ```

3. Copy `.env.example` to `.env` and set your Google AI Studio API key (`GOOGLE_API_KEY`). Keep `.env` local; it must not be committed.

4. Start the app.

   ```bash
   uv run streamlit run app.py
   ```

5. Upload a PDF, choose **Process PDF**, then ask questions about that processed document.

The command-line client can ingest and query one PDF:

```bash
uv run python main.py path/to/course.pdf
```

## Current limitations

- The app processes one selected PDF per Streamlit session.
- Chroma data is stored on the local filesystem; multi-instance deployment needs shared, access-controlled storage.
- Streamlit session isolation is not a substitute for user authentication or tenant authorization.
- The app has no account system, quotas, or document-retention policy.

## Project status

This is a learning project and prototype. Unit tests, CI, direct dependency pins, and a transitive lockfile are included. Production deployment still needs authentication, shared access-controlled storage, retention controls, and deployment monitoring.
