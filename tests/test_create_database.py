import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

import create_database


class CreateDatabaseTests(unittest.TestCase):
    def test_collection_name_is_scoped_to_session_and_document(self):
        digest = "a" * 64
        namespace = "1" * 32
        first = create_database.document_collection_name(namespace, digest)
        another_session = create_database.document_collection_name("2" * 32, digest)
        another_document = create_database.document_collection_name(
            namespace, "b" * 64
        )

        self.assertEqual(len(first), 60)
        self.assertNotEqual(first, another_session)
        self.assertNotEqual(first, another_document)

    @patch("create_database.Chroma.from_documents")
    @patch("create_database.GoogleGenerativeAIEmbeddings")
    @patch("create_database.RecursiveCharacterTextSplitter")
    @patch("create_database.PdfReader")
    def test_ingestion_sets_source_and_collection(
        self, reader_factory, splitter_factory, embeddings_factory, from_documents
    ):
        page = MagicMock()
        page.extract_text.return_value = "Course material text"
        reader = reader_factory.return_value
        reader.is_encrypted = False
        reader.pages = [page]

        splitter_factory.return_value.split_documents.side_effect = (
            lambda documents: documents
        )
        embedding = object()
        embeddings_factory.return_value = embedding
        store = object()
        from_documents.return_value = store

        result = create_database.create_vectorstore(
            "temporary.pdf", "cm_session_document", "lecture.pdf"
        )

        self.assertIs(result, store)
        chunks = from_documents.call_args.kwargs["documents"]
        self.assertEqual(chunks[0].metadata["source"], "lecture.pdf")
        kwargs = from_documents.call_args.kwargs
        self.assertEqual(kwargs["collection_name"], "cm_session_document")
        self.assertEqual(kwargs["embedding"], embedding)
        self.assertEqual(len(kwargs["ids"]), 1)

    @patch("create_database.GoogleGenerativeAIEmbeddings")
    @patch("create_database.PdfReader")
    def test_rejects_pdf_without_extractable_text(self, reader_factory, embeddings):
        page = MagicMock()
        page.extract_text.return_value = "  "
        reader = reader_factory.return_value
        reader.is_encrypted = False
        reader.pages = [page]

        with self.assertRaisesRegex(ValueError, "no extractable text"):
            create_database.create_vectorstore("empty.pdf", "cm_test_document")

        embeddings.assert_not_called()

    @patch("create_database.GoogleGenerativeAIEmbeddings")
    @patch("create_database.PdfReader")
    def test_rejects_pdf_over_page_limit_before_embedding(
        self, reader_factory, embeddings
    ):
        reader = reader_factory.return_value
        reader.is_encrypted = False
        reader.pages = [MagicMock() for _ in range(create_database.MAX_PDF_PAGES + 1)]

        with self.assertRaisesRegex(ValueError, "500 pages"):
            create_database.create_vectorstore("large.pdf", "cm_test_document")

        embeddings.assert_not_called()


if __name__ == "__main__":
    unittest.main()
