from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from typing import List
from db import VectorStore
import os
import uuid
import re
from datetime import datetime


class PdfProcessor:
    """
    A class that hanndles loading, processing, embedding and storing PDF documents in a vector DB.
    """

    def __init__(
        self,
        pdf_directory: str,
        vector_store: VectorStore,
        collection_name: str,
        embedding_model: str,
        openai_api_key: str,
        chunk_size: int = 500,
        chunk_overlap: int = 250,
    ):
        self.pdf_directory = pdf_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = vector_store

        # Open AI settings
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key required")

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=self.openai_api_key,
        )

    def _load_documents(self):
        loader = PyPDFDirectoryLoader(self.pdf_directory)
        documents = loader.load()
        return documents

    def _split_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        splits = text_splitter.split_documents(documents)
        return splits

    def _embed_documents(self, documents):
        """Generate embeddings for document chunks."""
        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        return embeddings

    def _store_documents(self, documents: List[Document]):
        embeddings = self._embed_documents(documents)
        metadata = [doc.metadata for doc in documents]
        # Include page_content in the metadata
        augmented_metadata = [
            {**meta, "text": doc.page_content} for doc, meta in zip(documents, metadata)
        ]
        ids = [str(uuid.uuid4()) for _ in documents]
        self.vector_store.store_embeddings(
            embeddings=embeddings, metadata=augmented_metadata, ids=ids
        )

    def _extract_metadata(self, documents: List[Document]):
        """Extract metadata from documents. Add information about fiscal quarter, year, and document date."""
        # Group documents by source file
        doc_groups = {}
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if source not in doc_groups:
                doc_groups[source] = []
            doc_groups[source].append(doc)

        enhanced_documents = []
        for source, docs in doc_groups.items():
            # Sort by page number to ensure we look at front pages first
            docs.sort(key=lambda d: d.metadata.get("page", 0))

            # Initialize metadata fields
            fiscal_quarter = None
            fiscal_year = None
            document_date = None

            # Get first page text for metadata extraction
            first_page_text = docs[0].page_content if docs else ""

            # Extract fiscal quarter
            # Look for Q1-Q4 pattern
            q_pattern = re.compile(
                r"Q([1-4])|FQ([1-4])|Quarter\s+([1-4])|([1-4])(?:st|nd|rd|th)\s+Quarter",
                re.IGNORECASE,
            )
            q_match = q_pattern.search(first_page_text)
            if q_match:
                q_value = next((g for g in q_match.groups() if g is not None), None)
                if q_value:
                    fiscal_quarter = f"Q{q_value}"
            else:
                # Try word-based quarters
                for idx, term in enumerate(["First", "Second", "Third", "Fourth"]):
                    if term.lower() in first_page_text.lower():
                        fiscal_quarter = f"Q{idx + 1}"
                        break

            # Extract fiscal year
            # Look for "FY20xx" or "Fiscal Year 20xx" or "Fiscal 20xx" or "FQx 20xx"
            fy_pattern = re.compile(
                r"FY(20\d{2})|FY(\d{2})|Fiscal\s+(?:Year\s+)?(20\d{2})|FQ\d\s+(20\d{2})",
                re.IGNORECASE,
            )
            fy_match = fy_pattern.search(first_page_text)
            if fy_match:
                groups = fy_match.groups()
                if groups[0]:  # FY2023
                    fiscal_year = groups[0]
                elif groups[1]:  # FY23
                    fiscal_year = f"20{groups[1]}"
                elif groups[2]:  # Fiscal Year 2023 or Fiscal 2023
                    fiscal_year = groups[2]
                elif groups[3]:  # FQ3 2023
                    fiscal_year = groups[3]
            # Extract document date
            # Month Day, Year
            date_patterns = [
                # January 1, 2023
                r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+20\d{2}",
                # 01/30/2023 or 01-30-2023
                r"(\d{1,2})[/-](\d{1,2})[/-](20\d{2})",
                # 2023-01-30
                r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})",
            ]

            for pattern in date_patterns:
                date_match = re.search(pattern, first_page_text)
                if date_match:
                    try:
                        date_str = date_match.group(0)
                        if re.match(r"\w+\s+\d{1,2},?\s+20\d{2}", date_str):
                            # Month name format
                            date_str = date_str.replace(",", "")
                            document_date = int(
                                datetime.strptime(date_str, "%B %d %Y").timestamp()
                            )
                        elif re.match(r"\d{1,2}[/-]\d{1,2}[/-]20\d{2}", date_str):
                            # MM/DD/YYYY format
                            date_str = date_str.replace("-", "/")
                            document_date = int(
                                datetime.strptime(date_str, "%m/%d/%Y").timestamp()
                            )
                        elif re.match(r"20\d{2}[/-]\d{1,2}[/-]\d{1,2}", date_str):
                            # YYYY-MM-DD format
                            date_str = date_str.replace("/", "-")
                            document_date = int(
                                datetime.strptime(date_str, "%Y-%m-%d").timestamp()
                            )
                        break
                    except ValueError:
                        pass

            # Enhance metadata for all docs in this group
            for doc in docs:
                doc.metadata["fiscal_quarter"] = fiscal_quarter
                doc.metadata["fiscal_year"] = fiscal_year
                doc.metadata["document_date"] = document_date
                try:
                    if doc.metadata.get("creationdate"):
                        doc.metadata["creationdate"] = int(
                            datetime.fromisoformat(
                                doc.metadata.get("creationdate")
                            ).timestamp()
                        )
                    if doc.metadata.get("moddate"):
                        doc.metadata["moddate"] = int(
                            datetime.fromisoformat(
                                doc.metadata.get("moddate")
                            ).timestamp()
                        )
                except ValueError:
                    # Handle any parsing errors
                    pass

                enhanced_documents.append(doc)

        return enhanced_documents

    def process_pdfs(self):
        """Main method to load, split, embed, and store PDFs."""
        documents = self._load_documents()
        split_docs = self._split_documents(documents)
        enhanced_docs = self._extract_metadata(split_docs)
        self._store_documents(enhanced_docs)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: list = [],
    ) -> List[Document]:
        query_embedding = self.embeddings.embed_query(query)

        db_filter = None
        if filters:
            db_filter = self.vector_store.filter_bulder(filters)

        raw_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter=db_filter,
        )

        documents = []
        for result in raw_results:
            doc = Document(
                page_content=result.payload.get("text", ""),
                metadata={
                    **{k: v for k, v in result.payload.items() if k != "text"},
                    "score": result.score,
                },
            )
            documents.append(doc)

        return documents
