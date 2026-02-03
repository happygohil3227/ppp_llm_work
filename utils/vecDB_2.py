import os
import pickle
from typing import List, Dict

import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# =====================================================
# VECTOR DB (RUNTIME-SAFE, NO EMBEDDINGS)
# =====================================================

class LegalVectorDB:
    """
    Vector database for legal / PPP documents.

    ARCHITECTURE (CRITICAL):
    - Embeddings are created ONLY during build phase
    - At runtime, this class performs FAISS search ONLY
    - No torch, no transformers, no embedding model here
    - Fully parallel-safe for LangGraph
    """

    def __init__(
        self,
        chunk_size: int = 3600,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Runtime objects
        self.index = None
        self.documents: List[Document] = []

    # =================================================
    # CHUNKING (LEGAL-SAFE, STRUCTURE-PRESERVING)
    # =================================================

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text in a legally safe way.
        Preserves paragraph and contextual integrity.
        """

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n",   # paragraph
                "\n",     # line
                ". ",     # sentence fallback
            ],
            keep_separator=True
        )

        chunks = splitter.split_text(text)

        # Hard safety filter: reject tiny chunks
        safe_chunks = [
            chunk.strip()
            for chunk in chunks
            if len(chunk.strip()) >= 300
        ]

        return safe_chunks

    # =================================================
    # DOCUMENT INGESTION (BUILD PHASE ONLY)
    # =================================================

    def ingest_document(
        self,
        text: str,
        source_name: str
    ) -> None:
        """
        Ingest a full legal document into vector DB.
        (BUILD PHASE ONLY)
        """

        chunks = self._chunk_text(text)

        for idx, chunk in enumerate(chunks):
            self.documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source_name,
                        "chunk_id": idx
                    }
                )
            )

    # =================================================
    # INDEX ATTACHMENT (EMBEDDINGS PRE-COMPUTED)
    # =================================================

    def attach_index(self, index: faiss.Index) -> None:
        """
        Attach a pre-built FAISS index.
        Embeddings must already exist.
        """
        self.index = index

    # =================================================
    # SECTION-WISE RETRIEVAL (RUNTIME SAFE)
    # =================================================

    def retrieve_for_section(
        self,
        query_vector: List[float],
        top_k: int = 10
    ) -> List[Document]:
        """
        Retrieve context-rich chunks for a section.

        IMPORTANT:
        - query_vector MUST be pre-embedded
        - NO embedding model is allowed here
        """

        if self.index is None:
            raise RuntimeError("FAISS index not loaded.")

        query_vector = (
            np.asarray(query_vector)
            .astype("float32")
            .reshape(1, -1)
        )

        distances, indices = self.index.search(query_vector, top_k)

        return [
            self.documents[i]
            for i in indices[0]
            if i != -1
        ]

    # =================================================
    # SAVE / LOAD (SERIALIZATION)
    # =================================================

    def save(self, db_path: str) -> None:
        """
        Save FAISS index + documents.
        """
        os.makedirs(db_path, exist_ok=True)

        if self.index is None:
            raise RuntimeError("Cannot save without FAISS index.")

        faiss.write_index(self.index, os.path.join(db_path, "index.faiss"))

        with open(os.path.join(db_path, "docs.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    @classmethod
    def load(cls, db_path: str) -> "LegalVectorDB":
        """
        Load vector DB for runtime usage.
        SAFE for parallel LangGraph execution.
        """

        instance = cls()

        index_path = os.path.join(db_path, "index.faiss")
        docs_path = os.path.join(db_path, "docs.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Missing FAISS index: {index_path}")

        if not os.path.exists(docs_path):
            raise FileNotFoundError(f"Missing docs file: {docs_path}")

        instance.index = faiss.read_index(index_path)

        with open(docs_path, "rb") as f:
            instance.documents = pickle.load(f)

        return instance
