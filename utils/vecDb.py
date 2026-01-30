# import os
# from typing import List, Optional

# import torch
# import numpy as np
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModel
# from sklearn.preprocessing import normalize

# from langchain.embeddings.base import Embeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document


# class InLegalMCAStore(Embeddings):
#     """
#     End-to-end utility:
#     - InLegalBERT embeddings
#     - Text chunking
#     - FAISS vector store creation / loading
#     """

#     def __init__(
#         self,
#         model_name: str = "law-ai/InLegalBERT",
#         max_length: int = 512,
#         chunk_size: int = 1000,
#         overlap: int = 200,
#         device: Optional[str] = None
#     ):
#         self.model_name = model_name
#         self.max_length = max_length
#         self.chunk_size = chunk_size
#         self.overlap = overlap
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name, use_safetensors=True)
#         self.model.to(self.device)
#         self.model.eval()

#     # =========================
#     # Embeddings (LangChain API)
#     # =========================

#     # def _embed_text(self, text: str) -> np.ndarray:
#     #     inputs = self.tokenizer(
#     #         text,
#     #         return_tensors="pt",
#     #         truncation=True,
#     #         max_length=self.max_length,
#     #         padding=True
#     #     )
#     #     inputs = {k: v.to(self.device) for k, v in inputs.items()}

#     #     with torch.no_grad():
#     #         outputs = self.model(**inputs)

#     #     embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
#     #     embedding = normalize(embedding)[0]

#     #     return embedding.astype(np.float32)

#     def _embed_text(self, text: str) -> np.ndarray:
#         inputs = self.tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             max_length=self.max_length,
#             padding=True
#         )
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}

#         with torch.no_grad():
#             outputs = self.model(**inputs)

#         token_embeddings = outputs.last_hidden_state
#         attention_mask = inputs["attention_mask"].unsqueeze(-1)

#         summed = (token_embeddings * attention_mask).sum(dim=1)
#         counts = attention_mask.sum(dim=1)
#         embedding = summed / counts

#         embedding = torch.nn.functional.normalize(embedding, dim=1)
#         return embedding[0].cpu().numpy().astype(np.float32)


#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return [self._embed_text(text) for text in tqdm(texts, desc="Embedding docs")]

#     def embed_query(self, text: str) -> List[float]:
#         return self._embed_text(text)

#     # =========================
#     # Vector Store Logic
#     # =========================

#     def _chunk_text(self, text: str) -> List[str]:
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=self.chunk_size,
#             chunk_overlap=self.overlap,
#             separators=["\n\n", "\n", " ", ""]
#         )
#         return splitter.split_text(text)

#     def build_from_file(
#         self,
#         file_path: str,
#         db_path: Optional[str] = None
#     ) -> FAISS:
#         with open(file_path, "r", encoding="utf-8") as f:
#             text = f.read()

#         chunks = self._chunk_text(text)

#         documents = [
#             Document(
#                 page_content=chunk,
#                 metadata={
#                     "source": os.path.basename(file_path),
#                     "chunk_id": idx
#                 }
#             )
#             for idx, chunk in enumerate(chunks)
#         ]

#         vector_store = FAISS.from_documents(documents, self)

#         if db_path:
#             vector_store.save_local(db_path)

#         return vector_store

#     def load(self, db_path: str) -> FAISS:
#         return FAISS.load_local(
#             db_path,
#             self,
#             allow_dangerous_deserialization=True
#         )


import os
import re
import numpy as np
from typing import List, Dict, Tuple

import torch
from transformers import AutoTokenizer, AutoModel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import faiss

import pickle




# =========================
# EMBEDDING MODEL (LEGAL)
# =========================

class LegalEmbeddingModel:
    """
    InLegalBERT-based embedding model.
    Safe for Indian legal / PPP language.
    """

    def __init__(self, model_name: str = "law-ai/InLegalBERT"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, use_safetensors=True)
        # self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = []

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                outputs = self.model(**inputs)
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]

                # Mean pooling (legal-safe)
                summed = (token_embeddings * attention_mask.unsqueeze(-1)).sum(dim=1)
                counts = attention_mask.sum(dim=1)
                embedding = summed / counts

                embeddings.append(embedding.squeeze().cpu().numpy())

        return np.array(embeddings)


# =========================
# LEGAL-SAFE VECTOR DB
# =========================

class LegalVectorDB:
    """
    Vector database for legal / PPP documents.

    DESIGN PRINCIPLES:
    - NO clause-level regex splitting
    - Large, paragraph-aware chunks
    - Section-wise semantic retrieval
    """

    def __init__(
        self,
        chunk_size: int = 3600,
        chunk_overlap: int = 200,
        embedding_model: str = "law-ai/InLegalBERT"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = LegalEmbeddingModel(embedding_model)

        self.index = None
        self.documents: List[Document] = []

    # =========================
    # CHUNKING (STRUCTURE SAFE)
    # =========================

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
                ". ",     # sentence (fallback only)
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

    # =========================
    # DOCUMENT INGESTION
    # =========================
    def ingest_document(
        self,
        text: str,
        source_name: str
    ) -> None:
        """
        Ingest a full legal document into vector DB.
        """

        chunks = self._chunk_text(text)

        for idx, chunk in enumerate(chunks):
            metadata = {
                "source": source_name,
                "chunk_id": idx
            }

            self.documents.append(
                Document(
                    page_content=chunk,
                    metadata=metadata
                )
            )

    # =========================
    # BUILD FAISS INDEX
    # =========================
    
    def build_index(self) -> None:
        """
        Build FAISS index from ingested documents.
        """

        if not self.documents:
            raise ValueError("No documents ingested.")

        texts = [doc.page_content for doc in self.documents]
        embeddings = self.embedder.embed_texts(texts)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    # =========================
    # SECTION-WISE RETRIEVAL
    # =========================

    def retrieve_for_section(
        self,
        section_query: str,
        top_k: int = 10
    ) -> List[Document]:
        """
        Retrieve large, context-rich chunks for ONE section.

        This is designed for:
        - Tariff analysis
        - Termination analysis
        - Risk allocation
        - Dispute resolution
        """

        if self.index is None:
            raise ValueError("Index not built.")

        query_embedding = self.embedder.embed_texts([section_query])
        distances, indices = self.index.search(query_embedding, top_k)

        retrieved_docs = []
        for idx in indices[0]:
            retrieved_docs.append(self.documents[idx])

        return retrieved_docs
    
    def save(self, db_path: str):
        os.makedirs(db_path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(db_path, "index.faiss"))

        with open(os.path.join(db_path, "docs.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
    @classmethod
    def load(cls, db_path: str):
        instance = cls()
        instance.index = faiss.read_index(os.path.join(db_path, "index.faiss"))

        with open(os.path.join(db_path, "docs.pkl"), "rb") as f:
            instance.documents = pickle.load(f)

        return instance


# =========================
# SECTION QUERY TEMPLATES
# =========================

SECTION_QUERIES: Dict[int, str] = {
    1: "objective purpose intent policy capacity efficiency competition",
    2: "scope concession rights permitted activities approval authority",
    3: "asset ownership control railway property operational authority",
    4: "inspection compliance directions approval micromanagement",
    5: "concession period tenure extension renewal",
    6: "tariff pricing charges freight rate revenue circular notified",
    7: "traffic demand volume risk diversion minimum guarantee",
    8: "change in law policy amendment compensation",
    9: "force majeure relief suspension extension",
    10: "termination step-in substitution compensation default",
    11: "dispute resolution arbitration governing law",
    12: "assignment financing refinancing substitution lender"
}


