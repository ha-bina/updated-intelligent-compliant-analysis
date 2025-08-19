import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import (
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_DIR
)
import numpy as np
import os
from typing import List, Dict, Any
import pandas as pd

class TextChunker:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        return self.text_splitter.split_text(text)

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self.model.encode(text)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        return self.model.encode(texts)

class VectorStore:
    def __init__(self, store_type: str = "faiss"):
        self.store_type = store_type
        self.embeddings = None
        self.metadata = None
    
    def create_store(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Create vector store from embeddings and metadata."""
        self.embeddings = embeddings
        self.metadata = metadata
        
        if self.store_type == "faiss":
            import faiss
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            self.index = index
    
    def save_store(self, file_prefix: str):
        """Save vector store to disk."""
        if not os.path.exists(VECTOR_STORE_DIR):
            os.makedirs(VECTOR_STORE_DIR)
        
        if self.store_type == "faiss":
            import faiss
            faiss.write_index(self.index, str(VECTOR_STORE_DIR / f"{file_prefix}.index"))
        
        # Save metadata
        pd.DataFrame(self.metadata).to_csv(
            VECTOR_STORE_DIR / f"{file_prefix}_metadata.csv", 
            index=False
        )
    
    @classmethod
    def load_store(cls, file_prefix: str, store_type: str = "faiss"):
        """Load vector store from disk."""
        vec_store = cls(store_type)
        
        if store_type == "faiss":
            import faiss
            vec_store.index = faiss.read_index(str(VECTOR_STORE_DIR / f"{file_prefix}.index"))
        
        # Load metadata
        vec_store.metadata = pd.read_csv(
            VECTOR_STORE_DIR / f"{file_prefix}_metadata.csv"
        ).to_dict('records')
        
        return vec_store