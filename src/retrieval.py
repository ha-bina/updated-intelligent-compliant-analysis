import numpy as np
from typing import List, Dict, Any
from config.settings import TOP_K, SIMILARITY_METRIC
from src.embedding import EmbeddingModel

class Retriever:
    def __init__(self, vector_store, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
    
    def _calculate_similarity(self, query_embedding: np.ndarray, doc_embeddings: np.ndarray) -> np.ndarray:
        """Calculate similarity between query and documents."""
        if SIMILARITY_METRIC == "cosine":
            # Cosine similarity
            dot_product = np.dot(doc_embeddings, query_embedding)
            doc_norms = np.linalg.norm(doc_embeddings, axis=1)
            query_norm = np.linalg.norm(query_embedding)
            return dot_product / (doc_norms * query_norm)
        else:
            # Default to L2 distance
            return -np.linalg.norm(doc_embeddings - query_embedding, axis=1)
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve most relevant documents for a query."""
        # Embed the query
        query_embedding = self.embedding_model.embed_text(query)
        
        if self.vector_store.store_type == "faiss":
            import faiss
            # FAISS search
            distances, indices = self.vector_store.index.search(
                np.array([query_embedding]).astype('float32'), 
                TOP_K
            )
            
            # Convert to similarity scores (assuming L2 distance was used)
            similarities = 1 / (1 + distances[0])
            retrieved_docs = []
            
            for idx, sim in zip(indices[0], similarities):
                if idx >= 0:  # FAISS may return -1 for invalid indices
                    doc_meta = self.vector_store.metadata[idx]
                    retrieved_docs.append({
                        "text": doc_meta['text_chunk'],
                        "similarity": float(sim),
                        "source": {
                            "complaint_id": doc_meta['complaint_id'],
                            "product": doc_meta['product'],
                            "original_narrative": doc_meta['original_narrative']
                        }
                    })
            
            # Sort by similarity score (descending)
            retrieved_docs.sort(key=lambda x: x['similarity'], reverse=True)
            return retrieved_docs
        else:
            raise ValueError(f"Unsupported vector store type: {self.vector_store.store_type}")