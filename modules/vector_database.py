# modules/vector_database.py
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import faiss
import os
import pickle
import sentence_transformers
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Class to handle vector database operations using FAISS"""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 index_path: str = "./data/vector_index",
                 metadata_path: str = "./data/vector_metadata"):
        self.embedding_model = embedding_model
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Initialize the embedding model
        try:
            self.model = sentence_transformers.SentenceTransformer(embedding_model)
            logger.info(f"Initialized embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Load or create index
        self.index, self.metadata = self._load_or_create_index()
        
    def _load_or_create_index(self) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """Load existing index or create a new one"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                # Load existing index and metadata
                index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                logger.info(f"Loaded existing vector index with {index.ntotal} entries")
                return index, metadata
            else:
                # Create a new index and empty metadata list
                # Get embedding dimension from model
                embedding_dim = self.model.get_sentence_embedding_dimension()
                index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index
                metadata = []
                logger.info(f"Created new vector index with dimension {embedding_dim}")
                return index, metadata
        except Exception as e:
            logger.error(f"Error loading/creating index: {e}")
            # Create a new index as fallback
            embedding_dim = self.model.get_sentence_embedding_dimension()
            index = faiss.IndexFlatL2(embedding_dim)
            metadata = []
            return index, metadata
    
    def _save_index(self) -> None:
        """Save the index and metadata to disk"""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Saved vector index with {self.index.ntotal} entries")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text"""
        embedding = self.model.encode(text)
        return embedding.reshape(1, -1).astype('float32')
    
    async def add_document(self, 
                     doc_id: str, 
                     text: str, 
                     doc_type: str,
                     metadata: Dict[str, Any]) -> int:
        """
        Add a document to the vector database
        
        Args:
            doc_id: unique identifier (e.g., paper URL or database ID)
            text: text to index (abstract or summary)
            doc_type: type of document ("abstract" or "summary")
            metadata: additional metadata about the document
            
        Returns:
            index of the added document
        """
        try:
            # Get embedding
            embedding = self._get_embedding(text)
            
            # Add to index
            idx = self.index.ntotal  # Get current count as new index
            self.index.add(embedding)
            
            # Add metadata
            self.metadata.append({
                "id": doc_id,
                "type": doc_type,
                "text": text,
                "metadata": metadata,
                "added_at": datetime.now().isoformat()
            })
            
            # Save index periodically (can be optimized to save less frequently)
            self._save_index()
            
            return idx
            
        except Exception as e:
            logger.error(f"Error adding document to vector database: {e}")
            raise
    
    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: search query
            limit: maximum number of results to return
            
        Returns:
            List of documents with similarity scores
        """
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            
            # Search the index
            distances, indices = self.index.search(query_embedding, limit)
            
            # Prepare results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.metadata):  # Check if valid index
                    doc = self.metadata[idx]
                    results.append({
                        "score": float(1 - distances[0][i] / 2),  # Convert L2 distance to similarity score
                        "id": doc["id"],
                        "type": doc["type"],
                        "text": doc["text"],
                        "metadata": doc["metadata"]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            return []
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the index
        Note: FAISS doesn't support direct deletion, so we rebuild the index
        
        Args:
            doc_id: ID of document to delete
            
        Returns:
            Success status
        """
        try:
            # Filter out the document to delete
            new_metadata = [doc for doc in self.metadata if doc["id"] != doc_id]
            
            if len(new_metadata) == len(self.metadata):
                logger.warning(f"Document {doc_id} not found for deletion")
                return False
            
            # Create a new index
            embedding_dim = self.model.get_sentence_embedding_dimension()
            new_index = faiss.IndexFlatL2(embedding_dim)
            
            # Add remaining documents
            for doc in new_metadata:
                embedding = self._get_embedding(doc["text"])
                new_index.add(embedding)
            
            # Replace old index and metadata
            self.index = new_index
            self.metadata = new_metadata
            
            # Save updated index
            self._save_index()
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False