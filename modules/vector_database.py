# modules/vector_database.py
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import faiss
import os
import pickle
import sentence_transformers
from datetime import datetime
import asyncio
from threading import Lock

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Class to handle vector database operations using FAISS with enhanced features"""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str = "./data/vector_index",
        metadata_path: str = "./data/vector_metadata",
        save_interval: int = 100,
        index_factory_str: str = "Flat"
    ):
        self.embedding_model = embedding_model
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.save_interval = save_interval
        self.index_factory_str = index_factory_str
        self._lock = Lock()
        self._unsaved_changes = 0
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Initialize the embedding model
        try:
            self.model = sentence_transformers.SentenceTransformer(embedding_model)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized embedding model: {embedding_model} with dimension {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Load or create index
        self.index, self.metadata = self._load_or_create_index()
        
    def _load_or_create_index(self) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """Load existing index or create a new one with configurable type"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                try:
                    index = faiss.read_index(self.index_path)
                    with open(self.metadata_path, 'rb') as f:
                        metadata = pickle.load(f)
                    
                    # Verify index dimension matches model dimension
                    if hasattr(index, 'dimension') and index.d != self.embedding_dim:
                        logger.warning(f"Index dimension ({index.d}) doesn't match model dimension ({self.embedding_dim}). Creating new index.")
                        raise ValueError("Dimension mismatch")
                        
                    logger.info(f"Loaded existing vector index with {index.ntotal} entries")
                    return index, metadata
                except Exception as e:
                    logger.warning(f"Could not load existing index, creating new one: {e}")
                    # Fall through to create new index
            
            # Create new index
            index = faiss.index_factory(self.embedding_dim, self.index_factory_str)
            
            # For non-GPU index types, add checking if it's training required
            if not index.is_trained and hasattr(index, 'train'):
                logger.info(f"Index type {self.index_factory_str} requires training before use")
                # We'll train it when first vectors are added
            
            metadata = []
            logger.info(f"Created new vector index '{self.index_factory_str}' with dimension {self.embedding_dim}")
            return index, metadata
            
        except Exception as e:
            logger.error(f"Error in _load_or_create_index: {e}")
            # Create a simple flat index as fallback
            index = faiss.IndexFlatL2(self.embedding_dim)
            return index, []

    def _save_index(self) -> None:
        """Save the index and metadata to disk safely"""
        with self._lock:
            temp_index_path = f"{self.index_path}.temp"
            temp_metadata_path = f"{self.metadata_path}.temp"
            
            try:
                # Write to temporary files first
                faiss.write_index(self.index, temp_index_path)
                with open(temp_metadata_path, 'wb') as f:
                    pickle.dump(self.metadata, f)
                
                # Rename to final files (atomic operation)
                os.replace(temp_index_path, self.index_path)
                os.replace(temp_metadata_path, self.metadata_path)
                
                logger.info(f"Saved vector index with {self.index.ntotal} entries")
                self._unsaved_changes = 0
            except Exception as e:
                logger.error(f"Error saving index: {e}")
                # Clean up temporary files if they exist
                for path in [temp_index_path, temp_metadata_path]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except:
                            pass
                raise

    def _maybe_save(self) -> None:
        """Save index if unsaved changes exceed interval"""
        self._unsaved_changes += 1
        if self._unsaved_changes >= self.save_interval:
            self._save_index()

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text"""
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid text for embedding: {text}. Using empty string.")
            text = ""
        
        try:
            embedding = self.model.encode(text)
            return embedding.reshape(1, -1).astype('float32')
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros((1, self.embedding_dim), dtype=np.float32)

    async def add_document(
        self,
        doc_id: str,
        text: str,
        doc_type: str,
        metadata: Dict[str, Any]
    ) -> int:
        """
        Add a document to the vector database
        
        Args:
            doc_id: Unique identifier for the document
            text: Text content to be embedded
            doc_type: Type of document (e.g., "paper", "summary", etc.)
            metadata: Additional document metadata
            
        Returns:
            Index position of the added document
        """
        # Run the CPU-intensive embedding and index operations in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._add_document_sync, doc_id, text, doc_type, metadata)
    
    def _add_document_sync(self, doc_id: str, text: str, doc_type: str, metadata: Dict[str, Any]) -> int:
        """Synchronous version of add_document for thread pool execution"""
        with self._lock:
            try:
                # Check if index needs training (for IVF, PQ, etc.)
                if not self.index.is_trained and hasattr(self.index, 'train'):
                    # For first-time training, we need a sample vector
                    sample_vector = self._get_embedding(text)
                    logger.info(f"Training index with first vector")
                    self.index.train(sample_vector)
                
                embedding = self._get_embedding(text)
                
                # Check for duplicate document ID
                for i, doc in enumerate(self.metadata):
                    if doc["id"] == doc_id:
                        logger.warning(f"Document with ID {doc_id} already exists. Replacing.")
                        # For non-Flat indices, we can't directly replace vectors
                        # So we'll mark old as deleted and add new one at the end
                        self.metadata[i]["is_deleted"] = True
                        break
                        
                idx = self.index.ntotal
                self.index.add(embedding)
                
                self.metadata.append({
                    "id": doc_id,
                    "type": doc_type,
                    "text": text,
                    "metadata": metadata,
                    "added_at": datetime.now().isoformat(),
                    "is_deleted": False
                })
                
                self._maybe_save()
                return idx
            except Exception as e:
                logger.error(f"Error adding document {doc_id}: {e}")
                raise

    async def search(self, query: str, limit: int = 10, doc_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Query text to search for
            limit: Maximum number of results to return
            doc_type: Optional filter by document type
            
        Returns:
            List of matching documents with similarity scores
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._search_sync, query, limit, doc_type)
    
    def _search_sync(self, query: str, limit: int = 10, doc_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Synchronous version of search for thread pool execution"""
        try:
            query_embedding = self._get_embedding(query)
            
            # We might need more results than limit if some are filtered out by type
            fetch_limit = limit * 3 if doc_type else limit
            fetch_limit = min(fetch_limit, self.index.ntotal)
            
            if fetch_limit == 0:
                return []
                
            distances, indices = self.index.search(query_embedding, fetch_limit)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx == -1 or idx >= len(self.metadata):
                    continue
                    
                doc = self.metadata[idx]
                
                # Skip deleted documents
                if doc.get("is_deleted", False):
                    continue
                    
                # Filter by doc_type if specified
                if doc_type and doc["type"] != doc_type:
                    continue
                
                # Convert distance to similarity score (assuming L2 distance)
                # For L2 distance: smaller is better, so we invert and normalize
                # This gives a score between 0 and 1 where 1 is perfect match
                distance = float(distances[0][i])
                max_distance = 2.0  # Typical maximum L2 distance for normalized embeddings
                similarity = max(0.0, 1.0 - (distance / max_distance))
                
                results.append({
                    "score": similarity,
                    "id": doc["id"],
                    "type": doc["type"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "added_at": doc.get("added_at", "")
                })
                
                if len(results) >= limit:
                    break
                    
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []

    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document by ID
        
        For FAISS, we can't easily delete vectors, so we mark them as deleted
        in metadata and rebuild the index periodically or on demand
        
        Args:
            doc_id: ID of document to delete
            
        Returns:
            True if document was found and marked for deletion, False otherwise
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._delete_document_sync, doc_id)
    
    def _delete_document_sync(self, doc_id: str) -> bool:
        """Synchronous version of delete_document for thread pool execution"""
        with self._lock:
            # First try soft deletion by marking in metadata
            found = False
            for i, doc in enumerate(self.metadata):
                if doc["id"] == doc_id:
                    self.metadata[i]["is_deleted"] = True
                    found = True
                    break
            
            if not found:
                logger.warning(f"Document {doc_id} not found for deletion")
                return False
                
            # Count deleted documents
            deleted_count = sum(1 for doc in self.metadata if doc.get("is_deleted", False))
            total_count = len(self.metadata)
            
            # If too many deleted documents (>25%), rebuild the index
            if deleted_count > total_count * 0.25 and total_count > 100:
                self._rebuild_index()
            else:
                # Otherwise just save the updated metadata
                self._save_index()
                
            return True
    
    def _rebuild_index(self) -> None:
        """Rebuild index without deleted documents"""
        logger.info(f"Rebuilding index to remove {sum(1 for doc in self.metadata if doc.get('is_deleted', False))} deleted documents")
        
        try:
            # Create new index with same parameters
            new_index = faiss.index_factory(self.embedding_dim, self.index_factory_str)
            new_metadata = []
            
            # Get documents to keep
            active_docs = [doc for doc in self.metadata if not doc.get("is_deleted", False)]
            
            # If we have a trainable index and documents
            if not new_index.is_trained and hasattr(new_index, 'train') and active_docs:
                # Collect embeddings for training
                train_embeddings = np.zeros((min(len(active_docs), 10000), self.embedding_dim), dtype=np.float32)
                for i, doc in enumerate(active_docs[:10000]):
                    emb = self._get_embedding(doc["text"])
                    train_embeddings[i] = emb
                
                logger.info(f"Training new index with {len(train_embeddings)} vectors")
                new_index.train(train_embeddings)
            
            # Add vectors to new index
            for doc in active_docs:
                emb = self._get_embedding(doc["text"])
                new_index.add(emb)
                new_metadata.append(doc)
            
            # Replace old index and metadata
            self.index = new_index
            self.metadata = new_metadata
            
            # Save updated index
            self._save_index()
            logger.info(f"Index rebuilt with {len(new_metadata)} documents")
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            # Continue using the old index
    
    async def bulk_add_documents(self, documents: List[Dict[str, Any]]) -> List[int]:
        """
        Add multiple documents at once, more efficient than individual adds
        
        Args:
            documents: List of document dicts with id, text, doc_type and metadata fields
            
        Returns:
            List of indices where documents were added
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._bulk_add_documents_sync, documents)
    
    def _bulk_add_documents_sync(self, documents: List[Dict[str, Any]]) -> List[int]:
        """Synchronous version of bulk_add_documents"""
        with self._lock:
            try:
                # Prepare all embeddings
                embeddings = np.zeros((len(documents), self.embedding_dim), dtype=np.float32)
                for i, doc in enumerate(documents):
                    emb = self._get_embedding(doc["text"])
                    embeddings[i] = emb[0]  # Remove the batch dimension
                
                # Check if index needs training
                if not self.index.is_trained and hasattr(self.index, 'train') and len(embeddings) > 0:
                    logger.info(f"Training index with {len(embeddings)} vectors")
                    self.index.train(embeddings)
                
                # Track starting index for return value
                start_idx = self.index.ntotal
                
                # Add all vectors at once
                self.index.add(embeddings)
                
                # Update metadata
                indices = []
                for i, doc in enumerate(documents):
                    idx = start_idx + i
                    self.metadata.append({
                        "id": doc["id"],
                        "type": doc["doc_type"],
                        "text": doc["text"],
                        "metadata": doc["metadata"],
                        "added_at": datetime.now().isoformat(),
                        "is_deleted": False
                    })
                    indices.append(idx)
                
                # Save after bulk operation
                self._save_index()
                return indices
                
            except Exception as e:
                logger.error(f"Error in bulk add documents: {e}")
                return []

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document data or None if not found
        """
        for doc in self.metadata:
            if doc["id"] == doc_id and not doc.get("is_deleted", False):
                return {
                    "id": doc["id"],
                    "type": doc["type"],
                    "text": doc["text"],
                    "metadata": doc["metadata"],
                    "added_at": doc.get("added_at", "")
                }
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        active_docs = sum(1 for doc in self.metadata if not doc.get("is_deleted", False))
        deleted_docs = sum(1 for doc in self.metadata if doc.get("is_deleted", False))
        
        doc_types = {}
        for doc in self.metadata:
            if doc.get("is_deleted", False):
                continue
            doc_type = doc["type"]
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
        return {
            "total_vectors": self.index.ntotal,
            "active_documents": active_docs,
            "deleted_documents": deleted_docs,
            "document_types": doc_types,
            "embedding_dimension": self.embedding_dim,
            "index_type": self.index_factory_str,
            "unsaved_changes": self._unsaved_changes
        }

    def close(self):
        """Ensure final save on shutdown"""
        if self._unsaved_changes > 0:
            try:
                self._save_index()
                logger.info("Vector database closed and saved successfully")
            except Exception as e:
                logger.error(f"Error saving index on close: {e}")
