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
from threading import Lock # Use threading Lock for CPU-bound sync code called from async

# Configure logging (ideally configured at application entry point)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDatabase:
    """
    Handles vector database operations using FAISS and Sentence Transformers.

    Manages embeddings, indexing, metadata storage, searching, and persistence.
    Uses async wrappers around synchronous, CPU-bound operations (embedding, FAISS).
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str = "./data/vector_index.faiss", # Added .faiss extension
        metadata_path: str = "./data/vector_metadata.pkl", # Added .pkl extension
        save_interval: int = 100, # Save after this many additions/deletions
        index_factory_string: str = "Flat", # FAISS index factory string (e.g., "Flat", "IVF100,Flat", "IndexHNSWFlat")
        rebuild_threshold: float = 0.25 # Rebuild index if deleted items exceed this fraction
    ):
        """
        Initializes the VectorDatabase.

        Args:
            embedding_model_name: Name of the Sentence Transformer model to use.
            index_path: Path to save/load the FAISS index file.
            metadata_path: Path to save/load the metadata file (pickle).
            save_interval: Number of changes before automatically saving the index.
            index_factory_string: FAISS index type string. "Flat" (exact search) is default.
                                   Consider "IndexFlatIP" for cosine similarity if model prefers it,
                                   or IVF/HNSW types for larger datasets.
            rebuild_threshold: Fraction of deleted items that triggers an index rebuild.
        """
        self.embedding_model_name = embedding_model_name
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.save_interval = max(1, save_interval) # Ensure interval is at least 1
        self.index_factory_string = index_factory_string
        self.rebuild_threshold = rebuild_threshold
        self._lock = Lock() # Lock to protect index/metadata access from concurrent threads
        self._unsaved_changes = 0

        # Ensure data directories exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        # Initialize the embedding model
        try:
            # Consider adding device='cuda' if GPU is available and faiss-gpu is installed
            self.model = sentence_transformers.SentenceTransformer(embedding_model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized embedding model '{embedding_model_name}' with dimension {self.embedding_dim}")
        except Exception as e:
            logger.exception(f"Fatal: Failed to load embedding model '{embedding_model_name}': {e}")
            raise # Critical error, stop initialization

        # Load or create index and metadata
        self.index, self.metadata = self._load_or_create_index()

    def _load_or_create_index(self) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """Loads index/metadata from disk or creates new ones if not found or invalid."""
        with self._lock: # Ensure exclusive access during load/create
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                try:
                    logger.info(f"Attempting to load index from: {self.index_path}")
                    index = faiss.read_index(self.index_path)
                    logger.info(f"Attempting to load metadata from: {self.metadata_path}")
                    with open(self.metadata_path, 'rb') as f:
                        metadata = pickle.load(f)

                    # --- Sanity Checks ---
                    # Check if index dimension matches the loaded model's dimension
                    if index.d != self.embedding_dim:
                        logger.warning(f"Index dimension ({index.d}) differs from model dimension ({self.embedding_dim}). Recreating index.")
                        raise ValueError("Dimension mismatch")

                    # Check if number of vectors in index matches metadata length
                    # Note: ntotal might not reflect soft deletes until rebuild
                    active_metadata_count = sum(1 for doc in metadata if not doc.get("is_deleted", False))
                    # This check is less reliable with soft deletes, focus on dimension match primarily.
                    # if index.ntotal != len(metadata): # This check might fail if soft deletes exist
                    #    logger.warning(f"Index ntotal ({index.ntotal}) differs from metadata length ({len(metadata)}). Metadata might be inconsistent.")
                        # Decide how to handle inconsistency: trust metadata? rebuild? For now, log warning.

                    logger.info(f"Successfully loaded existing vector index with {index.ntotal} vectors and {len(metadata)} metadata entries ({active_metadata_count} active).")
                    return index, metadata

                except (FileNotFoundError, EOFError, pickle.UnpicklingError, ValueError, AttributeError, RuntimeError) as e:
                    logger.warning(f"Could not load existing index/metadata (path: {self.index_path}, error: {e}). Creating new ones.")
                except Exception as e: # Catch other potential FAISS or I/O errors
                     logger.exception(f"Unexpected error loading index/metadata: {e}. Creating new ones.")


            # --- Create New Index ---
            try:
                logger.info(f"Creating new FAISS index with factory string: '{self.index_factory_string}' and dimension {self.embedding_dim}")
                index = faiss.index_factory(self.embedding_dim, self.index_factory_string)
                metadata = []
                logger.info("New index and metadata created successfully.")
                return index, metadata
            except Exception as e:
                logger.exception(f"Fatal: Failed to create new FAISS index with factory '{self.index_factory_string}': {e}")
                logger.warning("Falling back to basic IndexFlatL2.")
                # Fallback to a simple index if factory string fails
                index = faiss.IndexFlatL2(self.embedding_dim)
                metadata = []
                return index, metadata

    def _save_index(self) -> None:
        """Saves the current index and metadata to disk atomically using temp files."""
        # This method assumes the lock is already held by the caller (_maybe_save, _rebuild_index, close)
        if self._unsaved_changes == 0:
             logger.debug("No unsaved changes, skipping save.")
             return

        logger.info(f"Saving index ({self.index.ntotal} vectors) and metadata ({len(self.metadata)} entries)...")
        temp_index_path = f"{self.index_path}.{os.getpid()}.tmp" # Include PID for concurrent safety
        temp_metadata_path = f"{self.metadata_path}.{os.getpid()}.tmp"

        try:
            # 1. Write to temporary files
            faiss.write_index(self.index, temp_index_path)
            with open(temp_metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f, protocol=pickle.HIGHEST_PROTOCOL) # Use highest protocol

            # 2. Atomically replace old files with new files
            os.replace(temp_index_path, self.index_path)
            os.replace(temp_metadata_path, self.metadata_path)

            logger.info(f"Successfully saved index to {self.index_path} and metadata to {self.metadata_path}")
            self._unsaved_changes = 0 # Reset counter after successful save

        except Exception as e:
            logger.exception(f"Error saving index/metadata: {e}")
            # Attempt to clean up temporary files if they exist
            for path in [temp_index_path, temp_metadata_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        logger.info(f"Removed temporary file: {path}")
                    except OSError as rm_err:
                        logger.error(f"Error removing temporary file {path}: {rm_err}")
            # Re-raise the exception so the caller knows saving failed
            raise

    def _maybe_save(self) -> None:
        """Checks if the number of unsaved changes exceeds the interval and saves if needed."""
        # This method assumes the lock is already held by the caller (add/delete operations)
        self._unsaved_changes += 1
        logger.debug(f"Unsaved changes count: {self._unsaved_changes}")
        if self._unsaved_changes >= self.save_interval:
            logger.info(f"Save interval ({self.save_interval}) reached. Triggering save.")
            try:
                self._save_index()
            except Exception:
                 # Logged in _save_index, just note failure here if needed
                 logger.error("Failed to save index during periodic check.")
                 # Keep _unsaved_changes high so it tries again next time

    def _get_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generates embeddings for a single text or a list of texts.

        Args:
            text: A single string or a list of strings.

        Returns:
            A numpy array of embeddings (shape [1, dim] or [N, dim]).
            Returns zero vector(s) on error.
        """
        if not text:
            logger.warning("Received empty input for embedding. Returning zero vector(s).")
            if isinstance(text, list):
                 # Need to return array with correct shape for list input
                 num_texts = len(text) if text else 1 # Handle empty list case
                 return np.zeros((num_texts, self.embedding_dim), dtype=np.float32)
            else:
                 return np.zeros((1, self.embedding_dim), dtype=np.float32)

        try:
            # model.encode handles both single string and list of strings
            # normalize_embeddings=True might be beneficial if using IndexFlatIP or for consistent L2 distances
            embeddings = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=False) # Set normalize_embeddings based on index type?
            # Ensure output is float32 and has the correct shape
            if embeddings.ndim == 1: # Single text input
                return embeddings.reshape(1, -1).astype('float32')
            else: # List of texts input
                return embeddings.astype('float32')
        except Exception as e:
            logger.exception(f"Error generating embedding for text: '{str(text)[:100]}...': {e}")
            # Return zero vector(s) with appropriate shape
            if isinstance(text, list):
                 return np.zeros((len(text), self.embedding_dim), dtype=np.float32)
            else:
                 return np.zeros((1, self.embedding_dim), dtype=np.float32)

    async def add_document(
        self,
        doc_id: str,
        text: str,
        doc_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Adds or replaces a single document in the vector database.

        Args:
            doc_id: A unique identifier for the document. IMPORTANT: Choose a consistent
                    strategy, e.g., "paper_{sql_id}_abstract", "paper_{sql_id}_summary",
                    or the paper's unique URL.
            text: The text content to be embedded (e.g., abstract or summary).
            doc_type: A category for the document (e.g., "abstract", "summary"). Used for filtering.
            metadata: Optional dictionary of additional data to store alongside the vector.
                      Crucially, include identifiers to link back to the SQL DB,
                      e.g., {"sql_paper_id": 123, "url": "http://..."}.

        Returns:
            The internal index position (offset) where the document vector was added,
            or None if the addition failed.
        """
        if not all([doc_id, text, doc_type]):
             logger.error(f"Missing required arguments for add_document: doc_id={doc_id}, text_provided={bool(text)}, doc_type={doc_type}")
             return None

        # Run the potentially blocking/CPU-intensive sync logic in a thread pool
        loop = asyncio.get_event_loop()

        try:
            # Pass metadata or an empty dict if None, capture the returned doc_id
            result_doc_id = await loop.run_in_executor(
                None, self._add_document_sync, doc_id, text, doc_type, metadata or {}
            )
            return result_doc_id
        
        except Exception as e:
             # Error should be logged within _add_document_sync, just return None here
             logger.error(f"Async wrapper caught exception during add_document for doc_id {doc_id}: {e}")
             return None

    def _add_document_sync(self, doc_id: str, text: str, doc_type: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Synchronous internal logic for adding a document."""
        with self._lock: # Ensure exclusive access to index and metadata
            try:
                embedding = self._get_embedding(text)
                # Check if embedding generation failed (returned zeros)
                if np.all(embedding == 0):
                     logger.error(f"Failed to add document {doc_id} due to embedding generation failure.")
                     return None # Indicate failure

                # --- Handle potential existing document with the same ID ---
                existing_idx = -1
                for i, meta_entry in enumerate(self.metadata):
                    if meta_entry["id"] == doc_id:
                        logger.warning(f"Document with ID '{doc_id}' already exists at index {i}. Marking old entry as deleted and adding new one.")
                        # Mark the old entry as deleted (soft delete)
                        self.metadata[i]["is_deleted"] = True
                        existing_idx = i # Keep track if needed, though we add at the end
                        break

                # --- Check if index needs training (only before adding first vector or during rebuild) ---
                # This check is mainly relevant for IVF/PQ indices. Flat indices don't need training.
                if self.index.ntotal == 0 and not self.index.is_trained and hasattr(self.index, 'train'):
                    logger.info(f"Index type '{self.index_factory_string}' requires training. Training with the first vector.")
                    try:
                        self.index.train(embedding) # Train with the first embedding
                        logger.info("Index training complete.")
                    except Exception as train_e:
                        logger.exception(f"Failed to train index: {train_e}")
                        return None # Cannot add if training fails

                # --- Add the new vector and metadata ---
                current_index_pos = self.index.ntotal  # Get index before adding
                self.index.add(embedding)

                # Verify vector was added
                if self.index.ntotal <= current_index_pos:
                     logger.error(f"FAISS index count did not increase after adding vector for doc_id {doc_id}. Addition failed.")
                     # Indicate failure
                     return None

                # Check if metadata already exists (redundant check if existing ID handling above works, but safe)
                meta_exists = any(m['id'] == doc_id for m in self.metadata if not m.get('is_deleted'))
                if meta_exists:
                    logger.warning(f"Metadata for {doc_id} seems to exist already despite delete/add logic. Overwriting/appending.")
                    # Consider how to handle this case if it occurs. For now, just append.
                    
                self.metadata.append({
                    "id": doc_id,
                    "type": doc_type,
                    "text": text, # Store the original text? Optional, depends on use case.
                    "metadata": metadata, # Store provided metadata (e.g., sql_paper_id)
                    "added_at": datetime.utcnow().isoformat(),
                    "is_deleted": False # Mark as active
                })

                logger.info(f"Successfully added document '{doc_id}' (type: {doc_type}) at index {current_index_pos}.")
                self._maybe_save() # Check if saving is needed
                return doc_id  # Return doc_id on success

            except Exception as e:
                # Catch-all for unexpected errors during the process
                logger.exception(f"Unexpected error adding document {doc_id}: {e}")
                # Don't re-raise here, return None to indicate failure to the async wrapper
                return None


    async def search(self, query: str, limit: int = 10, doc_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Searches the vector database for documents similar to the query text.

        Args:
            query: The text to search for.
            limit: The maximum number of results to return.
            doc_type_filter: Optional string to filter results by document type
                             (e.g., "abstract", "summary").

        Returns:
            A list of result dictionaries, sorted by similarity score (highest first).
            Each dictionary contains: 'score', 'id', 'type', 'text', 'metadata', 'added_at'.
            Returns an empty list if no results or an error occurs.
        """
        if not query:
             logger.warning("Search query is empty.")
             return []
        if self.index.ntotal == 0:
             logger.warning("Search attempted on an empty index.")
             return []

        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None, self._search_sync, query, limit, doc_type_filter
            )
            return results
        except Exception as e:
             logger.error(f"Async wrapper caught exception during search for query '{query[:50]}...': {e}")
             return []

    def _search_sync(self, query: str, limit: int, doc_type_filter: Optional[str]) -> List[Dict[str, Any]]:
        """Synchronous internal logic for searching."""
        try:
            query_embedding = self._get_embedding(query)
            if np.all(query_embedding == 0):
                 logger.error("Failed to search: Query embedding generation failed.")
                 return []

            # Determine how many neighbors to fetch initially.
            # Need potentially more than 'limit' if filtering is applied.
            # Increase fetch_k significantly if filtering is active.
            fetch_k = limit * 5 if doc_type_filter else limit * 2
            # Ensure fetch_k is not more than the total number of items in the index.
            fetch_k = min(fetch_k, self.index.ntotal)

            if fetch_k <= 0: # Handle case where index is empty after check
                 logger.warning("Search sync: Index has no vectors to search.")
                 return []

            logger.debug(f"Searching index for '{query[:50]}...' with k={fetch_k}")
            # Perform the FAISS search
            # distances: L2 squared distances (lower is better)
            # indices: The internal FAISS indices of the neighbors
            distances, indices = self.index.search(query_embedding, fetch_k)

            results = []
            processed_indices = set() # Keep track of indices already processed

            # Iterate through the raw FAISS results
            for i, idx in enumerate(indices[0]): # indices[0] because query batch size is 1
                # FAISS can return -1 for invalid indices
                if idx == -1 or idx in processed_indices:
                    continue

                processed_indices.add(idx)

                # --- Validate Metadata ---
                if idx >= len(self.metadata):
                    logger.warning(f"Search result index {idx} is out of bounds for metadata (len: {len(self.metadata)}). Skipping.")
                    continue

                doc_meta = self.metadata[idx]

                # --- Apply Filters ---
                # Skip documents marked as deleted
                if doc_meta.get("is_deleted", False):
                    logger.debug(f"Skipping search result {idx} (ID: {doc_meta.get('id', 'N/A')}) because it's marked as deleted.")
                    continue

                # Apply document type filter if provided
                if doc_type_filter and doc_meta.get("type") != doc_type_filter:
                    logger.debug(f"Skipping search result {idx} (ID: {doc_meta.get('id', 'N/A')}, Type: {doc_meta.get('type')}) due to type filter '{doc_type_filter}'.")
                    continue

                # --- Calculate Similarity Score ---
                # Assuming L2 distance from a FlatL2 index. Lower distance = higher similarity.
                # Convert distance to a similarity score (e.g., 0 to 1).
                # Simple inversion: score = 1 / (1 + distance). Avoids division by zero.
                # Or normalize based on expected distance range if embeddings are normalized.
                distance = float(distances[0][i])
                similarity_score = 1.0 / (1.0 + distance) # Simple inversion score

                # Alternative for normalized embeddings (max L2 distance approx 2):
                # max_dist_sq = 4.0 # Max L2 squared distance for normalized vectors is 4
                # similarity_score = max(0.0, 1.0 - (distance / max_dist_sq))

                # --- Append to Results ---
                results.append({
                    "score": similarity_score,
                    "id": doc_meta.get("id", f"missing_id_at_idx_{idx}"), # Provide fallback ID
                    "type": doc_meta.get("type", "unknown"),
                    "text": doc_meta.get("text", ""), # Include text if stored
                    "metadata": doc_meta.get("metadata", {}),
                    "added_at": doc_meta.get("added_at", "")
                })

                # Stop if we have reached the desired number of results
                if len(results) >= limit:
                    break

            # Sort final results by score descending (highest similarity first)
            results.sort(key=lambda x: x["score"], reverse=True)

            logger.info(f"Search for '{query[:50]}...' found {len(results)} results after filtering (fetched {fetch_k}).")
            return results

        except Exception as e:
            logger.exception(f"Unexpected error during search sync for query '{query[:50]}...': {e}")
            return []

    async def delete_document(self, doc_id: str) -> bool:
        """
        Marks a document for deletion using its unique ID.

        Deletion is "soft" - the document metadata is marked, but the vector
        remains in the FAISS index until the index is rebuilt.
        Rebuilding happens automatically if the proportion of deleted documents
        exceeds `rebuild_threshold`.

        Args:
            doc_id: The unique ID of the document to mark for deletion.

        Returns:
            True if the document was found and marked, False otherwise.
        """
        if not doc_id:
             logger.error("Cannot delete document: doc_id is missing.")
             return False

        loop = asyncio.get_event_loop()
        try:
            deleted = await loop.run_in_executor(None, self._delete_document_sync, doc_id)
            return deleted
        except Exception as e:
             logger.error(f"Async wrapper caught exception during delete_document for doc_id {doc_id}: {e}")
             return False

    def _delete_document_sync(self, doc_id: str) -> bool:
        """Synchronous internal logic for marking a document as deleted."""
        with self._lock: # Ensure exclusive access
            found_and_marked = False
            active_count = 0
            deleted_count = 0

            # Iterate through metadata to find the document and mark it
            for i, doc in enumerate(self.metadata):
                is_deleted = doc.get("is_deleted", False)
                if doc["id"] == doc_id and not is_deleted:
                    logger.info(f"Marking document ID '{doc_id}' at index {i} as deleted.")
                    self.metadata[i]["is_deleted"] = True
                    found_and_marked = True
                    deleted_count += 1 # Count this newly marked one
                    self._unsaved_changes += 1 # Increment changes for potential save
                elif is_deleted:
                    deleted_count += 1
                else:
                    active_count += 1

            if not found_and_marked:
                logger.warning(f"Document ID '{doc_id}' not found or already marked as deleted.")
                return False # Nothing was changed

            # --- Check if index rebuild is needed ---
            total_count = active_count + deleted_count
            if total_count > 0 and (deleted_count / total_count) > self.rebuild_threshold:
                logger.info(f"Deleted document threshold ({self.rebuild_threshold:.1%}) exceeded ({deleted_count}/{total_count} deleted). Triggering index rebuild.")
                try:
                    # Rebuild acquires the lock again internally, but that's okay for threading.Lock
                    self._rebuild_index() # Rebuild also saves the index
                except Exception as rebuild_e:
                     logger.error(f"Index rebuild failed after marking {doc_id} for deletion: {rebuild_e}")
                     # Even if rebuild fails, the soft delete is marked. Attempt to save metadata.
                     try:
                          self._save_index()
                     except Exception as save_e:
                          logger.error(f"Failed to save metadata after failed rebuild: {save_e}")

            else:
                # If not rebuilding, check if periodic save is needed
                logger.debug(f"Deleted count ({deleted_count}/{total_count}) below threshold. Checking for periodic save.")
                self._maybe_save() # Will save if interval reached

            return True # Document was found and marked

    def _rebuild_index(self) -> None:
        """
        Rebuilds the FAISS index from scratch using only non-deleted documents.
        Warning: This can be computationally expensive for large datasets.
        """
        # This method assumes the lock is already held by the caller (_delete_document_sync)
        logger.info("Starting index rebuild process...")

        try:
            # 1. Filter out deleted documents from metadata
            active_metadata = [doc for doc in self.metadata if not doc.get("is_deleted", False)]
            num_active_docs = len(active_metadata)
            num_deleted = len(self.metadata) - num_active_docs
            logger.info(f"Rebuilding index with {num_active_docs} active documents (removing {num_deleted} deleted entries).")

            if num_active_docs == 0:
                 logger.warning("No active documents left. Creating an empty index.")
                 # Create a new empty index of the same type
                 new_index = faiss.index_factory(self.embedding_dim, self.index_factory_string)
                 new_metadata = []
            else:
                # 2. Create a new FAISS index
                new_index = faiss.index_factory(self.embedding_dim, self.index_factory_string)

                # 3. Generate embeddings for all active documents
                logger.info(f"Generating embeddings for {num_active_docs} active documents...")
                active_texts = [doc["text"] for doc in active_metadata]
                # Process embeddings in batches if memory is a concern
                # For simplicity here, process all at once. Adjust if needed.
                active_embeddings = self._get_embedding(active_texts)

                if active_embeddings.shape[0] != num_active_docs:
                     logger.error(f"Embedding generation mismatch during rebuild: Expected {num_active_docs}, Got {active_embeddings.shape[0]}. Aborting rebuild.")
                     # Don't modify the current index/metadata if embeddings fail
                     return

                # 4. Train the new index if necessary (e.g., for IVF)
                if not new_index.is_trained and hasattr(new_index, 'train'):
                    logger.info(f"Training new index with {num_active_docs} active vectors...")
                    try:
                        new_index.train(active_embeddings)
                        logger.info("Index training complete.")
                    except Exception as train_e:
                        logger.exception(f"Failed to train new index during rebuild: {train_e}. Aborting rebuild.")
                        return

                # 5. Add embeddings to the new index
                logger.info("Adding embeddings to the new index...")
                new_index.add(active_embeddings)

                # Sanity check add operation
                if new_index.ntotal != num_active_docs:
                     logger.error(f"Index count mismatch after adding vectors during rebuild: Expected {num_active_docs}, Got {new_index.ntotal}. Aborting rebuild.")
                     return

                new_metadata = active_metadata # Metadata is already filtered

            # 6. Replace the old index and metadata with the new ones
            self.index = new_index
            self.metadata = new_metadata
            self._unsaved_changes = 1 # Mark as changed so save occurs

            # 7. Save the newly rebuilt index and metadata immediately
            logger.info("Index rebuild complete. Saving new index and metadata.")
            self._save_index() # This will reset _unsaved_changes to 0

        except Exception as e:
            # If rebuild fails catastrophically, log the error but crucially *do not*
            # replace the existing self.index and self.metadata. The old index (with
            # soft deletes) remains active.
            logger.exception(f"Catastrophic error during index rebuild: {e}. Original index remains active.")
            # Optionally, try to save the existing metadata again in case soft delete marks were made
            # try:
            #     self._save_index()
            # except: pass

    async def bulk_add_documents(self, documents: List[Dict[str, Any]]) -> List[Optional[int]]:
        """
        Adds multiple documents efficiently in a single operation.

        Args:
            documents: A list of dictionaries. Each dictionary must contain:
                       'id' (str): Unique document ID.
                       'text' (str): Text content.
                       'doc_type' (str): Document type.
                       'metadata' (dict, optional): Additional metadata.

        Returns:
            A list containing the index position for each added document,
            or None for documents that failed to add. The list order matches the input order.
        """
        if not documents:
             logger.warning("bulk_add_documents called with an empty list.")
             return []

        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(None, self._bulk_add_documents_sync, documents)
            return results
        except Exception as e:
             logger.error(f"Async wrapper caught exception during bulk_add_documents: {e}")
             # Return list of Nones matching input length
             return [None] * len(documents)

    def _bulk_add_documents_sync(self, documents: List[Dict[str, Any]]) -> List[Optional[int]]:
        """Synchronous internal logic for bulk adding documents."""
        with self._lock: # Ensure exclusive access
            added_indices: List[Optional[int]] = [None] * len(documents)
            valid_docs_to_add = []
            original_indices_map = {} # Map index in valid_docs_to_add back to original index

            # --- Pre-process and Validate Input ---
            doc_ids_to_process = set()
            for i, doc in enumerate(documents):
                doc_id = doc.get("id")
                text = doc.get("text")
                doc_type = doc.get("doc_type")

                if not all([doc_id, text, doc_type]):
                    logger.error(f"Skipping document at index {i} in bulk add due to missing fields: id={doc_id}, text_provided={bool(text)}, type={doc_type}")
                    continue
                if doc_id in doc_ids_to_process:
                     logger.warning(f"Duplicate doc_id '{doc_id}' found within the bulk request at index {i}. Skipping.")
                     continue

                # Check for existing ID in current metadata and mark for deletion
                for meta_idx, meta_entry in enumerate(self.metadata):
                     if meta_entry["id"] == doc_id and not meta_entry.get("is_deleted", False):
                          logger.warning(f"Document ID '{doc_id}' from bulk request already exists. Marking old entry at index {meta_idx} as deleted.")
                          self.metadata[meta_idx]["is_deleted"] = True
                          break # Found the one to mark

                valid_docs_to_add.append(doc)
                original_indices_map[len(valid_docs_to_add) - 1] = i
                doc_ids_to_process.add(doc_id)


            if not valid_docs_to_add:
                 logger.warning("No valid documents found in the bulk add request after validation.")
                 return added_indices # Return list of Nones

            num_valid_docs = len(valid_docs_to_add)
            logger.info(f"Processing {num_valid_docs} valid documents in bulk add.")

            try:
                # --- Generate Embeddings ---
                logger.info(f"Generating embeddings for {num_valid_docs} documents...")
                texts_to_embed = [doc["text"] for doc in valid_docs_to_add]
                embeddings = self._get_embedding(texts_to_embed)

                # Check for embedding failures
                if embeddings.shape[0] != num_valid_docs:
                     logger.error(f"Embedding count mismatch in bulk add: Expected {num_valid_docs}, Got {embeddings.shape[0]}. Aborting.")
                     return added_indices # Return Nones

                # --- Train Index (if needed) ---
                if self.index.ntotal == 0 and not self.index.is_trained and hasattr(self.index, 'train'):
                    logger.info(f"Training index with {num_valid_docs} vectors from bulk add.")
                    try:
                        self.index.train(embeddings)
                        logger.info("Index training complete.")
                    except Exception as train_e:
                        logger.exception(f"Failed to train index during bulk add: {train_e}")
                        return added_indices # Return Nones

                # --- Add Embeddings to Index ---
                start_idx = self.index.ntotal
                logger.info(f"Adding {num_valid_docs} embeddings to FAISS index...")
                self.index.add(embeddings)

                # Verify addition
                if self.index.ntotal != start_idx + num_valid_docs:
                     logger.error(f"Index count mismatch after bulk add: Expected {start_idx + num_valid_docs}, Got {self.index.ntotal}. Aborting metadata update.")
                     # This is tricky - vectors might be partially added. State is inconsistent.
                     # Best effort: return Nones, manual recovery might be needed.
                     return added_indices

                # --- Update Metadata ---
                logger.info("Updating metadata for bulk added documents...")
                for i, doc in enumerate(valid_docs_to_add):
                    current_faiss_idx = start_idx + i
                    original_idx = original_indices_map[i]
                    self.metadata.append({
                        "id": doc["id"],
                        "type": doc["doc_type"],
                        "text": doc["text"],
                        "metadata": doc.get("metadata", {}),
                        "added_at": datetime.utcnow().isoformat(),
                        "is_deleted": False
                    })
                    added_indices[original_idx] = current_faiss_idx # Store FAISS index in correct original slot

                # --- Save After Bulk Operation ---
                self._unsaved_changes += num_valid_docs # Increment changes count
                logger.info(f"Bulk add complete. Total unsaved changes: {self._unsaved_changes}. Triggering save.")
                self._save_index() # Save immediately after successful bulk add

                return added_indices

            except Exception as e:
                logger.exception(f"Unexpected error during bulk add sync processing: {e}")
                # Return list of Nones matching original input length
                return [None] * len(documents)


    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the metadata and text for a single active document by its unique ID.

        Args:
            doc_id: The unique ID of the document to retrieve.

        Returns:
            A dictionary containing the document's data ('id', 'type', 'text',
            'metadata', 'added_at') if found and active, otherwise None.
        """
        if not doc_id: return None
        # No lock needed for read-only metadata iteration if list append is atomic enough
        # For absolute safety, a read lock could be used, but often overkill for reads.
        for doc in self.metadata:
            # Check ID and ensure it's not marked as deleted
            if doc.get("id") == doc_id and not doc.get("is_deleted", False):
                # Return a copy to avoid external modification of internal metadata
                return {
                    "id": doc.get("id"),
                    "type": doc.get("type"),
                    "text": doc.get("text"),
                    "metadata": doc.get("metadata", {}).copy(), # Return copy of metadata
                    "added_at": doc.get("added_at", "")
                }
        logger.debug(f"Document with ID '{doc_id}' not found or is deleted.")
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics about the current state of the vector database."""
        with self._lock: # Lock for consistent reading of metadata and index state
            active_docs = sum(1 for doc in self.metadata if not doc.get("is_deleted", False))
            deleted_docs = len(self.metadata) - active_docs

            doc_types_counts = {}
            for doc in self.metadata:
                if not doc.get("is_deleted", False):
                    doc_type = doc.get("type", "unknown")
                    doc_types_counts[doc_type] = doc_types_counts.get(doc_type, 0) + 1

            return {
                "total_vectors_in_index": self.index.ntotal, # Actual count in FAISS index
                "total_metadata_entries": len(self.metadata),
                "active_documents": active_docs, # Count based on metadata flag
                "deleted_documents": deleted_docs, # Count based on metadata flag
                "document_type_distribution": doc_types_counts,
                "embedding_dimension": self.embedding_dim,
                "embedding_model": self.embedding_model_name,
                "index_type": self.index_factory_string,
                "is_index_trained": self.index.is_trained,
                "unsaved_changes": self._unsaved_changes,
                "index_file_path": self.index_path,
                "metadata_file_path": self.metadata_path
            }

    def close(self):
        """
        Attempts to save the index and metadata if there are unsaved changes.
        Should be called during application shutdown.
        """
        logger.info("VectorDatabase close method called.")
        with self._lock:
             if self._unsaved_changes > 0:
                 logger.info(f"Found {self._unsaved_changes} unsaved changes. Attempting final save.")
                 try:
                     self._save_index()
                     logger.info("Vector database closed and final save successful.")
                 except Exception as e:
                     # Logged in _save_index
                     logger.error(f"Error saving index/metadata on close: {e}")
             else:
                 logger.info("No unsaved changes. Vector database closed.")

# --- Example Usage (Optional - for testing this module directly) ---
async def example_vector_db_usage():
    print("--- VectorDatabase Example Usage ---")
    # Use different paths for testing to avoid overwriting production data
    test_index_path = "./data/test_vector_index.faiss"
    test_meta_path = "./data/test_vector_metadata.pkl"

    # Clean up previous test files if they exist
    if os.path.exists(test_index_path): os.remove(test_index_path)
    if os.path.exists(test_meta_path): os.remove(test_meta_path)

    # Initialize DB
    vdb = VectorDatabase(
        index_path=test_index_path,
        metadata_path=test_meta_path,
        save_interval=5 # Save more frequently for testing
    )

    try:
        # 1. Add documents
        doc1_id = "paper_123_abstract"
        await vdb.add_document(doc1_id, "This is the abstract of paper 123 about medicine.", "abstract", {"sql_paper_id": 123, "year": 2023})
        doc2_id = "paper_123_summary"
        await vdb.add_document(doc2_id, "A short summary of paper 123.", "summary", {"sql_paper_id": 123})
        doc3_id = "paper_456_abstract"
        await vdb.add_document(doc3_id, "Another paper abstract, this one discusses biology.", "abstract", {"sql_paper_id": 456, "year": 2024})

        print("\n--- Stats after adding 3 docs ---")
        print(vdb.get_stats())

        # 2. Search for similar documents
        print("\n--- Search results for 'medical paper' ---")
        results = await vdb.search("medical paper", limit=5)
        for res in results:
            print(f"  Score: {res['score']:.4f}, ID: {res['id']}, Type: {res['type']}")

        print("\n--- Search results for 'biology abstract' (filtered by type) ---")
        results_filtered = await vdb.search("biology abstract", limit=5, doc_type_filter="abstract")
        for res in results_filtered:
            print(f"  Score: {res['score']:.4f}, ID: {res['id']}, Type: {res['type']}") # Should only show abstracts

        # 3. Get a document by ID
        print("\n--- Get document by ID 'paper_123_summary' ---")
        retrieved_doc = vdb.get_document_by_id(doc2_id)
        print(retrieved_doc)

        # 4. Delete a document
        print("\n--- Deleting document 'paper_123_abstract' ---")
        delete_success = await vdb.delete_document(doc1_id)
        print(f"Deletion successful: {delete_success}")

        print("\n--- Stats after deletion ---")
        print(vdb.get_stats()) # Note: total_vectors_in_index won't decrease until rebuild

        # Verify deleted doc is not retrieved
        print("\n--- Attempt to get deleted document by ID 'paper_123_abstract' ---")
        retrieved_deleted_doc = vdb.get_document_by_id(doc1_id)
        print(f"Retrieved: {retrieved_deleted_doc}") # Should be None

        # Verify deleted doc is not in search results
        print("\n--- Search results for 'medicine' after deletion ---")
        results_after_delete = await vdb.search("medicine", limit=5)
        for res in results_after_delete:
             # Should not contain paper_123_abstract
            print(f"  Score: {res['score']:.4f}, ID: {res['id']}, Type: {res['type']}")


        # 5. Bulk add
        print("\n--- Bulk adding documents ---")
        bulk_docs = [
            {"id": "paper_789_abstract", "text": "Abstract about chemistry.", "doc_type": "abstract", "metadata": {"sql_paper_id": 789}},
            {"id": "paper_789_summary", "text": "Summary of chemistry paper.", "doc_type": "summary", "metadata": {"sql_paper_id": 789}},
            # Add a duplicate ID within the bulk request (should be skipped)
            {"id": "paper_789_summary", "text": "Duplicate summary.", "doc_type": "summary", "metadata": {"sql_paper_id": 789}},
             # Add one with missing field (should be skipped)
            {"id": "paper_000_abstract", "text": "Missing type field.", "metadata": {"sql_paper_id": 0}},

        ]
        bulk_results = await vdb.bulk_add_documents(bulk_docs)
        print(f"Bulk add results (indices): {bulk_results}") # Should show indices for valid adds, None for skips

        print("\n--- Stats after bulk add ---")
        print(vdb.get_stats())


    except Exception as e:
        logger.exception(f"Error during example usage: {e}")
    finally:
        # Ensure index is saved on exit/error during testing
        vdb.close()
        print("\n--- Example Usage Complete ---")


if __name__ == "__main__":
    # Requires asyncio to run the async example function
    import asyncio
    asyncio.run(example_vector_db_usage())

