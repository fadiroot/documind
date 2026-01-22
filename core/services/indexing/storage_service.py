"""Storage service for uploading and managing documents in Azure AI Search."""
import time
from typing import List, Dict, Any
from core.utils.azure_utils import get_search_client
from core.utils.logger import logger
from app.config import settings


class StorageService:
    """Service for storing documents in Azure AI Search index."""
    
    def __init__(self):
        self.search_client = get_search_client()
        self.index_name = settings.AZURE_AI_SEARCH_INDEX_NAME or settings.AZURE_SEARCH_INDEX_NAME
    
    def upload_documents(self, documents: List[Dict[str, Any]], batch_size: int = 50, max_retries: int = 3) -> bool:
        """
        Upload documents to the search index with batching and retry logic.
        
        Args:
            documents: List of document dictionaries with id, content, contentVector, etc.
            batch_size: Number of documents to upload per batch (default: 50)
            max_retries: Maximum number of retry attempts for failed batches (default: 3)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.search_client:
            logger.error("Search client not available")
            return False
        
        if not documents:
            logger.warning("No documents to upload")
            return True
        
        total_docs = len(documents)
        total_uploaded = 0
        total_failed = 0
        
        # Split into batches
        batches = [documents[i:i + batch_size] for i in range(0, total_docs, batch_size)]
        logger.info(f"Uploading {total_docs} documents in {len(batches)} batch(es) of up to {batch_size} documents each")
        
        for batch_idx, batch in enumerate(batches, 1):
            batch_success = False
            retry_count = 0
            
            while retry_count <= max_retries and not batch_success:
                try:
                    if retry_count > 0:
                        wait_time = min(2 ** retry_count, 30)  # Exponential backoff, max 30 seconds
                        logger.info(f"  Retrying batch {batch_idx}/{len(batches)} (attempt {retry_count + 1}/{max_retries + 1}) after {wait_time}s...")
                        time.sleep(wait_time)
                    
                    result = self.search_client.upload_documents(documents=batch)
                    
                    # Check for errors in the result
                    failed = [r for r in result if not r.succeeded]
                    succeeded = [r for r in result if r.succeeded]
                    
                    if failed:
                        logger.warning(f"  Batch {batch_idx}/{len(batches)}: {len(succeeded)} succeeded, {len(failed)} failed")
                        for fail in failed[:5]:  # Log first 5 failures
                            logger.warning(f"    Failed document ID: {fail.key}, Error: {fail.error_message}")
                        if len(failed) > 5:
                            logger.warning(f"    ... and {len(failed) - 5} more failures")
                    else:
                        logger.info(f"  ✓ Batch {batch_idx}/{len(batches)}: Successfully uploaded {len(succeeded)} documents")
                    
                    total_uploaded += len(succeeded)
                    total_failed += len(failed)
                    batch_success = True
                    
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)
                    
                    # Check if it's an SSL/network error
                    is_ssl_error = any(keyword in error_msg.lower() for keyword in [
                        'ssl', 'unexpected eof', 'connection', 'timeout', 'network', 'socket'
                    ])
                    
                    if retry_count <= max_retries:
                        if is_ssl_error:
                            logger.warning(f"  Batch {batch_idx}/{len(batches)}: SSL/Network error (attempt {retry_count}/{max_retries}): {error_msg[:200]}")
                        else:
                            logger.warning(f"  Batch {batch_idx}/{len(batches)}: Error (attempt {retry_count}/{max_retries}): {error_msg[:200]}")
                    else:
                        logger.error(f"  ✗ Batch {batch_idx}/{len(batches)}: Failed after {max_retries + 1} attempts: {error_msg}")
                        total_failed += len(batch)
                        batch_success = False  # Mark as failed but continue with next batch
        
        # Summary
        if total_failed == 0:
            logger.info(f"✓ Successfully uploaded all {total_uploaded} documents to index '{self.index_name}'")
            return True
        elif total_uploaded > 0:
            logger.warning(f"⚠ Partially successful: {total_uploaded} uploaded, {total_failed} failed out of {total_docs} total")
            return False
        else:
            logger.error(f"✗ Failed to upload any documents: {total_failed} failed out of {total_docs} total")
            return False
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the search index.
        
        Args:
            document_id: ID of the document to delete
        
        Returns:
            True if successful, False otherwise
        """
        if not self.search_client:
            logger.error("Search client not available")
            return False
        
        try:
            self.search_client.delete_documents(documents=[{"id": document_id}])
            logger.info(f"Deleted document '{document_id}' from index '{self.index_name}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    def delete_documents_by_source(self, source_file: str) -> bool:
        """
        Delete all documents from a specific source file.
        
        Args:
            source_file: Name of the source file
        
        Returns:
            True if successful, False otherwise
        """
        if not self.search_client:
            logger.error("Search client not available")
            return False
        
        try:
            # Search for all documents with this source file
            # Note: This creates a circular import if done at module level
            # Import here to avoid circular dependency
            from core.services.retrieval.search_service import SearchService
            search_service = SearchService()
            # Use source_document field (or document_name as fallback)
            results = search_service.search_by_filter(f"source_document eq '{source_file}' or document_name eq '{source_file}'", top_k=1000)
            
            if not results:
                logger.info(f"No documents found for source file '{source_file}'")
                return True
            
            # Delete all found documents
            document_ids = [{"id": doc.get("id")} for doc in results if doc.get("id")]
            if document_ids:
                self.search_client.delete_documents(documents=document_ids)
                logger.info(f"Deleted {len(document_ids)} documents for source file '{source_file}'")
            
            return True
        except Exception as e:
            logger.error(f"Error deleting documents by source: {str(e)}")
            return False
