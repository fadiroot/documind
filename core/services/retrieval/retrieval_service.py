"""Retrieval service for RAG operations."""
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

from core.services.retrieval.embedding_service import EmbeddingService
from core.services.retrieval.search_service import SearchService
from core.services.retrieval.retrieval_result import RetrievalResult
from core.utils.logger import logger


class RetrievalService:
    """Service for retrieving relevant documents for RAG with quality controls."""
    
    def __init__(self, min_score_threshold: float = 0.3, enable_reranking: bool = False):
        self.embedding_service = EmbeddingService()
        self.search_service = SearchService()
        self.min_score_threshold = min_score_threshold
        self.enable_reranking = enable_reranking
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: Optional[float] = None,
        filters: Optional[str] = None,
        max_context_length: int = 8000
    ) -> RetrievalResult:
        """
        Retrieve relevant documents with quality controls.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve (before filtering)
            min_score: Minimum relevance score threshold (defaults to instance threshold)
            filters: Optional filters
            max_context_length: Maximum total context length in characters
        
        Returns:
            RetrievalResult with filtered documents
        """
        min_score = min_score or self.min_score_threshold
        
        # Create query embedding
        query_embedding = self.embedding_service.create_embedding(query)
        if not query_embedding:
            logger.error("Failed to create query embedding")
            return RetrievalResult(
                documents=[],
                scores=[],
                metadata=[],
                total_found=0,
                filtered_count=0
            )
        
        # Search index (retrieve more than needed for filtering)
        initial_k = top_k * 3 if min_score > 0 else top_k
        results = self.search_service.search(
            query_vector=query_embedding,
            top_k=initial_k,
            filters=filters
        )
        
        # Convert to Documents and filter by score
        documents = []
        scores = []
        metadata_list = []
        total_length = 0
        filtered_count = 0
        
        # Track scores for relaxed threshold retry
        all_scores = []
        
        for result in results:
            score = result.get("@search.score", 0.0)
            all_scores.append(score)
            
            # Filter by score threshold - but be more lenient if no results
            if score < min_score:
                filtered_count += 1
                continue
            
            # Check context length limit
            content = result.get("content", "")
            if total_length + len(content) > max_context_length:
                filtered_count += 1
                continue
            
            # Build document metadata using helper method
            doc_metadata = self._build_document_metadata(result, score)
            doc = Document(page_content=content, metadata=doc_metadata)
            
            documents.append(doc)
            scores.append(score)
            metadata_list.append(doc_metadata)
            total_length += len(content)
            
            # Stop if we have enough documents
            if len(documents) >= top_k:
                break
        
        # If no documents after filtering, try with lower threshold
        if len(documents) == 0 and len(results) > 0 and all_scores:
            max_score = max(all_scores)
            # Try with 50% of max score or 0.3, whichever is higher
            relaxed_threshold = max(0.3, max_score * 0.5)
            
            # Retry with relaxed threshold
            for result in results:
                score = result.get("@search.score", 0.0)
                if score >= relaxed_threshold:
                    content = result.get("content", "")
                    if total_length + len(content) <= max_context_length:
                        doc_metadata = self._build_document_metadata(result, score)
                        doc = Document(page_content=content, metadata=doc_metadata)
                        
                        documents.append(doc)
                        scores.append(score)
                        metadata_list.append(doc_metadata)
                        total_length += len(content)
                        
                        if len(documents) >= top_k:
                            break
        
        # Optional reranking (placeholder for future implementation)
        if self.enable_reranking and len(documents) > 1:
            documents, scores = self._rerank_documents(query, documents, scores)
        
        return RetrievalResult(
            documents=documents,
            scores=scores,
            metadata=metadata_list,
            total_found=len(results),
            filtered_count=filtered_count
        )
    
    def _build_document_metadata(self, result: Dict[str, Any], score: float) -> Dict[str, Any]:
        """
        Build document metadata from search result.
        Maps index fields to Document metadata structure.
        
        Args:
            result: Search result dictionary from Azure AI Search
            score: Relevance score
            
        Returns:
            Dictionary with all metadata fields matching index schema
        """
        source_doc = result.get("source_document") or result.get("document_name", "")
        
        return {
            # Core fields
            'id': result.get("id"),
            'document_name': result.get("document_name") or source_doc,
            'source_document': source_doc,
            'document_title': result.get("document_title"),
            'page_number': result.get("page_number"),
            'chunk_index': result.get("chunk_index"),
            'token_count': result.get("token_count"),
            'score': score,
            
            # Legal hierarchy fields (from index schema)
            'legal_part_name': result.get("legal_part_name"),
            'legal_chapter_name': result.get("legal_chapter_name"),
            'article_reference': result.get("article_reference"),
            'article_number': result.get("article_number"),
            'paragraph_number': result.get("paragraph_number"),
            'clause_number': result.get("clause_number"),
            
            # Operational fields
            'procedure_name': result.get("procedure_name"),
            'procedure_step': result.get("procedure_step"),
            'policy_name': result.get("policy_name"),
            'annex_name': result.get("annex_name"),
            
            # Cadre classification fields
            'rank': result.get("rank"),
            'grade': result.get("grade"),
            'category_class': result.get("category_class"),
            'group': result.get("group"),
            'cadre_classification': result.get("cadre_classification"),
            
            # Classification fields
            'category': result.get("category"),
            'target_audience': result.get("target_audience"),
            'keywords': result.get("keywords"),
            
            # Metadata fields (for prompt builder compatibility)
            'metadata_item_number': result.get("metadata_item_number"),
            'metadata_item_type': result.get("metadata_item_type"),
            'metadata_item_title': result.get("metadata_item_title"),
            'metadata_section_title': result.get("metadata_section_title"),
            'metadata_resource_path': result.get("metadata_resource_path"),
            'metadata_source_file': source_doc,  # Backward compatibility
        }
    
    def _rerank_documents(
        self,
        query: str,
        documents: List[Document],
        scores: List[float]
    ) -> tuple[List[Document], List[float]]:
        """
        Rerank documents using cross-encoder or semantic similarity.
        
        This is a placeholder for future reranking implementation.
        For now, returns documents as-is.
        
        Future implementation could use:
        - Cross-encoder models (e.g., ms-marco-MiniLM)
        - Semantic similarity with query embedding
        - Hybrid scoring combining vector similarity + keyword matching
        """
        # Placeholder: return as-is
        # TODO: Implement reranking with cross-encoder model
        return documents, scores
    
    # Backward compatibility method
    def retrieve_legacy(self, query: str, top_k: int = 5, filters: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Legacy method for backward compatibility.
        Returns raw list format.
        """
        result = self.retrieve(query, top_k=top_k, filters=filters)
        
        formatted_results = []
        for doc, score, metadata in zip(result.documents, result.scores, result.metadata):
            formatted_results.append({
                "id": metadata.get("id"),
                "content": doc.page_content,
                "document_name": metadata.get("document_name", ""),
                "source_document": metadata.get("source_document"),
                "document_title": metadata.get("document_title"),
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index"),
                "token_count": metadata.get("token_count"),
                "score": score,
                # Legal hierarchy
                "legal_part_name": metadata.get("legal_part_name"),
                "legal_chapter_name": metadata.get("legal_chapter_name"),
                "article_reference": metadata.get("article_reference"),
                "article_number": metadata.get("article_number"),
                "paragraph_number": metadata.get("paragraph_number"),
                "clause_number": metadata.get("clause_number"),
                # Operational
                "procedure_name": metadata.get("procedure_name"),
                "procedure_step": metadata.get("procedure_step"),
                "policy_name": metadata.get("policy_name"),
                "annex_name": metadata.get("annex_name"),
                # Cadre
                "rank": metadata.get("rank"),
                "grade": metadata.get("grade"),
                "category_class": metadata.get("category_class"),
                "group": metadata.get("group"),
                "cadre_classification": metadata.get("cadre_classification"),
                # Classification
                "category": metadata.get("category"),
                "target_audience": metadata.get("target_audience"),
                # Metadata fields
                "metadata_item_number": metadata.get("metadata_item_number"),
                "metadata_item_type": metadata.get("metadata_item_type"),
                "metadata_item_title": metadata.get("metadata_item_title"),
                "metadata_section_title": metadata.get("metadata_section_title"),
                "metadata_resource_path": metadata.get("metadata_resource_path"),
                "metadata_source_file": metadata.get("metadata_source_file"),
            })
        
        return formatted_results
